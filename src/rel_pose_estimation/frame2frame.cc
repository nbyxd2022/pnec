/**
 BSD 3-Clause License

 This file is part of the PNEC project.
 https://github.com/tum-vision/pnec

 Copyright (c) 2022, Dominik Muhle.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "frame2frame.h"

#include "math.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <string_view>
#include <opencv2/core/eigen.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>
#include <opengv/types.hpp>

#include "common.h"
#include "essential_matrix_methods.h"
#include "odometry_output.h"
#include "photometric_increment.h"
#include "pnec.h"

namespace pnec {
namespace rel_pose_estimation {
namespace {

std::set<int> DiagnosticFrames(const char *variable) {
  std::set<int> selected;
  const char *setting = std::getenv(variable);
  if (!setting || !*setting) {
    return selected;
  }
  std::string_view rest(setting);
  while (true) {
    const auto comma = rest.find(',');
    const auto token = rest.substr(0, comma);
    int frame = 0;
    const auto end = token.data() + token.size();
    const auto result = std::from_chars(token.data(), end, frame);
    if (result.ec != std::errc() || result.ptr != end || frame < 0) {
      throw std::invalid_argument(std::string(variable) +
                                  " expects comma-separated frame IDs");
    }
    selected.insert(frame);
    if (comma == std::string_view::npos) {
      return selected;
    }
    rest.remove_prefix(comma + 1);
  }
}

bool MatchDiagnosticsEnabled(int target_frame) {
  static const auto frames = DiagnosticFrames("PNEC_MATCH_DIAGNOSTICS_FRAMES");
  return frames.count(target_frame) != 0;
}

bool RefinementDiagnosticsEnabled(int target_frame) {
  static const auto frames = DiagnosticFrames("PNEC_REFINEMENT_DIAGNOSTICS_FRAMES");
  return frames.count(target_frame) != 0;
}

std::set<int> OrderedKltIds(const FeatureMatches &matches, size_t count) {
  // GetFeatures assembles the solver inputs from maps sorted by keypoint ID.
  // The KLT matcher pairs the same ID across frames. Check this before mapping
  // RANSAC indices back to pixel coordinates.
  std::set<int> feature_ids;
  for (const auto &match : matches) {
    if (match.queryIdx < 0 || match.queryIdx != match.trainIdx ||
        !feature_ids.insert(match.queryIdx).second) {
      throw std::runtime_error("Expected unique KLT IDs shared by both frames");
    }
  }
  if (feature_ids.size() != count) {
    throw std::runtime_error("KLT IDs and covariance count mismatch");
  }
  return feature_ids;
}

struct PhotometricTraces {
  std::vector<double> values;
  double total = 0;
};

PhotometricTraces MeasurePhotometricTraces(
    frames::BaseFrame &host, frames::BaseFrame &target,
    const FeatureMatches &matches, const opengv::bearingVectors_t &target_bvs,
    const std::vector<int> &inliers) {
  const auto sorted_ids = OrderedKltIds(matches, target_bvs.size());
  const std::vector<int> ids(sorted_ids.begin(), sorted_ids.end());
  const cv::Mat image1 = host.getImage(), image2 = target.getImage();
  const photometric::GrayImage gray1{image1.ptr<std::uint8_t>(), image1.cols,
                                     image1.rows, image1.step[0]};
  const photometric::GrayImage gray2{image2.ptr<std::uint8_t>(), image2.cols,
                                     image2.rows, image2.step[0]};
  Eigen::Matrix3d intrinsic;
  cv::cv2eigen(Camera::instance().cameraParameters().intrinsic(), intrinsic);
  const Eigen::Matrix3d inverse_intrinsic = intrinsic.inverse();
  PhotometricTraces traces;
  traces.values.reserve(inliers.size());
  for (int index : inliers) {
    const auto &host_point = host.keypoints().at(ids.at(index)).point_;
    const auto &target_point = target.keypoints().at(ids.at(index)).point_;
    const auto ambiguity =
        photometric::Measure(gray1, gray2, host_point.x(), host_point.y(),
                             target_point.x(), target_point.y());
    const auto &bearing = target_bvs.at(index);
    const Eigen::Vector3d ray =
        inverse_intrinsic * Eigen::Vector3d(target_point.x(), target_point.y(), 1.0);
    const Eigen::Vector3d shift =
        inverse_intrinsic.leftCols<2>() * Eigen::Vector2d(ambiguity.dx, ambiguity.dy);
    const Eigen::Vector3d tangent_shift = shift - bearing * bearing.dot(shift);
    const double trace = ambiguity.ratio * tangent_shift.squaredNorm() /
                         (4 * ray.squaredNorm());
    traces.values.push_back(trace);
    traces.total += trace;
  }
  if (!std::isfinite(traces.total)) {
    throw std::runtime_error("Nonfinite photometric refinement increment");
  }
  return traces;
}

size_t AddPhotometricIncrements(
    const PhotometricTraces &traces, const opengv::bearingVectors_t &target_bvs,
    const std::vector<int> &inliers, std::vector<Eigen::Matrix3d> &covariances,
    photometric::Mode mode) {
  if (covariances.size() != inliers.size() || traces.values.size() != inliers.size()) {
    throw std::runtime_error("Invalid photometric refinement covariance input");
  }
  const double mean_trace = inliers.empty() ? 0 : traces.total / inliers.size();
  size_t nonzero = 0;
  for (size_t i = 0; i < covariances.size(); ++i) {
    const auto &bearing = target_bvs.at(inliers[i]);
    const double trace = mode == photometric::Mode::Uniform ? mean_trace : traces.values[i];
    covariances[i] += (trace / 2) *
        (Eigen::Matrix3d::Identity() - bearing * bearing.transpose());
    nonzero += trace > 0;
  }
  return nonzero;
}

void WritePoseFields(std::ostream &stream, const Sophus::SE3d &pose) {
  const auto &t = pose.translation();
  const auto q = pose.unit_quaternion();
  stream << ',' << t.x() << ',' << t.y() << ',' << t.z() << ','
         << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w();
}

void CompareRefinements(
    frames::BaseFrame &host, frames::BaseFrame &target, const FeatureMatches &matches,
    const opengv::bearingVectors_t &target_bvs, const std::vector<int> &inliers,
    const std::vector<Eigen::Matrix3d> &base_covariances, const Sophus::SE3d &initial,
    const RefinementResult &original, const CovarianceRefiner &refine,
    const std::string &folder) {
  const auto traces = MeasurePhotometricTraces(host, target, matches, target_bvs, inliers);
  const auto suffix = std::to_string(target.id()) + ".csv";
  const auto path = folder + "refinement_" + suffix;
  std::ofstream out(path);
  out << "host_frame,target_frame,host_time_s,target_time_s,inliers,mode,"
         "increment_trace,nonzero_increments,"
         "init_tx,init_ty,init_tz,init_qx,init_qy,init_qz,init_qw,"
         "tx,ty,tz,qx,qy,qz,qw,solution_usable,termination,initial_cost,final_cost\n";
  out << std::setprecision(17);
  const auto write_row = [&](const char *mode, const RefinementResult &result,
                             double trace, size_t nonzero) {
    out << host.id() << ',' << target.id() << ',' << host.Timestamp() << ','
        << target.Timestamp() << ',' << inliers.size() << ',' << mode << ','
        << trace << ',' << nonzero;
    WritePoseFields(out, initial);
    WritePoseFields(out, result.pose);
    out << ',' << result.status.usable << ',' << result.status.termination
        << ',' << result.status.initial_cost << ',' << result.status.final_cost << '\n';
  };
  write_row("original", original, 0, 0);
  for (auto mode : {photometric::Mode::Uniform, photometric::Mode::PerMatch}) {
    auto covariances = base_covariances;
    const size_t nonzero = AddPhotometricIncrements(traces, target_bvs, inliers, covariances, mode);
    write_row(mode == photometric::Mode::Uniform ? "uniform" : "per_match",
              refine(covariances), traces.total, nonzero);
  }
  out.close();
  if (!out) {
    throw std::runtime_error("Failed to write refinement comparison: " + path);
  }

  const auto increment_path = folder + "refinement_increments_" + suffix;
  std::ofstream points(increment_path);
  points << "solver_index,original_covariance_trace,per_match_increment_trace\n"
         << std::setprecision(17);
  for (size_t i = 0; i < inliers.size(); ++i) {
    points << inliers[i] << ',' << base_covariances[i].trace() << ','
           << traces.values[i] << '\n';
  }
  points.close();
  if (!points) {
    throw std::runtime_error("Failed to write refinement increments: " + increment_path);
  }
}

void WriteMatchDiagnostics(frames::BaseFrame &host, frames::BaseFrame &target,
                           const FeatureMatches &matches,
                           const std::vector<Eigen::Matrix3d> &projected_covs,
                           const std::vector<int> &inliers,
                           const Options &options, const std::string &folder) {
  const auto feature_ids = OrderedKltIds(matches, projected_covs.size());
  std::vector<bool> ransac_inlier(feature_ids.size(), false);
  for (const auto index : inliers) {
    ransac_inlier.at(index) = true;
  }

  const auto path = folder + "matches_" + std::to_string(host.id()) + "_" +
                    std::to_string(target.id()) + ".csv";
  std::ofstream out(path);
  out << "host_frame,target_frame,host_time_s,target_time_s,solver_index,"
         "feature_id,host_x_px,host_y_px,target_x_px,target_y_px,"
         "ransac_enabled,ransac_inlier,noise_frame,"
         "host_cov00,host_cov01,host_cov10,host_cov11,"
         "target_cov00,target_cov01,target_cov10,target_cov11,"
         "projected_cov00,projected_cov01,projected_cov02,"
         "projected_cov10,projected_cov11,projected_cov12,"
         "projected_cov20,projected_cov21,projected_cov22\n";
  out << std::setprecision(17);
  size_t index = 0;
  for (const auto id : feature_ids) {
    const auto &h = host.keypoints().at(id);
    const auto &t = target.keypoints().at(id);
    out << host.id() << ',' << target.id() << ',' << host.Timestamp() << ','
        << target.Timestamp() << ',' << index << ',' << id << ',' << h.point_.x()
        << ',' << h.point_.y() << ',' << t.point_.x() << ',' << t.point_.y()
        << ',' << options.use_ransac_ << ',' << ransac_inlier[index] << ','
        << (options.noise_frame_ == common::Host ? "host" : "target");
    for (const auto *cov : {&h.img_covariance_, &t.img_covariance_}) {
      out << ',' << (*cov)(0, 0) << ',' << (*cov)(0, 1) << ',' << (*cov)(1, 0)
          << ',' << (*cov)(1, 1);
    }
    const auto &cov = projected_covs[index];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        out << ',' << cov(row, col);
      }
    }
    out << '\n';
    ++index;
  }
  out.close();
  if (!out) {
    throw std::runtime_error("Failed to write match diagnostics: " + path);
  }
}

} // namespace

Frame2Frame::Frame2Frame(const pnec::rel_pose_estimation::Options &options)
    : options_{options} {}
Frame2Frame::~Frame2Frame() {}

Sophus::SE3d Frame2Frame::Align(pnec::frames::BaseFrame::Ptr frame1,
                                pnec::frames::BaseFrame::Ptr frame2,
                                pnec::FeatureMatches &matches,
                                Sophus::SE3d prev_rel_pose,
                                std::vector<int> &inliers,
                                pnec::common::FrameTiming &frame_timing,
                                bool ablation, std::string ablation_folder) {
  if (ablation) {
    if (!boost::filesystem::exists(ablation_folder)) {
      boost::filesystem::create_directory(ablation_folder);
    }
  }

  curr_timestamp_ = frame2->Timestamp();

  opengv::bearingVectors_t bvs1;
  opengv::bearingVectors_t bvs2;
  std::vector<Eigen::Matrix3d> proj_covs;
  GetFeatures(frame1, frame2, matches, bvs1, bvs2, proj_covs);

  if (ablation) {
    AblationAlign(bvs1, bvs2, proj_covs, ablation_folder);
  }

  const auto mode = photometric::ExperimentMode();
  const bool active = mode == photometric::Mode::Uniform || mode == photometric::Mode::PerMatch;
  const bool compare = RefinementDiagnosticsEnabled(frame2->id());
  RefinementCovarianceUpdate update;
  RefinementObserver observer;
  if ((active || compare) && (options_.use_nec_ || !options_.use_ransac_ ||
                              !options_.use_ceres_ || options_.noise_frame_ != common::Target)) {
    throw std::invalid_argument(
        "Photometric increments require target-side PNEC with RANSAC and Ceres");
  }
  if (active) {
    update = [&](auto &covariances, const auto &indices) {
      const auto start = std::chrono::steady_clock::now();
      const auto traces = MeasurePhotometricTraces(*frame1, *frame2, matches, bvs2, indices);
      frame_timing.increment_nonzero_ =
          AddPhotometricIncrements(traces, bvs2, indices, covariances, mode);
      frame_timing.increment_inliers_ = indices.size();
      frame_timing.increment_trace_ = traces.total;
      frame_timing.covariance_increment_ = std::chrono::steady_clock::now() - start;
    };
  }
  if (compare) {
    if (!ablation || mode != photometric::Mode::Original) {
      throw std::invalid_argument("Refinement comparison requires original experiment mode and an output folder");
    }
    observer = [&](const auto &covariances, const auto &indices, const auto &initial,
                   const auto &original, const auto &refine) {
      CompareRefinements(*frame1, *frame2, matches, bvs2, indices, covariances,
                         initial, original, refine, ablation_folder);
    };
  }
  const auto rel_pose = PNECAlign(bvs1, bvs2, proj_covs, prev_rel_pose, inliers,
                                  frame_timing, update, observer);
  if (ablation && MatchDiagnosticsEnabled(frame2->id())) {
    WriteMatchDiagnostics(*frame1, *frame2, matches, proj_covs, inliers,
                          options_, ablation_folder);
  }
  return rel_pose;
}

Sophus::SE3d Frame2Frame::AlignFurther(pnec::frames::BaseFrame::Ptr frame1,
                                       pnec::frames::BaseFrame::Ptr frame2,
                                       pnec::FeatureMatches &matches,
                                       Sophus::SE3d prev_rel_pose,
                                       std::vector<int> &inliers,
                                       bool &success) {
  opengv::bearingVectors_t bvs1;
  opengv::bearingVectors_t bvs2;
  std::vector<Eigen::Matrix3d> proj_covs;
  GetFeatures(frame1, frame2, matches, bvs1, bvs2, proj_covs);

  pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  Sophus::SE3d rel_pose =
      PNECAlign(bvs1, bvs2, proj_covs, prev_rel_pose, inliers, dummy_timing);

  if (ransac_iterations_ >= options_.max_ransac_iterations_) {
    success = false;
  } else if (inliers.size() < options_.min_inliers_ && inliers.size() != 0) {
    success = false;
  } else {
    success = true;
  }
  return rel_pose;
}

const std::map<std::string, std::vector<std::pair<double, Sophus::SE3d>>> &
Frame2Frame::GetAblationResults() const {
  return ablation_rel_poses_;
}

Sophus::SE3d Frame2Frame::PNECAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    Sophus::SE3d prev_rel_pose, std::vector<int> &inliers,
    pnec::common::FrameTiming &frame_timing, const RefinementCovarianceUpdate &update,
    const RefinementObserver &observer) {
  pnec::rel_pose_estimation::PNEC pnec(options_);
  Sophus::SE3d rel_pose = pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose,
                                     inliers, frame_timing, update, observer);

  if (ablation_rel_poses_.count("PNEC") == 0) {
    ablation_rel_poses_["PNEC"] = {
        std::make_pair<double, Sophus::SE3d>(0.0, Sophus::SE3d())};
  }
  ablation_rel_poses_["PNEC"].push_back(
      std::make_pair(curr_timestamp_, rel_pose));
  return rel_pose;
}

void Frame2Frame::AblationAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    std::string ablation_folder) {

  auto GetPrevRelPose = [this](std::string method_name) -> Sophus::SE3d {
    Sophus::SE3d prev_rel_pose;
    if (ablation_rel_poses_.count(method_name) == 0) {
      ablation_rel_poses_[method_name] = {
          std::make_pair<double, Sophus::SE3d>(0.0, Sophus::SE3d())};
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = ablation_rel_poses_[method_name].back().second;
    }
    return prev_rel_pose;
  };

  {
    std::string name = "NEC";

    Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;
    nec_options.use_ceres_ = false;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    ablation_rel_poses_[name].push_back(
        std::make_pair(curr_timestamp_, rel_pose));
  }

  {
    std::string name = "NEC-LS";

    Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    ablation_rel_poses_[name].push_back(
        std::make_pair(curr_timestamp_, rel_pose));
  }

  // {
  //   std::string name = "NEC+PNEC-LS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 1;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNECwoLS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.use_ceres_ = false;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNEConlyLS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 0;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNEConlyLSfull";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 0;
  //   options.ceres_options_.max_num_iterations = 10000;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::vector<size_t> iterations = {5, 15};

  //   for (const auto &max_it : iterations) {
  //     {
  //       std::string name = "PNEC-" + std::to_string(max_it);

  //       Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //       pnec::rel_pose_estimation::Options options = options_;
  //       options.weighted_iterations_ = max_it;

  //       pnec::rel_pose_estimation::PNEC pnec(options);
  //       pnec::common::FrameTiming dummy_timing =
  //       pnec::common::FrameTiming(0); Sophus::SE3d rel_pose =
  //           pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose,
  //           dummy_timing);

  //       ablation_rel_poses_[name].push_back(
  //           std::make_pair(curr_timestamp_, rel_pose));
  //       pnec::out::SavePose(ablation_folder, name, curr_timestamp_,
  //       rel_pose);
  //     }
  //   }
  // }

  // {
  //   std::string name = "8pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           EIGHTPT);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "Nister5pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           NISTER);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "Stewenius5pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           STEWENIUS);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "7pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           SEVENPT);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }
}

void Frame2Frame::GetFeatures(pnec::frames::BaseFrame::Ptr host_frame,
                              pnec::frames::BaseFrame::Ptr target_frame,
                              pnec::FeatureMatches &matches,
                              opengv::bearingVectors_t &host_bvs,
                              opengv::bearingVectors_t &target_bvs,
                              std::vector<Eigen::Matrix3d> &proj_covs) {
  std::vector<size_t> host_matches;
  std::vector<size_t> target_matches;
  for (const auto &match : matches) {
    host_matches.push_back(match.queryIdx);
    target_matches.push_back(match.trainIdx);
  }
  pnec::features::KeyPoints host_keypoints =
      host_frame->keypoints(host_matches);
  pnec::features::KeyPoints target_keypoints =
      target_frame->keypoints(target_matches);

  std::vector<Eigen::Matrix3d> host_covs;
  std::vector<Eigen::Matrix3d> target_covs;
  for (auto const &[id, keypoint] : host_keypoints) {
    host_bvs.push_back(keypoint.bearing_vector_);
    host_covs.push_back(keypoint.bv_covariance_);
  }
  for (auto const &[id, keypoint] : target_keypoints) {
    target_bvs.push_back(keypoint.bearing_vector_);
    target_covs.push_back(keypoint.bv_covariance_);
  }

  if (options_.noise_frame_ == pnec::common::Host) {
    proj_covs = host_covs;
  } else {
    proj_covs = target_covs;
  }
}

} // namespace rel_pose_estimation
} // namespace pnec
