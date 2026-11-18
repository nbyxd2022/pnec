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

bool MatchDiagnosticsEnabled(int target_frame) {
  static const auto frames = [] {
    std::set<int> selected;
    const char *setting = std::getenv("PNEC_MATCH_DIAGNOSTICS_FRAMES");
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
        throw std::invalid_argument(
            "PNEC_MATCH_DIAGNOSTICS_FRAMES expects comma-separated frame IDs");
      }
      selected.insert(frame);
      if (comma == std::string_view::npos) {
        return selected;
      }
      rest.remove_prefix(comma + 1);
    }
  }();
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

struct IncrementAuditPoint {
  photometric::Ambiguity ambiguity;
  photometric::Measurement measurement;
  double trace;
};

void AddPhotometricIncrements(
    frames::BaseFrame &host, frames::BaseFrame &target,
    const FeatureMatches &matches, const opengv::bearingVectors_t &target_bvs,
    const std::vector<int> &inliers, std::vector<Eigen::Matrix3d> &covariances,
    photometric::Mode mode, common::FrameTiming &timing,
    std::vector<IncrementAuditPoint> *audit = nullptr) {
  const auto start = std::chrono::steady_clock::now();
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
  std::vector<double> traces;
  traces.reserve(inliers.size());
  double total_trace = 0;
  for (int index : inliers) {
    const auto &host_point = host.keypoints().at(ids.at(index)).point_;
    const auto &target_point = target.keypoints().at(ids.at(index)).point_;
    photometric::Measurement measurement;
    const auto ambiguity =
        photometric::Measure(gray1, gray2, host_point.x(), host_point.y(),
                             target_point.x(), target_point.y(),
                             audit ? &measurement : nullptr);
    const auto &bearing = target_bvs.at(index);
    const Eigen::Vector3d ray =
        inverse_intrinsic * Eigen::Vector3d(target_point.x(), target_point.y(), 1.0);
    const Eigen::Vector3d shift =
        inverse_intrinsic.leftCols<2>() * Eigen::Vector2d(ambiguity.dx, ambiguity.dy);
    const Eigen::Vector3d tangent_shift = shift - bearing * bearing.dot(shift);
    const double trace = ambiguity.ratio * tangent_shift.squaredNorm() /
                         (4 * ray.squaredNorm());
    traces.push_back(trace);
    total_trace += trace;
    if (audit) {
      audit->push_back({ambiguity, measurement, trace});
    }
  }
  if (!std::isfinite(total_trace) || covariances.size() != inliers.size()) {
    throw std::runtime_error("Invalid photometric refinement covariance input");
  }
  const double mean_trace = traces.empty() ? 0 : total_trace / traces.size();
  for (size_t i = 0; i < covariances.size(); ++i) {
    const auto &bearing = target_bvs.at(inliers[i]);
    const bool uniform = mode == photometric::Mode::Uniform ||
                         mode == photometric::Mode::SymmetricUniform;
    const double trace = uniform ? mean_trace : traces[i];
    covariances[i] += (trace / 2) *
        (Eigen::Matrix3d::Identity() - bearing * bearing.transpose());
    timing.increment_nonzero_ += trace > 0;
  }
  timing.increment_inliers_ = inliers.size();
  timing.increment_trace_ = total_trace;
  timing.covariance_increment_ = std::chrono::steady_clock::now() - start;
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


void WriteFixedInputAudit(
    frames::BaseFrame &host, frames::BaseFrame &target,
    const FeatureMatches &matches, const opengv::bearingVectors_t &all_target_bvs,
    const opengv::bearingVectors_t &left, const opengv::bearingVectors_t &right,
    const std::vector<Eigen::Matrix3d> &nominal,
    const std::vector<Eigen::Matrix3d> *source_covariances,
    const std::vector<int> &indices, const Sophus::SE3d &init,
    const Sophus::SE3d &baseline, const Options &options,
    const std::string &folder) {
  if (!source_covariances || source_covariances->size() != nominal.size() ||
      nominal.empty() || left.size() != nominal.size() ||
      right.size() != nominal.size() || indices.size() != nominal.size()) {
    throw std::runtime_error("Incomplete fixed-input symmetric refinement inputs");
  }
  auto per_match = nominal;
  std::vector<IncrementAuditPoint> measurements;
  measurements.reserve(nominal.size());
  common::FrameTiming diagnostic_timing(0);
  AddPhotometricIncrements(host, target, matches, all_target_bvs, indices,
                           per_match, photometric::Mode::SymmetricPerMatch,
                           diagnostic_timing, &measurements);
  double mean_trace = 0;
  for (const auto &point : measurements) {
    mean_trace += point.trace;
  }
  mean_trace /= measurements.size();
  auto uniform = nominal;
  for (size_t i = 0; i < uniform.size(); ++i) {
    uniform[i] += (mean_trace / 2) *
        (Eigen::Matrix3d::Identity() - right[i] * right[i].transpose());
  }
  PNEC solver(options);
  const auto uniform_pose = solver.CeresSolver(left, right, uniform, init,
                                              source_covariances);
  const auto per_match_pose = solver.CeresSolver(left, right, per_match, init,
                                                source_covariances);
  const auto prefix = folder + "fixed_" + std::to_string(host.id()) + "_" +
                      std::to_string(target.id());
  if (boost::filesystem::exists(prefix + "_points.csv") ||
      boost::filesystem::exists(prefix + "_poses.csv")) {
    throw std::runtime_error("Fixed-input audit output already exists: " + prefix);
  }
  const auto sorted = OrderedKltIds(matches, all_target_bvs.size());
  const std::vector<int> ids(sorted.begin(), sorted.end());
  std::ofstream points(prefix + "_points.csv");
  points << "solver_index,feature_id,ratio,delta_trace,mean_trace,best_cost,"
            "second_cost,endpoint_cost,best_dx,best_dy,mode_dx,mode_dy,regularization";
  for (const auto *name : {"host_b", "target_b"}) {
    for (int k = 0; k < 3; ++k) points << ',' << name << k;
  }
  for (const auto *name : {"host_cov", "target_cov"}) {
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) points << ',' << name << r << c;
  }
  points << '\n' << std::setprecision(17);
  for (size_t i = 0; i < nominal.size(); ++i) {
    const auto &point = measurements.at(i);
    const auto &measure = point.measurement;
    points << indices[i] << ',' << ids.at(indices[i]) << ',' << point.ambiguity.ratio
           << ',' << point.trace << ',' << mean_trace << ',' << measure.best_cost
           << ',' << measure.second_cost << ',' << measure.endpoint_cost << ','
           << measure.best_dx << ',' << measure.best_dy << ',' << point.ambiguity.dx
           << ',' << point.ambiguity.dy << ',' << options.regularization_;
    for (const auto *bearing : {&left[i], &right[i]})
      for (int k = 0; k < 3; ++k) points << ',' << (*bearing)(k);
    for (const auto *covariance : {&source_covariances->at(i), &nominal[i]})
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) points << ',' << (*covariance)(r, c);
    points << '\n';
  }
  points.close();
  std::ofstream poses(prefix + "_poses.csv");
  poses << "mode";
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) poses << ",R" << r << c;
  poses << ",t0,t1,t2\n" << std::setprecision(17);
  const auto write_pose = [&](const char *name, const Sophus::SE3d &pose) {
    poses << name;
    const auto rotation = pose.rotationMatrix();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) poses << ',' << rotation(r, c);
    for (int k = 0; k < 3; ++k) poses << ',' << pose.translation()(k);
    poses << '\n';
  };
  write_pose("initial", init);
  write_pose("symmetric_analytic", baseline);
  write_pose("symmetric_uniform", uniform_pose);
  write_pose("symmetric_per_match", per_match_pose);
  poses.close();
  if (!points || !poses) {
    throw std::runtime_error("Failed to save fixed-input audit: " + prefix);
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
  const auto mode = photometric::ExperimentMode();
  const bool symmetric = mode == photometric::Mode::SymmetricAnalytic ||
                         mode == photometric::Mode::SymmetricUniform ||
                         mode == photometric::Mode::SymmetricPerMatch;
  if (symmetric &&
      (options_.use_nec_ || !options_.use_ransac_ || !options_.use_ceres_ ||
       options_.noise_frame_ != common::Target)) {
    throw std::invalid_argument(
        "Symmetric analytic refinement requires target-side PNEC with RANSAC and Ceres");
  }
  std::vector<Eigen::Matrix3d> host_covariances;
  auto *host_covs = symmetric ? &host_covariances : nullptr;
  GetFeatures(frame1, frame2, matches, bvs1, bvs2, proj_covs, host_covs);

  if (ablation) {
    AblationAlign(bvs1, bvs2, proj_covs, ablation_folder);
  }

  RefinementCovarianceUpdate update;
  if (mode == photometric::Mode::Uniform || mode == photometric::Mode::PerMatch ||
      mode == photometric::Mode::SymmetricUniform ||
      mode == photometric::Mode::SymmetricPerMatch) {
    if (options_.use_nec_ || !options_.use_ransac_ || !options_.use_ceres_ ||
        options_.noise_frame_ != common::Target) {
      throw std::invalid_argument(
          "Photometric increments require target-side PNEC with RANSAC and Ceres");
    }
    update = [&](auto &covariances, const auto &indices) {
      AddPhotometricIncrements(*frame1, *frame2, matches, bvs2, indices,
                               covariances, mode, frame_timing);
    };
  }
  RefinementObserver observer;
  const char *audit_setting = std::getenv("PNEC_FIXED_INPUT_AUDIT");
  if (audit_setting && *audit_setting && std::string_view(audit_setting) != "1") {
    throw std::invalid_argument("PNEC_FIXED_INPUT_AUDIT expects 1 or unset");
  }
  if (audit_setting && *audit_setting) {
    if (!ablation || mode != photometric::Mode::SymmetricAnalytic) {
      throw std::invalid_argument("Fixed-input audit requires the symmetric analytic baseline with ablations");
    }
    if (MatchDiagnosticsEnabled(frame2->id())) {
      observer = [&](const auto &left, const auto &right, const auto &nominal,
                     const auto *source_covs, const auto &indices,
                     const auto &init, const auto &baseline) {
        WriteFixedInputAudit(*frame1, *frame2, matches, bvs2, left, right,
                             nominal, source_covs, indices, init, baseline,
                             options_, ablation_folder);
      };
    }
  }
  const auto rel_pose = PNECAlign(bvs1, bvs2, proj_covs, prev_rel_pose, inliers,
                                  frame_timing, update, host_covs, observer);
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
    pnec::common::FrameTiming &frame_timing,
    const RefinementCovarianceUpdate &update,
    const std::vector<Eigen::Matrix3d> *host_covariances,
    const RefinementObserver &observer) {
  pnec::rel_pose_estimation::PNEC pnec(options_);
  Sophus::SE3d rel_pose = pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose,
                                     inliers, frame_timing, update,
                                     host_covariances, observer);

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
                              std::vector<Eigen::Matrix3d> &proj_covs,
                              std::vector<Eigen::Matrix3d> *host_covariances) {
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
  if (host_covariances) {
    host_covariances->swap(host_covs);
  }
}

} // namespace rel_pose_estimation
} // namespace pnec
