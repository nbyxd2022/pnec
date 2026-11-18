// Re-solve one captured frame pair with the native symmetric PNEC objective.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pnec_ceres.h"

namespace {
template <typename Matrix>
void ReadMatrix(Matrix &matrix) {
  for (int row = 0; row < matrix.rows(); ++row) {
    for (int col = 0; col < matrix.cols(); ++col) {
      if (!(std::cin >> matrix(row, col)) || !std::isfinite(matrix(row, col))) {
        throw std::runtime_error("Missing or invalid matrix entry");
      }
    }
  }
}

void Run(bool secondary_peak) {
  int count;
  double regularizer, weak_scale;
  if (!(std::cin >> count >> regularizer >> weak_scale) || count <= 0 ||
      !std::isfinite(regularizer) || regularizer <= 0 ||
      !(weak_scale > 0 && weak_scale < 1)) {
    throw std::runtime_error("Invalid pair header");
  }
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  ReadMatrix(rotation);
  ReadMatrix(translation);
  std::vector<Eigen::Vector3d> left(count), right(count);
  std::vector<Eigen::Matrix3d> host(count), nominal(count);
  std::vector<double> traces(count), tapered, matched;
  if (secondary_peak) {
    tapered.resize(count);
    matched.resize(count);
  }
  double total = 0;
  for (int i = 0; i < count; ++i) {
    ReadMatrix(left[i]);
    ReadMatrix(right[i]);
    ReadMatrix(host[i]);
    ReadMatrix(nominal[i]);
    if (!(std::cin >> traces[i]) || !std::isfinite(traces[i]) || traces[i] < 0) {
      throw std::runtime_error("Invalid target trace");
    }
    if (secondary_peak &&
        (!(std::cin >> tapered[i] >> matched[i]) || !std::isfinite(tapered[i]) ||
         !std::isfinite(matched[i]) || tapered[i] < 0 || matched[i] < 0)) {
      throw std::runtime_error("Invalid secondary-peak arm traces");
    }
    total += traces[i];
  }
  const auto solve = [&](const char *label, double scale,
                         const std::vector<double> &arm_traces) {
    auto target = nominal;
    for (int i = 0; i < count; ++i) {
      const double trace = scale * arm_traces[i];
      target[i] += (trace / 2) *
          (Eigen::Matrix3d::Identity() - right[i] * right[i].transpose());
    }
    // The capture stores the very rotationMatrix() used by PNEC::CeresSolver
    // to initialize its quaternion, in row-major text with 17-digit precision.
    pnec::optimization::PNECCeres optimizer;
    optimizer.InitValues(Eigen::Quaterniond(rotation), translation);
    optimizer.Optimize(left, right, target, regularizer, pnec::common::Target, &host);
    if (!optimizer.IsSolutionUsable()) {
      throw std::runtime_error(std::string("Unusable Ceres solution for ") + label);
    }
    const Eigen::Matrix3d result = optimizer.Result().rotationMatrix();
    std::cout << label;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) std::cout << ' ' << result(r, c);
    std::cout << '\n';
  };
  std::cout << std::setprecision(17);
  solve("symmetric_analytic", 0, traces);
  if (secondary_peak) {
    solve("weak_per_match", weak_scale, traces);
    solve("matched_budget", weak_scale, matched);
    solve("far_peak_taper", weak_scale, tapered);
    return;
  }
  const std::vector<double> uniform(count, total / count);
  solve("symmetric_uniform", 1, uniform);
  solve("symmetric_per_match", 1, traces);
  solve("weak_uniform", weak_scale, uniform);
  solve("weak_per_match", weak_scale, traces);
}
} // namespace

int main(int argc, char **argv) {
  const bool secondary_peak = argc == 2 && std::string(argv[1]) == "--secondary-peak";
  if (argc != 1 && !secondary_peak) {
    std::cerr << "Usage: pnec_fixed_refine [--secondary-peak]\n";
    return 1;
  }
  try {
    Run(secondary_peak);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
