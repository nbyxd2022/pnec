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

void Run() {
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
  std::vector<double> traces(count);
  double total = 0;
  for (int i = 0; i < count; ++i) {
    ReadMatrix(left[i]);
    ReadMatrix(right[i]);
    ReadMatrix(host[i]);
    ReadMatrix(nominal[i]);
    if (!(std::cin >> traces[i]) || !std::isfinite(traces[i]) || traces[i] < 0) {
      throw std::runtime_error("Invalid target trace");
    }
    total += traces[i];
  }
  const double mean = total / count;
  const auto solve = [&](const char *label, double scale, bool uniform) {
    auto target = nominal;
    for (int i = 0; i < count; ++i) {
      const double trace = scale * (uniform ? mean : traces[i]);
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
  solve("symmetric_analytic", 0, false);
  solve("symmetric_uniform", 1, true);
  solve("symmetric_per_match", 1, false);
  solve("weak_uniform", weak_scale, true);
  solve("weak_per_match", weak_scale, false);
}
} // namespace

int main() {
  try {
    Run();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
