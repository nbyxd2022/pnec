#include "photometric_increment.h"

#include <iostream>

using pnec::rel_pose_estimation::photometric::Ambiguity;
using pnec::rel_pose_estimation::photometric::GrayImage;
using pnec::rel_pose_estimation::photometric::Measure;

int main() {
  constexpr int width = 63, height = 61;
  std::array<std::uint8_t, width * height> host, target;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      host[y * width + x] = (11 * x + 17 * y + 3 * x * y + x * x) % 251;
      target[y * width + x] = (21 * x + 13 * y + 5 * x * y + y * y) % 253;
    }
  }
  const GrayImage h{host.data(), width, height, width};
  const GrayImage t{target.data(), width, height, width};
  const auto check = [](const Ambiguity &actual, double ratio, int dx, int dy) {
    return std::abs(actual.ratio - ratio) < 1e-12 &&
           actual.dx == dx && actual.dy == dy;
  };
  // Reference values come from the float64 Python lattice and minimum filter.
  if (!check(Measure(h, t, 32.375, 31.625, 30.25, 29.375),
             0.9767712448235378, -2, 7) ||
      !check(Measure(h, t, 8.25, 12.125, 9.5, 11.625),
             0.9874917362566418, -12, -13) ||
      !check(Measure(h, t, 8.25, 12.125, 5.5, 11.625), 0, 0, 0)) {
    std::cerr << "Photometric search differs from the Python reference\n";
    return 1;
  }
  pnec::rel_pose_estimation::photometric::Measurement detail;
  const auto measured = Measure(h, t, 32.375, 31.625, 30.25, 29.375, &detail);
  const auto original = Measure(h, t, 32.375, 31.625, 30.25, 29.375);
  if (measured.ratio != original.ratio || measured.dx != original.dx ||
      measured.dy != original.dy || !std::isfinite(detail.best_cost) ||
      !std::isfinite(detail.second_cost) ||
      detail.best_cost > detail.second_cost ||
      detail.best_cost > detail.endpoint_cost ||
      std::abs(detail.best_dx) > 16 || std::abs(detail.best_dy) > 16) {
    std::cerr << "Optional search measurements changed the result\n";
    return 1;
  }
  Measure(h, t, 8.25, 12.125, 5.5, 11.625, &detail);
  if (std::isfinite(detail.best_cost) || std::isfinite(detail.endpoint_cost)) {
    std::cerr << "Unsupported patch retained stale measurements\n";
    return 1;
  }
  host.fill(0);
  target.fill(0);
  if (!check(Measure(h, t, 32, 31, 30, 29), 1, 4, 0)) {
    std::cerr << "Photometric search tie order differs from the Python reference\n";
    return 1;
  }
  using namespace pnec::rel_pose_estimation::photometric;
  detail.best_cost = 1;
  detail.second_cost = 2;
  detail.best_dx = 2;
  const double far = FarPeakRetention(detail, {0.5, 14, 0});
  const double boundary = FarPeakRetention(detail, {0.5, 10, 0});
  detail.best_dx = 3;
  const double off_endpoint = FarPeakRetention(detail, {0.5, 14, 0});
  detail.best_dx = 0;
  detail.second_cost = std::numeric_limits<double>::infinity();
  if (far != 0.5625 || boundary != 1 || off_endpoint != 1 ||
      FarPeakRetention(detail, {0.5, 16, 0}) != 1) {
    std::cerr << "Far-peak retention differs from the fixed-input rule\n";
    return 1;
  }
  std::vector<double> weak{4, 2, 0}, tapered = weak, matched = weak;
  const std::vector<double> retention{far, 1, 1};
  const double weak_total = ApplyWeakIncrement(Mode::WeakPerMatch, {}, 6, weak);
  const double tapered_total = ApplyWeakIncrement(Mode::FarPeakTaper, retention, 6, tapered);
  const double matched_total = ApplyWeakIncrement(Mode::MatchedBudget, retention, 6, matched);
  std::vector<double> zero(3, 0);
  if (weak_total != 6.0 / 32 || tapered_total != 4.25 / 32 ||
      std::abs(tapered_total - matched_total) > 1e-15 ||
      tapered[0] != 2.25 / 32 || tapered[1] != 2.0 / 32 ||
      matched[0] != 2 * matched[1] || matched[2] != 0 ||
      ApplyWeakIncrement(Mode::MatchedBudget, retention, 0, zero) != 0) {
    std::cerr << "Target traces differ from the weak or equal-total control\n";
    return 1;
  }
  std::cout << "Photometric search and weak increment reference cases passed\n";
}
