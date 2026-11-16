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
  host.fill(0);
  target.fill(0);
  if (!check(Measure(h, t, 32, 31, 30, 29), 1, 4, 0)) {
    std::cerr << "Photometric search tie order differs from the Python reference\n";
    return 1;
  }
  std::cout << "Photometric search reference cases passed\n";
}
