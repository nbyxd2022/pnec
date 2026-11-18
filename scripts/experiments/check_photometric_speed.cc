#include "photometric_increment.h"
#include "photometric_increment_reference.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace current = pnec::rel_pose_estimation::photometric;
namespace reference = pnec::rel_pose_estimation::photometric_reference;

struct Point {
  double host_x, host_y, target_x, target_y;
};

int main() {
  constexpr int width = 83, height = 79, stride = 89;
  std::vector<std::uint8_t> host(stride * height), target(stride * height);
  const current::GrayImage h{host.data(), width, height, stride};
  const current::GrayImage t{target.data(), width, height, stride};
  const reference::GrayImage rh{host.data(), width, height, stride};
  const reference::GrayImage rt{target.data(), width, height, stride};
  std::mt19937 random(20261118);
  int checked = 0;

  const auto verify = [&](Point p) {
    const auto expected = reference::Measure(rh, rt, p.host_x, p.host_y, p.target_x, p.target_y);
    const auto actual = current::Measure(h, t, p.host_x, p.host_y, p.target_x, p.target_y);
    if (expected.dx != actual.dx || expected.dy != actual.dy ||
        std::memcmp(&expected.ratio, &actual.ratio, sizeof(double)) != 0) {
      std::cerr << std::setprecision(17)
                << "Mismatch at case " << checked << ": expected "
                << expected.ratio << ',' << expected.dx << ',' << expected.dy
                << ", actual " << actual.ratio << ',' << actual.dx << ','
                << actual.dy << '\n';
      return false;
    }
    ++checked;
    return true;
  };

  // Independent noise, repeated stripes, flat, checkerboard and identical images.
  for (int pattern = 0; pattern < 5; ++pattern) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int i = y * stride + x;
        switch (pattern) {
        case 0:
          host[i] = random() & 255;
          target[i] = random() & 255;
          break;
        case 1:
          host[i] = ((x % 8) * 29 + (y % 7) * 17) % 256;
          target[i] = (((x + 2) % 8) * 29 + ((y + 1) % 7) * 17) % 256;
          break;
        case 2:
          host[i] = target[i] = 71;
          break;
        case 3:
          host[i] = target[i] = ((x / 3 + y / 3) % 2) * 255;
          break;
        default:
          host[i] = target[i] = random() & 255;
          break;
        }
      }
    }
    for (int i = 0; i < 64; ++i) {
      const Point p{8.0 + random() % 65 + (random() % 8) / 8.0,
                    8.0 + random() % 61 + (random() % 8) / 8.0,
                    8.0 + random() % 65 + (random() % 8) / 8.0,
                    8.0 + random() % 61 + (random() % 8) / 8.0};
      if (!verify(p)) {
        return 1;
      }
    }
    for (Point p : {Point{7, 7, 7, 7}, Point{74, 70, 74, 70},
                    Point{30, 31, -1, 31}, Point{-1, 31, 30, 31},
                    Point{30, 31, 82, 31}, Point{30.125, 31.875, 29.25, 28.5}}) {
      if (!verify(p)) {
        return 1;
      }
    }
  }

  for (auto &value : host) {
    value = random() & 255;
  }
  for (auto &value : target) {
    value = random() & 255;
  }
  std::vector<double> old_samples, new_samples;
  volatile double checksum = 0;
  const auto measure = [&](bool optimized, int repeat) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 128; ++i) {
      const double x = 29 + i % 19 + repeat / 8.0;
      const double y = 28 + i % 17 + (repeat + 1) / 8.0;
      const double ratio = optimized
          ? current::Measure(h, t, x, y, x + 1.25, y + 0.75).ratio
          : reference::Measure(rh, rt, x, y, x + 1.25, y + 0.75).ratio;
      checksum += ratio;
    }
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - begin).count();
  };
  for (int repeat = 0; repeat < 5; ++repeat) {
    if (repeat % 2 == 0) {
      old_samples.push_back(measure(false, repeat));
      new_samples.push_back(measure(true, repeat));
    } else {
      new_samples.push_back(measure(true, repeat));
      old_samples.push_back(measure(false, repeat));
    }
  }
  std::sort(old_samples.begin(), old_samples.end());
  std::sort(new_samples.begin(), new_samples.end());
  std::cout << std::setprecision(12)
            << "{\"exact_reference_cases\": " << checked
            << ", \"calls_per_timing_sample\": 128"
            << ", \"reference_ms_median\": " << old_samples[2]
            << ", \"optimized_ms_median\": " << new_samples[2]
            << ", \"speedup\": " << old_samples[2] / new_samples[2]
            << ", \"checksum\": " << checksum << "}\n";
}
