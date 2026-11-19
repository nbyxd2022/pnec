#ifndef PNEC_PHOTOMETRIC_INCREMENT_H_
#define PNEC_PHOTOMETRIC_INCREMENT_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace pnec::rel_pose_estimation::photometric {

enum class Mode {
  Disabled, Original, Uniform, PerMatch, SymmetricAnalytic,
  SymmetricUniform, SymmetricPerMatch, WeakPerMatch, MatchedBudget, FarPeakTaper
};

inline Mode ExperimentMode() {
  static const Mode mode = [] {
    const char *setting = std::getenv("PNEC_PHOTOMETRIC_INCREMENT");
    if (!setting || !*setting) {
      return Mode::Disabled;
    }
    const std::string_view value(setting);
    if (value == "original") {
      return Mode::Original;
    }
    if (value == "uniform") {
      return Mode::Uniform;
    }
    if (value == "per_match") {
      return Mode::PerMatch;
    }
    if (value == "symmetric_analytic") {
      return Mode::SymmetricAnalytic;
    }
    if (value == "symmetric_uniform") {
      return Mode::SymmetricUniform;
    }
    if (value == "symmetric_per_match") {
      return Mode::SymmetricPerMatch;
    }
    if (value == "weak_per_match") {
      return Mode::WeakPerMatch;
    }
    if (value == "matched_budget") {
      return Mode::MatchedBudget;
    }
    if (value == "far_peak_taper") {
      return Mode::FarPeakTaper;
    }
    throw std::invalid_argument(
        "PNEC_PHOTOMETRIC_INCREMENT expects original, uniform, per_match, "
        "symmetric_analytic, symmetric_uniform, symmetric_per_match, "
        "weak_per_match, matched_budget or far_peak_taper");
  }();
  return mode;
}

struct GrayImage {
  const std::uint8_t *data;
  int width;
  int height;
  std::size_t stride;

  bool Supports(double x, double y, int radius) const {
    return x - radius >= 0 && y - radius >= 0 &&
           x + radius < width - 1 && y + radius < height - 1;
  }

  // The caller checks support for the entire patch before sampling.
  double Bilinear(double x, double y) const {
    const int ix = static_cast<int>(x);
    const int iy = static_cast<int>(y);
    const double dx = x - ix, dy = y - iy;
    const auto *row = data + iy * stride;
    const auto *next = row + stride;
    return (1 - dx) * (1 - dy) * (row[ix] / 255.0) +
           dx * (1 - dy) * (row[ix + 1] / 255.0) +
           (1 - dx) * dy * (next[ix] / 255.0) +
           dx * dy * (next[ix + 1] / 255.0);
  }
};

struct Ambiguity {
  double ratio = 0.0;
  int dx = 0;
  int dy = 0;
};

struct Measurement {
  double best_cost = std::numeric_limits<double>::infinity();
  double second_cost = std::numeric_limits<double>::infinity();
  double endpoint_cost = std::numeric_limits<double>::infinity();
  int best_dx = 0;
  int best_dy = 0;
};

inline bool IsWeakIncrement(Mode mode) {
  return mode == Mode::WeakPerMatch || mode == Mode::MatchedBudget ||
         mode == Mode::FarPeakTaper;
}

inline bool UsesFarPeakRetention(Mode mode) {
  return mode == Mode::MatchedBudget || mode == Mode::FarPeakTaper;
}

inline double FarPeakRetention(const Measurement &measurement,
                               const Ambiguity &ambiguity) {
  constexpr double best_limit = 2;
  constexpr double far_limit = 12;
  if (!std::isfinite(measurement.best_cost) ||
      !std::isfinite(measurement.second_cost) ||
      std::hypot(measurement.best_dx, measurement.best_dy) > best_limit) {
    return 1;
  }
  const double distance = std::hypot(measurement.best_dx + ambiguity.dx,
                                     measurement.best_dy + ambiguity.dy);
  if (distance <= far_limit) {
    return 1;
  }
  const double ratio = far_limit / distance;
  return ratio * ratio;
}

// In each run the same frame's original traces define its matched total.
inline double ApplyWeakIncrement(Mode mode, const std::vector<double> &retention,
                                 double original_total, std::vector<double> &traces) {
  constexpr double scale = 1.0 / 32;
  const bool matched = mode == Mode::MatchedBudget;
  const bool tapered = mode == Mode::FarPeakTaper;
  double factor = 1;
  if (matched && original_total > 0) {
    double tapered_total = 0;
    for (size_t i = 0; i < traces.size(); ++i) {
      tapered_total += traces[i] * retention[i];
    }
    factor = tapered_total / original_total;
  }
  double total = 0;
  for (size_t i = 0; i < traces.size(); ++i) {
    traces[i] *= tapered ? retention[i] : factor;
    traces[i] *= scale;
    total += traces[i];
  }
  return total;
}

// Keep the offline 15x15 raw SSD, +/-16 px grid and 4 px mode separation.
inline Ambiguity Measure(const GrayImage &host, const GrayImage &target,
                         double host_x, double host_y, double target_x,
                         double target_y, Measurement *measurement = nullptr) {
  if (measurement) {
    *measurement = {};
  }
  constexpr int patch_radius = 7;
  constexpr int search_radius = 16;
  constexpr int patch_side = 2 * patch_radius + 1;
  constexpr int search_side = 2 * search_radius + 1;
  constexpr int raster_radius = patch_radius + search_radius;
  constexpr int raster_side = 2 * raster_radius + 1;
  constexpr double invalid = std::numeric_limits<double>::infinity();

  if (!host.Supports(host_x, host_y, patch_radius) ||
      !target.Supports(target_x, target_y, patch_radius)) {
    return {};
  }

  std::array<double, patch_side * patch_side> source;
  for (int y = 0; y < patch_side; ++y) {
    for (int x = 0; x < patch_side; ++x) {
      source[y * patch_side + x] =
          host.Bilinear(host_x + (x - patch_radius), host_y + (y - patch_radius));
    }
  }

  std::array<double, raster_side * raster_side> raster{};
  for (int y = 0; y < raster_side; ++y) {
    for (int x = 0; x < raster_side; ++x) {
      const double px = target_x + (x - raster_radius);
      const double py = target_y + (y - raster_radius);
      if (target.Supports(px, py, 0)) {
        raster[y * raster_side + x] = target.Bilinear(px, py);
      }
    }
  }

  std::array<double, search_side * search_side> costs;
  costs.fill(invalid);
  int best_x = 0, best_y = 0;
  double best = invalid;
  for (int y = 0; y < search_side; ++y) {
    double *row_costs = costs.data() + y * search_side;
    for (int x = 0; x < search_side; ++x) {
      row_costs[x] = target.Supports(target_x + (x - search_radius),
                                     target_y + (y - search_radius), patch_radius)
                         ? 0.0 : invalid;
    }
    // Independent candidate accumulators let the compiler vectorize columns.
    // Each candidate keeps the original patch-pixel summation order.
    for (int row = 0; row < patch_side; ++row) {
      const double *pixels = raster.data() + (y + row) * raster_side;
      for (int col = 0; col < patch_side; ++col) {
        const double value = source[row * patch_side + col];
        for (int x = 0; x < search_side; ++x) {
          const double difference = pixels[x + col] - value;
          row_costs[x] += difference * difference;
        }
      }
    }
    // Visit equal-cost candidates in (dy, dx) order.
    for (int x = 0; x < search_side; ++x) {
      if (row_costs[x] < best) {
        best = row_costs[x];
        best_x = x;
        best_y = y;
      }
    }
  }

  Ambiguity result;
  double second = invalid;
  for (int y = 0; y < search_side; ++y) {
    for (int x = 0; x < search_side; ++x) {
      const double cost = costs[y * search_side + x];
      const int dx = x - best_x, dy = y - best_y;
      if (cost >= second || dx * dx + dy * dy < 16) {
        continue;
      }
      bool local_minimum = true;
      for (int ny = y - 1; ny <= y + 1; ++ny) {
        for (int nx = x - 1; nx <= x + 1; ++nx) {
          if (nx >= 0 && nx < search_side && ny >= 0 && ny < search_side &&
              costs[ny * search_side + nx] < cost) {
            local_minimum = false;
          }
        }
      }
      if (local_minimum) {
        second = cost;
        result = {second > 0 ? best / second : 1.0, dx, dy};
      }
    }
  }
  if (measurement) {
    measurement->best_cost = best;
    measurement->second_cost = second;
    measurement->endpoint_cost = costs[search_radius * search_side + search_radius];
    measurement->best_dx = best_x - search_radius;
    measurement->best_dy = best_y - search_radius;
  }
  return result;
}

} // namespace pnec::rel_pose_estimation::photometric

#endif // PNEC_PHOTOMETRIC_INCREMENT_H_
