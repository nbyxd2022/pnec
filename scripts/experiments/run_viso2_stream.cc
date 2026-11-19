// Stream rectified grayscale stereo pairs to standalone libviso2.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "viso_stereo.h"

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "Usage: viso2_stream WIDTH HEIGHT FRAMES F CU CV BASELINE\n";
    return 2;
  }
  try {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
      throw std::runtime_error("Could not enable binary input");
    }
#endif
    std::ios::sync_with_stdio(false);
    const int32_t width = std::stoi(argv[1]);
    const int32_t height = std::stoi(argv[2]);
    const int32_t count = std::stoi(argv[3]);
    VisualOdometryStereo::parameters options;
    options.calib.f = std::stod(argv[4]);
    options.calib.cu = std::stod(argv[5]);
    options.calib.cv = std::stod(argv[6]);
    options.base = std::stod(argv[7]);
    if (width <= 0 || height <= 0 || count < 2 || options.calib.f <= 0 || options.base <= 0) {
      throw std::runtime_error("Invalid image dimensions, frame count or calibration");
    }

    const auto pixels = static_cast<std::size_t>(width) * height;
    int32_t dimensions[] = {width, height, width};
    std::vector<uint8_t> left(pixels), right(pixels);
    VisualOdometryStereo odometry(options);
    std::cout << std::setprecision(17);
    for (int32_t index = 0; index < count; ++index) {
      if (!std::cin.read(reinterpret_cast<char *>(left.data()), pixels) ||
          !std::cin.read(reinterpret_cast<char *>(right.data()), pixels)) {
        throw std::runtime_error("Incomplete stereo input at frame index " + std::to_string(index));
      }
      const bool ok = odometry.process(left.data(), right.data(), dimensions);
      if (index == 0) {
        continue;
      }

      std::cout << index << ' ' << static_cast<int>(ok) << ' '
                << odometry.getNumberOfMatches() << ' '
                << (ok ? odometry.getNumberOfInliers() : 0);
      if (ok) {
        const auto motion = odometry.getMotion();
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 4; ++col) {
            std::cout << ' ' << motion.val[row][col];
          }
        }
      }
      std::cout << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
