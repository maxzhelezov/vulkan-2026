#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdint.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "nlohmann/json.hpp"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"


struct alignas(sizeof(double)) Color {
  double r, g, b, a;
};


std::vector<Color> LoadImage(const std::filesystem::path &path, size_t &width, size_t &height) {
  int iwidth{}, iheight{}, ichannels{};
  uint8_t *data = stbi_load(path.string().c_str(), &iwidth, &iheight, &ichannels, 4);
  if (!data) {
    std::ostringstream oss;
    oss << "Failed to load '" << path << "'";
    throw std::runtime_error(oss.str());
  }
  assert(iwidth != 0);
  assert(iheight != 0);
  assert(ichannels == 4);

  size_t size = iwidth * iheight;
  std::vector<Color> image(size);
  for (size_t i = 0; i < size; i++) {
    image[i].r = data[i * 4 + 0] / 255.0;
    image[i].g = data[i * 4 + 1] / 255.0;
    image[i].b = data[i * 4 + 2] / 255.0;
    image[i].a = data[i * 4 + 3] / 255.0;
  }

  stbi_image_free(data);
  width  = iwidth;
  height = iheight;
  return image;
}


void SaveImage(const std::vector<Color> &image,
               size_t width, size_t height,
               const std::filesystem::path &path) {
  assert(!image.empty());
  assert(image.size() == width * height);

  std::vector<uint8_t> buffer(image.size() * 4);
  for (size_t i = 0; i < width * height; i++) {
    buffer[i * 4 + 0] = static_cast<uint8_t>(std::clamp(image[i].r * 255, 0.0, 255.0));
    buffer[i * 4 + 1] = static_cast<uint8_t>(std::clamp(image[i].g * 255, 0.0, 255.0));
    buffer[i * 4 + 2] = static_cast<uint8_t>(std::clamp(image[i].b * 255, 0.0, 255.0));
    buffer[i * 4 + 3] = static_cast<uint8_t>(std::clamp(image[i].a * 255, 0.0, 255.0));
  }

  stbi_write_png(path.string().c_str(), width, height, 4,
                 buffer.data(), width * 4 * sizeof(uint8_t));
}


void Saturation(std::vector<Color> &dst,
                const std::vector<Color> &src,
                double scale) {
  assert(!src.empty());
  assert(dst.size() == src.size());
  assert(scale >= 0.0);
  assert(scale <= 2.0);

  double rscale = 1.0 - scale;
  for (size_t i = 0; i < src.size(); i++)
  {
    double desaturated = (src[i].r * 0.2126 +
                          src[i].g * 0.7152 +
                          src[i].b * 0.0722) * rscale;
    dst[i].r = desaturated + src[i].r * scale;
    dst[i].g = desaturated + src[i].g * scale;
    dst[i].b = desaturated + src[i].b * scale;
    dst[i].a = src[3].a;
  }
}


void BrightnessContrast(std::vector<Color> &dst,
                        const std::vector<Color> &src,
                        int8_t brightness, int8_t contrast) {
  assert(!src.empty());
  assert(dst.size() == src.size());
  assert(std::abs(brightness) <= 127);
  assert(std::abs(contrast)   <= 127);

  double brightness_shift = brightness / 127.0;
  double contrast_factor = std::tan((contrast / 127.0 + 1.0) * (std::numbers::pi / 4.0));
  for (size_t i = 0; i < src.size(); i++)
  {
    dst[i].r = (src[i].r - 0.5) * contrast_factor + brightness_shift + 0.5;
    dst[i].g = (src[i].g - 0.5) * contrast_factor + brightness_shift + 0.5;
    dst[i].b = (src[i].b - 0.5) * contrast_factor + brightness_shift + 0.5;
    dst[i].a = src[i].a;
  }
}


void Exposure(std::vector<Color> &dst,
              const std::vector<Color> &src,
              float black_level, float exposure) {
  assert(!src.empty());
  assert(dst.size() == src.size());
  assert(std::abs(black_level) <= 0.1);
  assert(std::abs(exposure)    <= 10.0);

  double white = std::exp(-exposure);
  double diff  = std::max(white - black_level, 0.000001);
  double gain  = 1.0f / diff;

  for (size_t i = 0;  i <src.size(); i++)
  {
    dst[i].r = (src[i].r - black_level) * gain;
    dst[i].g = (src[i].g - black_level) * gain;
    dst[i].b = (src[i].b - black_level) * gain;
    dst[i].a = src[i].a;
  }
}


int main(int argc, const char **argv) {
  try {
    if (argc != 3 || !std::filesystem::exists(argv[1]) || !std::filesystem::exists(argv[2])) {
      throw std::runtime_error("Wrong arguments or files not found");
    }
    
    size_t width{}, height{};
    std::vector<Color> img = LoadImage(argv[1], width, height);
    std::ifstream config(argv[2]);
    nlohmann::json json;
    config >> json;
    if (!config) {
      throw std::runtime_error("Failed to parse JSON configuration file");
    }

    std::vector<Color> buf1(img.size()), buf2(img.size()), buf3(img.size());
    Saturation(buf1, img,
               (double)json["Saturation"]["scale"]);
    BrightnessContrast(buf2, buf1,
                       (int8_t)json["BrightnessContrast"]["brightness"],
                       (int8_t)json["BrightnessContrast"]["contrast"]);
    Exposure(buf3, buf2,
             (double)json["Exposure"]["black_level"],
             (double)json["Exposure"]["exposure"]);
    
    SaveImage(buf3, width, height, "gold.png");
    return 0;
  }
  catch (std::exception &ex) {
    std::cerr << ex.what() << std::endl;
    return 42;
  }
}
