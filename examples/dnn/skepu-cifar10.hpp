// Minimal loader for the CIFAR-10 binary version (https://www.cs.toronto.edu/~kriz/cifar.html).
// Each record is 3073 bytes: 1 label byte followed by 3 x 1024 pixel bytes in channel-first (R, G, B) order.
// Images are returned in the layout used by SkePU-DNN, (n, 32, 32, 3) (height, width, channel), as floats in [0, 1].
#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <skepu>

namespace cifar10 {

constexpr size_t IMAGE_SIZE = 32;
constexpr size_t CHANNELS = 3;
constexpr size_t NUM_CLASSES = 10;
constexpr size_t RECORD_SIZE = 1 + CHANNELS * IMAGE_SIZE * IMAGE_SIZE;

static const char *class_names[NUM_CLASSES] = {
	"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
};

// Loads up to max_count images from the given batch files (0 = all). Labels are returned as
// one-hot (n, 1, 1, 10) when onehot is true.
inline void load(const std::vector<std::string> &files, size_t max_count,
	skepu::Tensor4<float> *&images, skepu::Tensor4<float> *&labels)
{
	std::vector<uint8_t> raw;
	for (const std::string &file : files)
	{
		std::ifstream in(file, std::ios::binary);
		if (!in)
		{
			std::cerr << "CIFAR-10: cannot open " << file << "\n";
			exit(1);
		}
		in.seekg(0, std::ios::end);
		size_t bytes = in.tellg();
		in.seekg(0);
		size_t old = raw.size();
		raw.resize(old + bytes);
		in.read(reinterpret_cast<char *>(raw.data() + old), bytes);
		if (max_count > 0 && raw.size() / RECORD_SIZE >= max_count) break;
	}

	size_t n = raw.size() / RECORD_SIZE;
	if (max_count > 0 && max_count < n) n = max_count;

	images = new skepu::Tensor4<float>(n, IMAGE_SIZE, IMAGE_SIZE, CHANNELS, "CIFAR-10 images");
	labels = new skepu::Tensor4<float>(n, 1, 1, NUM_CLASSES, "CIFAR-10 labels", 0.f);
	for (size_t i = 0; i < n; ++i)
	{
		const uint8_t *rec = raw.data() + i * RECORD_SIZE;
		(*labels)(i, 0, 0, rec[0]) = 1.f;
		for (size_t c = 0; c < CHANNELS; ++c)
			for (size_t y = 0; y < IMAGE_SIZE; ++y)
				for (size_t x = 0; x < IMAGE_SIZE; ++x)
					(*images)(i, y, x, c) = rec[1 + c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] / 255.f;
	}
}

// Per-channel standardization using the statistics of the given (training) set
inline void channel_stats(skepu::Tensor4<float> &images, float mean[CHANNELS], float stddev[CHANNELS])
{
	for (size_t c = 0; c < CHANNELS; ++c)
	{
		double sum = 0, sq = 0;
		size_t count = images.size_i() * images.size_j() * images.size_k();
		for (size_t i = 0; i < images.size_i(); ++i)
			for (size_t y = 0; y < images.size_j(); ++y)
				for (size_t x = 0; x < images.size_k(); ++x)
				{
					double v = images(i, y, x, c);
					sum += v; sq += v * v;
				}
		mean[c] = sum / count;
		stddev[c] = std::sqrt(sq / count - (double)mean[c] * mean[c]);
	}
}

inline void standardize(skepu::Tensor4<float> &images, const float mean[CHANNELS], const float stddev[CHANNELS])
{
	for (size_t i = 0; i < images.size_i(); ++i)
		for (size_t y = 0; y < images.size_j(); ++y)
			for (size_t x = 0; x < images.size_k(); ++x)
				for (size_t c = 0; c < images.size_l(); ++c)
					images(i, y, x, c) = (images(i, y, x, c) - mean[c]) / stddev[c];
}

} // namespace cifar10
