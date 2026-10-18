// Minimal loader for the CIFAR-10 binary version (https://www.cs.toronto.edu/~kriz/cifar.html).
// Each record is 3073 bytes: 1 label byte followed by 3 x 1024 pixel bytes in channel-first (R, G, B) order.
// Images are returned in the layout used by SkePU-DNN, (n, 32, 32, 3) (height, width, channel), as floats in [0, 1].
// (The standardization of the channels is made by skeletons in the program, dnn_cifar10.cpp: skeletons are precompiled in the file
// of the program, and the statistics must be known by every process of an MPI program, which the code of skepu::external is not.)
#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <skepu>
#include <skepu-lib/ml.hpp>

namespace cifar10 {

constexpr size_t IMAGE_SIZE = 32;
constexpr size_t CHANNELS = 3;
constexpr size_t NUM_CLASSES = 10;
constexpr size_t RECORD_SIZE = 1 + CHANNELS * IMAGE_SIZE * IMAGE_SIZE;

static const char *class_names[NUM_CLASSES] = {
	"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
};

// Loads up to max_count images from the given batch files (0 = all), and their classes (0 to 9, of the type Label).
template<typename Label = int>
inline void load_classes(const std::vector<std::string> &files, size_t max_count,
	skepu::Tensor4<float> *&images, skepu::Vector<Label> *&classes)
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
	classes = new skepu::Vector<Label>(n, "CIFAR-10 classes");

	// The host writes the elements of the containers: in skepu::external, which makes the containers
	// current on the host first and the host's data the current one afterwards (also with a device and
	// with distributed containers: the code runs on the first process and the data goes to the others after it).
	skepu::external([&]
	{
		for (size_t i = 0; i < n; ++i)
		{
			const uint8_t *rec = raw.data() + i * RECORD_SIZE;
			(*classes)(i) = rec[0];
			for (size_t c = 0; c < CHANNELS; ++c)
				for (size_t y = 0; y < IMAGE_SIZE; ++y)
					for (size_t x = 0; x < IMAGE_SIZE; ++x)
						(*images)(i, y, x, c) = rec[1 + c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] / 255.f;
		}
	}, skepu::write(*images, *classes));
}

// The same, with the labels as one-hot vectors (n, 1, 1, 10), made by skepu::dnn::one_hot.
inline void load(const std::vector<std::string> &files, size_t max_count,
	skepu::Tensor4<float> *&images, skepu::Tensor4<float> *&labels)
{
	skepu::Vector<int> *classes = nullptr;
	load_classes(files, max_count, images, classes);
	labels = new skepu::Tensor4<float>(skepu::dnn::one_hot(*classes, NUM_CLASSES, "CIFAR-10 labels"));
	delete classes;
}

} // namespace cifar10
