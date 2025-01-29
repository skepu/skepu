/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.
    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <skepu>

namespace mnist {
	using label_t = int;//size_t;
	using float_t = float;
	using vec_t = skepu::Vector<float>;


namespace detail {


template <typename T>
T *reverse_endian(T *p) {
  std::reverse(reinterpret_cast<char *>(p),
               reinterpret_cast<char *>(p) + sizeof(T));
  return p;
}

inline bool is_little_endian() {
  int x = 1;
  return *reinterpret_cast<char *>(&x) != 0;
}

struct mnist_header {
  uint32_t magic_number;
  uint32_t num_items;
  uint32_t num_rows;
  uint32_t num_cols;
};

void mnist_error(std::string err)
{
	std::cerr << err << "\n";
	exit(1);
}

inline void parse_mnist_header(std::ifstream &ifs, mnist_header &header)
{
  ifs.read(reinterpret_cast<char *>(&header.magic_number), 4);
  ifs.read(reinterpret_cast<char *>(&header.num_items), 4);
  ifs.read(reinterpret_cast<char *>(&header.num_rows), 4);
  ifs.read(reinterpret_cast<char *>(&header.num_cols), 4);

  if (is_little_endian()) {
    reverse_endian(&header.magic_number);
    reverse_endian(&header.num_items);
    reverse_endian(&header.num_rows);
    reverse_endian(&header.num_cols);
  }

  if (header.magic_number != 0x00000803 || header.num_items <= 0)
		detail::mnist_error("MNIST label-file format error");

  if (ifs.fail() || ifs.bad())
		detail::mnist_error("file error");
}

inline void parse_mnist_image(
	std::ifstream &ifs,
  const mnist_header &header,
	skepu::Tensor4<float_t> &dst,
	size_t image_index_offset,
  float_t scale_min, float_t scale_max
)
{
  std::vector<uint8_t> image_vec(header.num_rows * header.num_cols);
	ifs.read(reinterpret_cast<char *>(&image_vec[0]), header.num_rows * header.num_cols);

  for (uint32_t y = 0; y < header.num_rows; y++)
    for (uint32_t x = 0; x < header.num_cols; x++)
      dst(image_index_offset, y, x, 0) = (image_vec[y * header.num_cols + x] / float_t(255)) * (scale_max - scale_min) + scale_min;
}

}  // namespace detail

/**
 * parse MNIST database format labels with rescaling/resizing
 * http://yann.lecun.com/exdb/mnist/
 *
 * @param label_file [in]  filename of database (i.e.train-labels-idx1-ubyte)
 * @param labels     [out] parsed label data
 **/
inline skepu::Vector<label_t> parse_mnist_labels(const std::string &label_file) {
  std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

  if (ifs.bad() || ifs.fail())
    detail::mnist_error("failed to open file:" + label_file);

  uint32_t magic_number, num_items;

  ifs.read(reinterpret_cast<char *>(&magic_number), 4);
  ifs.read(reinterpret_cast<char *>(&num_items), 4);

  if (detail::is_little_endian()) {  // MNIST data is big-endian format
    detail::reverse_endian(&magic_number);
    detail::reverse_endian(&num_items);
  }

  if (magic_number != 0x00000801 || num_items <= 0)
    detail::mnist_error("MNIST label-file format error");

	skepu::Vector<label_t> labels(num_items, "MNIST-labels");
	skepu::external("Parse MNIST labels", [&]
	{
	  for (uint32_t i = 0; i < num_items; i++)
		{
	    uint8_t label;
	    ifs.read(reinterpret_cast<char *>(&label), 1);
	    labels(i) = static_cast<label_t>(label);
	  }
	}, skepu::write(labels));
	return labels;
}

/**
 * parse MNIST database format images with rescaling/resizing
 * http://yann.lecun.com/exdb/mnist/
 * - if original image size is WxH, output size is
 *(W+2*x_padding)x(H+2*y_padding)
 * - extra padding pixels are filled with scale_min
 *
 * @param image_file [in]  filename of database (i.e.train-images-idx3-ubyte)
 * @param images     [out] parsed image data
 * @param scale_min  [in]  min-value of output
 * @param scale_max  [in]  max-value of output
 * @param x_padding  [in]  adding border width (left,right)
 * @param y_padding  [in]  adding border width (top,bottom)
 *
 * [example]
 * scale_min=-1.0, scale_max=1.0, x_padding=1, y_padding=0
 *
 * [input]       [output]
 *  64  64  64   -1.0 -0.5 -0.5 -0.5 -1.0
 * 128 128 128   -1.0  0.0  0.0  0.0 -1.0
 * 255 255 255   -1.0  1.0  1.0  1.0 -1.0
 *
 **/
inline skepu::Tensor4<float> parse_mnist_images(
	const std::string &image_file,
	float_t scale_min, float_t scale_max
)
{
  if (scale_min >= scale_max)
    detail::mnist_error("scale_max must be greater than scale_min");

  std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

  if (ifs.bad() || ifs.fail())
    detail::mnist_error("failed to open file:" + image_file);

  detail::mnist_header header;
  detail::parse_mnist_header(ifs, header);

	skepu::Tensor4<float> images(header.num_items, header.num_rows, header.num_cols, 1, "MNIST-images");
	skepu::external("Parse MNIST images", [&]
	{
	  for (uint32_t i = 0; i < header.num_items; i++)
			detail::parse_mnist_image(ifs, header, images, i, scale_min, scale_max);
	}, skepu::write(images));
	return images;
}




}  // namespace mnist
