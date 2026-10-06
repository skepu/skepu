#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/filter.hpp>
#include <test_helper_automatic_setup.hpp>

// The image filter library (skepu-lib/filter.hpp): its pixel types and user functions are in a
// namespace (skepu::filter) and spell the types unqualified. The kernels of every backend must refer
// to them consistently, as elementwise parameters, results, multiple results, regions and in the bodies.

using skepu::filter::RGBPixel;
using skepu::filter::RGBExtended;
using skepu::filter::GrayscalePixel;

auto intensity = skepu::Map<1>(skepu::filter::intensity_kernel);
auto extend = skepu::Map<1>(skepu::filter::precision_extend_rgb);
auto reduce_precision = skepu::Map<1>(skepu::filter::precision_reduce_rgb);
auto red = skepu::Map<1>(skepu::filter::red_channel);
auto channels = skepu::Map<1>(skepu::filter::channels_from_rgb);
auto distance = skepu::Map<2>(skepu::filter::distance_kernel);
auto convolution = skepu::MapOverlap(skepu::filter::convolution_kernel_rgb_1d);
auto convolution_gray = skepu::MapOverlap(skepu::filter::convolution_kernel);
auto median = skepu::MapOverlap(skepu::filter::median_kernel);

static skepu::Vector<RGBPixel> pixels(size_t size)
{
	skepu::Vector<RGBPixel> v(size);
	for (size_t i = 0; i < size; ++i)
	{
		v(i).r = (unsigned char)(i * 3);
		v(i).g = (unsigned char)(i * 3 + 1);
		v(i).b = (unsigned char)(i * 3 + 2);
	}
	return v;
}

TEST_CASE("skepu::filter, element-wise functions on pixels")
{
	const size_t size{32};
	auto in = pixels(size);
	skepu::Vector<GrayscalePixel> gray(size), r(size), g(size), b(size);
	skepu::Vector<RGBExtended> extended(size);
	skepu::Vector<RGBPixel> back(size);

	intensity(gray, in);
	red(r, in);
	channels(r, g, b, in);
	extend(extended, in);
	reduce_precision(back, extended);

	gray.flush(); r.flush(); g.flush(); b.flush(); extended.flush(); back.flush();
	for (size_t i = 0; i < size; ++i)
	{
		CHECK(gray(i).intensity == (unsigned char)((unsigned(i * 3) + unsigned(i * 3 + 1) + unsigned(i * 3 + 2)) / 3));
		CHECK(r(i).intensity == (unsigned char)(i * 3));
		CHECK(g(i).intensity == (unsigned char)(i * 3 + 1));
		CHECK(b(i).intensity == (unsigned char)(i * 3 + 2));
		CHECK(extended(i).g == Approx((i * 3 + 1) / 255.f));
		CHECK(back(i).r == in(i).r);
		CHECK(back(i).g == in(i).g);
		CHECK(back(i).b == in(i).b);
	}
}

TEST_CASE("skepu::filter, a function of two pixels")
{
	const size_t size{16};
	skepu::Vector<GrayscalePixel> x(size), y(size), d(size);
	for (size_t i = 0; i < size; ++i)
	{
		x(i).intensity = (unsigned char)(120 + i);
		y(i).intensity = (unsigned char)(135 - i);
	}

	distance(d, x, y);

	d.flush();
	for (size_t i = 0; i < size; ++i)
	{
		// (255 minus the distance from the middle of the range, scaled; devices round differently)
		const float xf = (120 + i) / 255.f - 0.5f, yf = (135 - i) / 255.f - 0.5f;
		const int expected = int(255 - std::sqrt(xf * xf + yf * yf) * 2 * 255);
		CHECK(std::abs(int(d(i).intensity) - expected) <= 1);
	}
}

TEST_CASE("skepu::filter, a convolution with regions of pixels")
{
	const size_t size{24};
	auto in = pixels(size);
	skepu::Vector<RGBPixel> out(size);
	skepu::Vector<float> filter(3);
	filter(0) = 1.f; filter(1) = 2.f; filter(2) = 1.f;

	convolution.setOverlap(1);
	convolution.setEdgeMode(skepu::Edge::Duplicate);
	convolution(out, in, filter, 0.f, 0.25f);

	out.flush();
	for (size_t i = 1; i + 1 < size; ++i)
	{
		const unsigned expected_r = (in(i - 1).r + 2u * in(i).r + in(i + 1).r) / 4u;
		CHECK(out(i).r == (unsigned char)expected_r);
		const unsigned expected_b = (in(i - 1).b + 2u * in(i).b + in(i + 1).b) / 4u;
		CHECK(out(i).b == (unsigned char)expected_b);
	}
}

TEST_CASE("skepu::filter, a convolution of grayscale pixels with a filter container")
{
	const size_t size{16};
	skepu::Vector<GrayscalePixel> in(size), out(size);
	for (size_t i = 0; i < size; ++i)
		in(i).intensity = (unsigned char)(8 * i);
	skepu::Vector<float> filter(3, 1.f);

	convolution_gray.setOverlap(1);
	convolution_gray.setEdgeMode(skepu::Edge::Duplicate);
	convolution_gray(out, in, filter, 0.f, 1.f / 3.f);

	out.flush();
	for (size_t i = 1; i + 1 < size; ++i)
		CHECK(out(i).intensity == (unsigned char)(8 * i));
}

TEST_CASE("skepu::filter, a median over a region of pixels")
{
	const size_t rows{8}, cols{8};
	skepu::Matrix<RGBPixel> in(rows, cols), out(rows, cols);
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
		{
			in(i, j).r = in(i, j).g = in(i, j).b = (unsigned char)(i + j);
		}

	median.setOverlap(1, 1);
	median.setEdgeMode(skepu::Edge::Duplicate);
	median(out, in);

	out.flush();
	// away from the edges, the neighbourhood of (i, j) has the values i+j-2 .. i+j+2 with the median i+j
	for (size_t i = 1; i + 1 < rows; ++i)
		for (size_t j = 1; j + 1 < cols; ++j)
			CHECK(out(i, j).r == (unsigned char)(i + j));
}
