#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>
#include <vector>

// the output has 1 + (n - pool) / stride elements

int pool_sum(skepu::Pool1D<int> p)
{
	int sum = 0;
	for (int i = 0; i < p.si; ++i)
		sum += p(i);
	return sum;
}

float pool_scaled(skepu::Index1D index, skepu::Pool1D<float> p, const skepu::Vec<float> offsets, float scale)
{
	float sum = 0;
	for (int i = 0; i < p.si; ++i)
		sum += p(i);
	return sum * scale + offsets[index.i];
}

auto sums = skepu::MapPool(pool_sum);
auto scaled = skepu::MapPool(pool_scaled);

static std::vector<int> expected_sums(const skepu::Vector<int> &in, int pool, int stride, size_t out_size)
{
	std::vector<int> expected(out_size);
	for (size_t i = 0; i < out_size; ++i)
		for (int k = 0; k < pool; ++k)
			expected[i] += in(i * stride + k);
	return expected;
}

TEST_CASE("MapPool 1D: pool sizes and strides")
{
	const size_t n = 103;
	skepu::Vector<int> in(n);
	for (size_t i = 0; i < n; ++i)
		in(i) = static_cast<int>(i * i % 17);

	for (int pool : {1, 2, 5, 16})
		for (int stride : {1, 3, pool})
		{
			const size_t out_size = 1 + (n - pool) / stride;
			skepu::Vector<int> out(out_size);
			sums.setPoolSize(pool);
			sums.setStride(stride);
			sums(out, in);
			out.flush();

			const std::vector<int> expected = expected_sums(in, pool, stride, out_size);
			for (size_t i = 0; i < out_size; ++i)
				CHECK(out(i) == expected[i]);
		}
}

TEST_CASE("MapPool 1D: pool size and stride together, a smaller output than the input allows")
{
	skepu::Vector<int> in(20);
	for (size_t i = 0; i < 20; ++i)
		in(i) = static_cast<int>(i);

	skepu::Vector<int> out(3);
	sums.setPoolSizeAndStride(4);
	sums(out, in);
	out.flush();
	CHECK(out(0) == 0 + 1 + 2 + 3);
	CHECK(out(1) == 4 + 5 + 6 + 7);
	CHECK(out(2) == 8 + 9 + 10 + 11);
}

TEST_CASE("MapPool 1D: index, uniform and container arguments")
{
	skepu::Vector<float> in(12), offsets(5), out(5);
	for (size_t i = 0; i < 12; ++i)
		in(i) = static_cast<float>(i);
	for (size_t i = 0; i < 5; ++i)
		offsets(i) = 100.f * i;

	scaled.setPoolSize(3);
	scaled.setStride(2);
	scaled(out, in, offsets, 2.f);
	out.flush();
	for (size_t i = 0; i < 5; ++i)
		CHECK(out(i) == 2.f * (2 * i + (2 * i + 1) + (2 * i + 2)) + 100.f * i);
}

TEST_CASE("MapPool 1D: an input that is too small is an error")
{
	skepu::Vector<int> in(10), out(10);
	sums.setPoolSize(2);
	sums.setStride(1);
	CHECK_THROWS(sums(out, in));
}

skepu::multiple<int, float> pool_sum_and_mean(skepu::Pool1D<int> p)
{
	int sum = 0;
	for (int i = 0; i < p.si; ++i)
		sum += p(i);
	return skepu::ret(sum, (float)sum / p.si);
}

skepu::multiple<int, int, float> pool_extremes(skepu::Index1D index, skepu::Pool1D<int> p, float scale)
{
	int low = p(0), high = p(0);
	for (int i = 1; i < p.si; ++i)
	{
		low = p(i) < low ? p(i) : low;
		high = p(i) > high ? p(i) : high;
	}
	return skepu::ret(low, high, (high - low) * scale + index.i);
}

auto sum_and_mean = skepu::MapPool(pool_sum_and_mean);
auto extremes = skepu::MapPool(pool_extremes);

TEST_CASE("MapPool 1D: several return values")
{
	const size_t n = 103;
	skepu::Vector<int> in(n);
	for (size_t i = 0; i < n; ++i)
		in(i) = static_cast<int>(i * i % 17);

	for (int pool : {2, 4, 5})
		for (int stride : {1, pool})
		{
			const size_t out_size = 1 + (n - pool) / stride;
			skepu::Vector<int> sums(out_size), lows(out_size), highs(out_size);
			skepu::Vector<float> means(out_size), spread(out_size);

			sum_and_mean.setPoolSize(pool);
			sum_and_mean.setStride(stride);
			sum_and_mean(sums, means, in);
			sums.flush();
			means.flush();

			extremes.setPoolSize(pool);
			extremes.setStride(stride);
			extremes(lows, highs, spread, in, 0.5f);
			lows.flush();
			highs.flush();
			spread.flush();

			for (size_t i = 0; i < out_size; ++i)
			{
				int sum = 0, low = in(i * stride), high = in(i * stride);
				for (int k = 0; k < pool; ++k)
				{
					const int v = in(i * stride + k);
					sum += v;
					low = std::min(low, v);
					high = std::max(high, v);
				}
				CHECK(sums(i) == sum);
				CHECK(means(i) == (float)sum / pool);
				CHECK(lows(i) == low);
				CHECK(highs(i) == high);
				CHECK(spread(i) == (high - low) * 0.5f + i);
			}
		}
}
