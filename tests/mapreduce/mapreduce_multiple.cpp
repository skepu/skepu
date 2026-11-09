#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

// each returned value is reduced separately; the result is a tuple. The reduction starts from 0.

skepu::multiple<float, int> value_and_double(float a)
{
	return skepu::ret(a, (int)a * 2);
}

skepu::multiple<float, float, float> three_values(float a, float scale)
{
	return skepu::ret(a, a * scale, a * a);
}

float plus(float a, float b) { return a + b; }
float larger(float a, float b) { return a > b ? a : b; }

auto largest = skepu::MapReduce<1>(value_and_double, larger);
auto sums = skepu::MapReduce<1>(three_values, plus);

TEST_CASE("MapReduce with several values, several blocks")
{
	const size_t n = 100000;
	skepu::Vector<float> v(n);
	for (size_t i = 0; i < n; ++i)
		v(i) = static_cast<float>(i % 10);

	auto result = largest(v);
	CHECK(static_cast<float>(std::get<0>(result)) == 9.f);
	CHECK(static_cast<int>(std::get<1>(result)) == 18);

	auto s = sums(v, 2.f);
	CHECK(static_cast<float>(std::get<0>(s)) == 450000.f);
	CHECK(static_cast<float>(std::get<1>(s)) == 900000.f);
	CHECK(static_cast<float>(std::get<2>(s)) == 2850000.f);
}

TEST_CASE("MapReduce with several values, fewer elements than threads")
{
	skepu::Vector<float> v(5);
	for (size_t i = 0; i < 5; ++i)
		v(i) = static_cast<float>(i + 1);

	auto result = largest(v);
	CHECK(static_cast<float>(std::get<0>(result)) == 5.f);
	CHECK(static_cast<int>(std::get<1>(result)) == 10);
}
