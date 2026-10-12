#include "../../external/catch2/catch.hpp"

#include <cmath>
#include <skepu>
#include <skepu-lib/randomize.hpp>
#include <test_helper_automatic_setup.hpp>

// The fills of skepu::randomize in double precision (not on OpenCL: the devices of some platforms have no doubles).

namespace
{
	// 62 random bits from two draws of a stream, as the fills make them
	unsigned long long bits62(skepu::PRNG &stream)
	{
		unsigned long long high = stream.get();
		unsigned long long low = stream.get();
		return (high << 31) | low;
	}

}

TEST_CASE("randomize::uniform, double precision")
{
	const size_t n = 500;
	skepu::PRNG stream(5), reference(5);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<double> v(n);
	skepu::randomize::uniform(v, 10.0, 20.0);
	v.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		const double want = 10.0 + (double)(bits62(reference) >> 9) * std::ldexp(1.0, -53) * 10.0;
		REQUIRE(v(i) == want);
	}
}


TEST_CASE("randomize::normal, double precision")
{
	const size_t n = 100000;
	skepu::PRNG stream(21);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<double> v(n);
	skepu::randomize::normal(v, -1.0, 0.5);
	v.updateHost();
	double sum = 0, sum2 = 0;
	for (size_t i = 0; i < n; ++i) { sum += v(i); sum2 += v(i) * v(i); }
	const double mean = sum / n;
	REQUIRE(std::fabs(mean + 1.0) < 0.01);
	REQUIRE(std::fabs(std::sqrt(sum2 / n - mean * mean) - 0.5) < 0.01);
}
