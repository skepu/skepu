#include "../../external/catch2/catch.hpp"

#include <cmath>
#include <skepu>
#include <skepu-lib/randomize.hpp>
#include <test_helper_automatic_setup.hpp>

// The fills of skepu::randomize against what a single stream gives, element by element, on every backend: the values
// do not depend on how the backend splits the work.

namespace
{
	// 62 random bits from two draws of a stream, as the fills make them
	unsigned long long bits62(skepu::PRNG &stream)
	{
		unsigned long long high = stream.get();
		unsigned long long low = stream.get();
		return (high << 31) | low;
	}

	void require_same_state(skepu::PRNG &used, skepu::PRNG &reference)
	{
		// the next number of the two streams is the same: the fill advanced the stream as much as the reference did
		REQUIRE(used.get() == reference.get());
	}
}

TEST_CASE("randomize::uniform, integers, the values of the stream, in all shapes")
{
	const size_t n = 1000;
	skepu::PRNG stream(7), reference(7);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<int> v(n);
	skepu::randomize::uniform(v, -50, 49);
	v.updateHost();
	for (size_t i = 0; i < n; ++i)
		REQUIRE(v(i) == -50 + (int)(bits62(reference) % 100));
	require_same_state(stream, reference); // (one number on in both)

	skepu::Matrix<long> m(20, 30);
	skepu::randomize::uniform(m, 99);
	m.updateHost();
	for (size_t i = 0; i < 20; ++i)
		for (size_t j = 0; j < 30; ++j)
			REQUIRE(m(i, j) == (long)(bits62(reference) % 100));

	skepu::Tensor3<unsigned> t3(4, 5, 6);
	skepu::randomize::uniform(t3, 9u);
	t3.updateHost();
	for (size_t i = 0; i < 4 * 5 * 6; ++i)
		REQUIRE(t3.getAddress()[i] == (unsigned)(bits62(reference) % 10));

	skepu::Tensor4<short> t4(3, 4, 2, 5);
	skepu::randomize::uniform(t4, (short)3, (short)7);
	t4.updateHost();
	for (size_t i = 0; i < 3 * 4 * 2 * 5; ++i)
		REQUIRE(t4.getAddress()[i] == (short)(3 + bits62(reference) % 5));
}

TEST_CASE("randomize::uniform, floating point, the values of the stream, and the defaults")
{
	const size_t n = 1000;
	skepu::PRNG stream(11), reference(11);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<float> v(n);
	skepu::randomize::uniform(v);
	v.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		const float want = (float)(bits62(reference) >> 38) * (1.0f / 16777216.0f);
		REQUIRE(v(i) == want);
		REQUIRE(v(i) >= 0.f);
		REQUIRE(v(i) < 1.f);
	}

	skepu::Matrix<float> m(30, 20);
	skepu::randomize::uniform(m, -2.f, 6.f);
	m.updateHost();
	for (size_t i = 0; i < 600; ++i)
	{
		const float want = -2.f + (float)(bits62(reference) >> 38) * (1.0f / 16777216.0f) * 8.f;
		REQUIRE(m.getAddress()[i] == want);
	}
}

TEST_CASE("randomize, the stream: the program's, a set one, a scope, and the order of the calls")
{
	skepu::PRNG a(3), b(3), other(99);
	skepu::Vector<int> first(100), second(100), both(200);

	{
		skepu::randomize::Scope scope(a);
		skepu::randomize::uniform(first, 1000);
		skepu::randomize::uniform(second, 1000); // the stream goes on
	}
	skepu::randomize::set_prng(b);
	skepu::randomize::uniform(both, 1000);
	skepu::randomize::reset_prng();

	first.updateHost(); second.updateHost(); both.updateHost();
	for (size_t i = 0; i < 100; ++i)
	{
		REQUIRE(both(i) == first(i));
		REQUIRE(both(100 + i) == second(i));
	}

	// without a stream set, the one of the program is used (and advanced)
	skepu::PRNG &global = skepu::getGlobalPRNG();
	skepu::PRNG expected(SKEPU_RAND_DEFAULT_SEED);
	REQUIRE(&skepu::randomize::prng() == &global);
	skepu::Vector<int> c(10);
	skepu::randomize::uniform(c, 10);
	c.updateHost();
	for (size_t i = 0; i < 10; ++i)
		REQUIRE(c(i) == (int)(bits62(expected) % 11));

	// a scope restores what was before it
	skepu::randomize::set_prng(a);
	{
		skepu::randomize::Scope scope(other);
		REQUIRE(&skepu::randomize::prng() == &other);
	}
	REQUIRE(&skepu::randomize::prng() == &a);
	skepu::randomize::reset_prng();
}

TEST_CASE("randomize::normal, the moments")
{
	const size_t n = 200000;
	skepu::PRNG stream(13);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<float> v(n);
	skepu::randomize::normal(v, 3.f, 2.f);
	v.updateHost();
	double sum = 0, sum2 = 0;
	for (size_t i = 0; i < n; ++i) { sum += v(i); sum2 += (double)v(i) * v(i); }
	const double mean = sum / n, variance = sum2 / n - mean * mean;
	REQUIRE(std::fabs(mean - 3.0) < 0.03);
	REQUIRE(std::fabs(std::sqrt(variance) - 2.0) < 0.03);

	// the standard normal is the default, and the shape does not matter
	skepu::Matrix<float> m(400, 500);
	skepu::randomize::normal(m);
	m.updateHost();
	sum = sum2 = 0;
	for (size_t i = 0; i < n; ++i) { sum += m.getAddress()[i]; sum2 += (double)m.getAddress()[i] * m.getAddress()[i]; }
	REQUIRE(std::fabs(sum / n) < 0.02);
	REQUIRE(std::fabs(std::sqrt(sum2 / n) - 1.0) < 0.02);
}

TEST_CASE("randomize::bernoulli and randomize::poisson, the moments")
{
	const size_t n = 100000;
	skepu::PRNG stream(17);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<int> b(n);
	skepu::randomize::bernoulli(b, 0.3);
	b.updateHost();
	size_t ones = 0;
	for (size_t i = 0; i < n; ++i) { REQUIRE((b(i) == 0 || b(i) == 1)); ones += b(i); }
	REQUIRE(std::fabs((double)ones / n - 0.3) < 0.01);

	skepu::randomize::bernoulli(b, 0.0);
	b.updateHost();
	for (size_t i = 0; i < n; ++i) REQUIRE(b(i) == 0);
	skepu::randomize::bernoulli(b, 1.0);
	b.updateHost();
	for (size_t i = 0; i < n; ++i) REQUIRE(b(i) == 1);

	skepu::Vector<int> p(n);
	skepu::randomize::poisson(p, 4.0);
	p.updateHost();
	double sum = 0, sum2 = 0;
	for (size_t i = 0; i < n; ++i) { REQUIRE(p(i) >= 0); sum += p(i); sum2 += (double)p(i) * p(i); }
	const double mean = sum / n;
	REQUIRE(std::fabs(mean - 4.0) < 0.05);
	REQUIRE(std::fabs((sum2 / n - mean * mean) - 4.0) < 0.1);
}
