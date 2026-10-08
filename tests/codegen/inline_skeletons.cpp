#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

// Anonymous skeleton instances: a skeleton that is created by the factory function and called in the same
// expression, with settings in between where it needs them. The precompiler generates one (static) instance for
// each such expression, which is created once, whatever the number of calls.

namespace
{
	float add(float a, float b)
	{
		return a + b;
	}

	float square(float a)
	{
		return a * a;
	}

	float scaled(skepu::Index1D index, float a)
	{
		return a * float(index.i);
	}

	float box(skepu::Region2D<float> r)
	{
		float sum = 0;
		for (int i = -r.oi; i <= r.oi; ++i)
			for (int j = -r.oj; j <= r.oj; ++j)
				sum += r(i, j);
		return sum;
	}

	float stencil1d(skepu::Region1D<float> r)
	{
		return r(-1) + r(0) + r(1);
	}

	// the same skeleton and user function in another function
	void squares(skepu::Vector<float> &out, skepu::Vector<float> &in)
	{
		skepu::Map<1>(square)(out, in);
	}

	skepu::Vector<float> iota(size_t size)
	{
		skepu::Vector<float> v(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = float(i);
		return v;
	}
}

// (a named instance beside the anonymous ones)
auto named_square = skepu::Map<1>(square);

TEST_CASE("Anonymous Map")
{
	const size_t size{32};
	auto a = iota(size), b = iota(size);
	skepu::Vector<float> out(size), out_named(size), out_other(size);

	skepu::Map<2>(add)(out, a, b);
	named_square(out_named, a);
	squares(out_other, a);

	out.flush(); out_named.flush(); out_other.flush();
	for (size_t i = 0; i < size; ++i)
	{
		CHECK(out(i) == 2.f * i);
		CHECK(out_named(i) == float(i) * i);
		CHECK(out_other(i) == float(i) * i);
	}
}

TEST_CASE("Anonymous Map called in a loop, with a different expression for each iteration")
{
	const size_t size{16};
	skepu::Vector<float> out(size);
	auto in = iota(size);
	std::vector<float> expected(size);
	for (size_t i = 0; i < size; ++i)
		expected[i] = float(i);

	for (int round = 1; round <= 3; ++round)
	{
		skepu::Map<1>(scaled)(out, in);
		out.flush();
		for (size_t i = 0; i < size; ++i)
		{
			expected[i] *= float(i);
			CHECK(out(i) == expected[i]);
		}
		in = out;
	}
}

TEST_CASE("Anonymous MapOverlap with settings")
{
	const size_t size{12};
	skepu::Matrix<float> in(size, size, 1.f), out(size, size);

	skepu::MapOverlap(box).setOverlap(1).setEdgeMode(skepu::Edge::Pad).setPad(0.f)(out, in);

	out.flush();
	for (size_t y = 0; y < size; ++y)
		for (size_t x = 0; x < size; ++x)
		{
			const bool top = (y == 0 || y == size - 1), side = (x == 0 || x == size - 1);
			const float expected = (top && side) ? 4.f : (top || side) ? 6.f : 9.f;
			CHECK(out(y, x) == expected);
		}
}

TEST_CASE("Anonymous MapOverlap, the settings are made at every call")
{
	const size_t size{10};
	skepu::Matrix<float> in(size, size, 1.f), out(size, size);

	// the same expression with a different pad value in each iteration (the pad is at the corner)
	for (float pad : {0.f, 1.f, 2.f})
	{
		skepu::MapOverlap(box).setOverlap(1).setEdgeMode(skepu::Edge::Pad).setPad(pad)(out, in);
		out.flush();
		CHECK(out(0, 0) == 4.f + 5.f * pad);
		CHECK(out(size / 2, size / 2) == 9.f);
	}
}

TEST_CASE("Anonymous Reduce with a start value")
{
	const size_t size{10};
	auto a = iota(size);

	// (a Scalar is not comparable in Catch)
	const float plain = skepu::Reduce(add)(a);
	const float started = skepu::Reduce(add).setStartValue(100.f)(a);
	CHECK(plain == 45.f);
	CHECK(started == 145.f);
}

TEST_CASE("Anonymous Map with a backend setting")
{
	const size_t size{8};
	auto a = iota(size);
	skepu::Vector<float> out(size);

	skepu::Map<1>(square).setBackend(skepu::BackendSpec{"cpu"})(out, a);

	out.flush();
	for (size_t i = 0; i < size; ++i)
		CHECK(out(i) == float(i) * i);
}
