// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>


// ======== inline_skeletons ========

namespace inline_skeletons_part
{
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
		return a * (float)index.i;
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

TEST_CASE("inline_skeletons: Anonymous Map")
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

TEST_CASE("inline_skeletons: Anonymous Map called in a loop, with a different expression for each iteration")
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

TEST_CASE("inline_skeletons: Anonymous MapOverlap with settings")
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

TEST_CASE("inline_skeletons: Anonymous MapOverlap, the settings are made at every call")
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

TEST_CASE("inline_skeletons: Anonymous Reduce with a start value")
{
	const size_t size{10};
	auto a = iota(size);

	// (a Scalar is not comparable in Catch)
	const float plain = skepu::Reduce(add)(a);
	const float started = skepu::Reduce(add).setStartValue(100.f)(a);
	CHECK(plain == 45.f);
	CHECK(started == 145.f);
}

TEST_CASE("inline_skeletons: Anonymous Map with a backend setting")
{
	const size_t size{8};
	auto a = iota(size);
	skepu::Vector<float> out(size);

	skepu::Map<1>(square).setBackend(skepu::BackendSpec{"cpu"})(out, a);

	out.flush();
	for (size_t i = 0; i < size; ++i)
		CHECK(out(i) == float(i) * i);
}

TEST_CASE("inline_skeletons: Anonymous MapReduce with a lambda for one of its functions and a function for the other")
{
	auto a = iota(10);

	// (the sum of the squares of 0 to 9 is 285)
	const float lambda_map = skepu::MapReduce<1>([](float x) -> float { return x * x; }, add)(a);
	const float lambda_reduction = skepu::MapReduce<1>(square, [](float x, float y) -> float { return x + y; })(a);
	const float both_lambdas = skepu::MapReduce<1>([](float x) -> float { return x * x; }, [](float x, float y) -> float { return x + y; })(a);
	CHECK(lambda_map == 285.f);
	CHECK(lambda_reduction == 285.f);
	CHECK(both_lambdas == 285.f);
}
}


// ======== inline_skeletons_all ========

namespace inline_skeletons_all_part
{
// Anonymous skeleton instances of the other skeletons: each is compared with a named instance that has the same
// settings (made with separate statements). See inline_skeletons.cpp for Map, MapOverlap 2D and Reduce.

namespace
{
	int mul(int a, int b) { return a * b; }
	int sum(int a, int b) { return a + b; }

	int indexed_product(skepu::Index2D i, int a, int b) { return a * b + (int)i.row; }
	int index_only(skepu::Index2D i, int u) { return (int)i.row * 10 + (int)i.col + u; }

	int stencil(skepu::Region1D<int> r) { return r(-1) + r(0) + r(1); }
	int stencil3(skepu::Region3D<int> r) { return r(0, 0, 0) + r(1, 0, 0) + r(0, 1, 0) + r(0, 0, 1); }
	int pool_sum(skepu::Pool2D<int> p)
	{
		int val = 0;
		for (int row = 0; row < p.si; ++row)
			for (int col = 0; col < p.sj; ++col)
				val += p(row, col);
		return val;
	}

	skepu::Vector<int> iota(size_t size, int first = 1)
	{
		skepu::Vector<int> v(size);
		for (size_t i = 0; i < size; ++i)
			v(i) = first + int(i);
		return v;
	}

	template<typename C>
	void same(C &a, C &b)
	{
		a.flush(); b.flush();
		REQUIRE(a.size() == b.size());
		for (size_t i = 0; i < a.size(); ++i)
			CHECK(a[i] == b[i]);
	}
}

TEST_CASE("inline_skeletons_all: Anonymous MapReduce")
{
	auto a = iota(20), b = iota(20, 3);

	auto named = skepu::MapReduce<2>(mul, sum);
	named.setStartValue(10);
	const int expected = named(a, b);
	const int chained = skepu::MapReduce<2>(mul, sum).setStartValue(10)(a, b);
	const int plain = skepu::MapReduce<2>(mul, sum)(a, b);

	CHECK(chained == expected);
	CHECK(chained == plain + 10);
}

TEST_CASE("inline_skeletons_all: Anonymous Scan")
{
	const size_t size{32};
	auto in = iota(size);
	skepu::Vector<int> expected(size), chained(size), plain(size);

	auto named = skepu::Scan(sum);
	named.setScanMode(skepu::ScanMode::Exclusive);
	named.setStartValue(5);
	named(expected, in);

	skepu::Scan(sum).setScanMode(skepu::ScanMode::Exclusive).setStartValue(5)(chained, in);
	skepu::Scan(sum)(plain, in);

	same(chained, expected);
	plain.flush();
	CHECK(plain(size - 1) == int(size * (size + 1) / 2));
}

TEST_CASE("inline_skeletons_all: Anonymous MapPairs")
{
	const size_t rows{12}, cols{7};
	auto v = iota(rows), h = iota(cols, 10);
	skepu::Matrix<int> expected(rows, cols), chained(rows, cols);

	auto named = skepu::MapPairs(mul);
	named(expected, v, h);
	skepu::MapPairs(mul).setBackend(skepu::BackendSpec{"cpu"})(chained, v, h);
	same(chained, expected);

	// a size instead of containers
	skepu::Matrix<int> sized_expected(rows, cols), sized_chained(rows, cols);
	auto named_sized = skepu::MapPairs<0, 0>(index_only);
	named_sized.setDefaultSize(rows, cols);
	named_sized(sized_expected, 3);
	skepu::MapPairs<0, 0>(index_only).setDefaultSize(rows, cols)(sized_chained, 3);
	same(sized_chained, sized_expected);
}

TEST_CASE("inline_skeletons_all: Anonymous MapPairsReduce")
{
	const size_t rows{9}, cols{6};
	auto v = iota(rows), h = iota(cols, 2);
	skepu::Vector<int> expected(rows), chained(rows);

	auto named = skepu::MapPairsReduce(indexed_product, sum);
	named.setReduceMode(skepu::ReduceMode::RowWise);
	named.setStartValue(100);
	named(expected, v, h);

	skepu::MapPairsReduce(indexed_product, sum).setReduceMode(skepu::ReduceMode::RowWise).setStartValue(100)(chained, v, h);
	same(chained, expected);
}

TEST_CASE("inline_skeletons_all: Anonymous MapOverlap 1D")
{
	const size_t size{20};
	auto in = iota(size);
	skepu::Vector<int> out(size);

	skepu::MapOverlap(stencil).setOverlap(1).setEdgeMode(skepu::Edge::Pad).setPad(0)(out, in);

	out.flush();
	for (size_t i = 0; i < size; ++i)
	{
		const int prev = (i == 0) ? 0 : int(i);
		const int next = (i == size - 1) ? 0 : int(i) + 2;
		CHECK(out(i) == prev + int(i) + 1 + next);
	}
}

TEST_CASE("inline_skeletons_all: Anonymous MapOverlap 3D")
{
	const size_t n{5};
	skepu::Tensor3<int> in(n, n, n), expected(n, n, n), chained(n, n, n);
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			for (size_t k = 0; k < n; ++k)
				in(i, j, k) = int(i * 25 + j * 5 + k);

	auto named = skepu::MapOverlap(stencil3);
	named.setOverlap(1);
	named.setEdgeMode(skepu::Edge::Pad);
	named.setPad(-1);
	named(expected, in);

	skepu::MapOverlap(stencil3).setOverlap(1).setEdgeMode(skepu::Edge::Pad).setPad(-1)(chained, in);
	same(chained, expected);
}

TEST_CASE("inline_skeletons_all: Anonymous Reduce of a matrix")
{
	const size_t rows{6}, cols{9};
	skepu::Matrix<int> m(rows, cols);
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			m(i, j) = int(i * cols + j);

	// one user function: a vector of the sums of the columns (or rows)
	skepu::Vector<int> expected(cols), chained(cols);
	auto named = skepu::Reduce(sum);
	named.setReduceMode(skepu::ReduceMode::ColWise);
	named(expected, m);
	skepu::Reduce(sum).setReduceMode(skepu::ReduceMode::ColWise)(chained, m);
	same(chained, expected);

	// two user functions (Reduce2D): a single value
	auto named_2d = skepu::Reduce(sum, mul);
	named_2d.setReduceMode(skepu::ReduceMode::RowWise);
	named_2d.setStartValue(1);
	const int expected_2d = named_2d(m);
	const int chained_2d = skepu::Reduce(sum, mul).setReduceMode(skepu::ReduceMode::RowWise).setStartValue(1)(m);
	CHECK(chained_2d == expected_2d);
}

TEST_CASE("inline_skeletons_all: Anonymous MapPool")
{
	const size_t n{8};
	skepu::Matrix<int> in(n, n), expected(n / 2, n / 2), chained(n / 2, n / 2);
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			in(i, j) = int(i * n + j);

	auto named = skepu::MapPool(pool_sum);
	named.setPoolSizeAndStride(2);
	named(expected, in);

	skepu::MapPool(pool_sum).setPoolSizeAndStride(2)(chained, in);
	same(chained, expected);
}
}
