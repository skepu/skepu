#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

// Anonymous skeleton instances of the other skeletons: each is compared with a named instance that has the same
// settings (made with separate statements). See inline_skeletons.cpp for Map, MapOverlap 2D and Reduce.

namespace
{
	int mul(int a, int b) { return a * b; }
	int sum(int a, int b) { return a + b; }

	int indexed_product(skepu::Index2D i, int a, int b) { return a * b + int(i.row); }
	int index_only(skepu::Index2D i, int u) { return int(i.row) * 10 + int(i.col) + u; }

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

TEST_CASE("Anonymous MapReduce")
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

TEST_CASE("Anonymous Scan")
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

TEST_CASE("Anonymous MapPairs")
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

TEST_CASE("Anonymous MapPairsReduce")
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

TEST_CASE("Anonymous MapOverlap 1D")
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

TEST_CASE("Anonymous MapOverlap 3D")
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

TEST_CASE("Anonymous Reduce of a matrix")
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

TEST_CASE("Anonymous MapPool")
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
