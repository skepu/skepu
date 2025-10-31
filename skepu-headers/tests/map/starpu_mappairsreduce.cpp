#include <random>

#include "../../../external/catch2/catch.hpp"

#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/map/mappairsreduce.hpp>

std::mt19937 gen(1234);
std::uniform_int_distribution<int> rigen(0,10);

struct mappr_index
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 1;
	using IndexType = skepu::Index2D;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<unsigned long>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 1;

	auto inline static
	OMP(skepu::Index2D i, unsigned long width)
	-> int
	{
		return ((int)i.row * width) + i.col;
	}

	auto inline static
	CPU(skepu::Index2D i, unsigned long width)
	-> int
	{
		return ((int)i.row * width) + i.col;
	}
};

struct add
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int, int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto inline static
	OMP(int a, int b)
	-> int
	{
		return a + b;
	}

	auto inline static
	CPU(int a, int b)
	-> int
	{
		return a + b;
	}
};

TEST_CASE("Rowwise zero arity mappairs reduce")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapPairsReduce<0,0, mappr_index, add, bool, void>(
			false));
	auto mpr =
		skepu::backend::MapPairsReduce<0,0, mappr_index, add, bool, void>(
			false);
	REQUIRE_NOTHROW(mpr.setReduceMode(skepu::ReduceMode::RowWise));
	REQUIRE_NOTHROW(mpr.setDefaultSize(0,0));

	SECTION("size zero")
	{
		skepu::Vector<int> ret(0);
		REQUIRE_NOTHROW(mpr(ret, 0));
	}

	SECTION("size one")
	{
		mpr.setDefaultSize(1,1);
		skepu::Vector<int> ret(1);
		REQUIRE_NOTHROW(mpr(ret, 1));

		ret.flush();
		CHECK(ret(0) == 0);
	}

	SECTION("size 2x2")
	{
		mpr.setDefaultSize(2,2);
		skepu::Vector<int> ret(2);
		REQUIRE_NOTHROW(mpr(ret, ret.size()));

		ret.flush();
		CHECK(ret(0) == 1);
		CHECK(ret(1) == 5);
	}

	SECTION("size 100x100")
	{
		size_t constexpr N{100};
		mpr.setDefaultSize(N,N);
		skepu::Vector<int> ret(N);
		REQUIRE_NOTHROW(mpr(ret, N));

		std::vector<int> expected(N, 0);
		for(size_t row(0); row < N; ++row)
			for(size_t col(0); col < N; ++col)
				expected[row] += (row * N) + col;

		ret.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}
}

TEST_CASE("Colwise zero arity mappairs reduce")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapPairsReduce<0,0, mappr_index, add, bool, void>(
			false));
	auto mpr =
		skepu::backend::MapPairsReduce<0,0, mappr_index, add, bool, void>(
			false);
	REQUIRE_NOTHROW(mpr.setReduceMode(skepu::ReduceMode::ColWise));
	REQUIRE_NOTHROW(mpr.setDefaultSize(0,0));

	SECTION("size zero")
	{
		skepu::Vector<int> ret(0);
		REQUIRE_NOTHROW(mpr(ret, 0));
	}

	SECTION("size one")
	{
		mpr.setDefaultSize(1,1);
		skepu::Vector<int> ret(1);
		REQUIRE_NOTHROW(mpr(ret, 1));

		ret.flush();
		CHECK(ret(0) == 0);
	}

	SECTION("size 2x2")
	{
		mpr.setDefaultSize(2,2);
		skepu::Vector<int> ret(2);
		REQUIRE_NOTHROW(mpr(ret, ret.size()));

		ret.flush();
		CHECK(ret(0) == 2);
		CHECK(ret(1) == 4);
	}

	SECTION("size 100x100")
	{
		size_t constexpr N{100};
		mpr.setDefaultSize(N,N);
		skepu::Vector<int> ret(N);
		REQUIRE_NOTHROW(mpr(ret, N));

		std::vector<int> expected(N, 0);
		for(size_t col(0); col < N; ++col)
			for(size_t row(0); row < N; ++row)
				expected[col] += (row * N) + col;

		ret.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}
}

struct mappr_elwise
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<int, int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 1;

	auto inline static
	OMP(int a, int b)
	-> int
	{
		return a * b;
	}

	auto inline static
	CPU(int a, int b)
	-> int
	{
		return a * b;
	}
};

TEST_CASE("Rowwise reduce with elwise map")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapPairsReduce<1,1, mappr_elwise, add, bool, void>(
			false));
	auto mppr =
		skepu::backend::MapPairsReduce<1,1, mappr_elwise, add, bool, void>(
			false);
	REQUIRE_NOTHROW(mppr.setReduceMode(skepu::ReduceMode::RowWise));
	REQUIRE_NOTHROW(mppr.setDefaultSize(0,0));

	SECTION("size 0x0")
	{
		skepu::Vector<int> ret(0);
		skepu::Vector<int> v1(0);
		skepu::Vector<int> v2(0);
		REQUIRE_NOTHROW(mppr(ret, v1, v2));
	}

	SECTION("size 1x1")
	{
		skepu::Vector<int> ret(1);
		skepu::Vector<int> v1(1, rigen(gen));
		skepu::Vector<int> v2(1, rigen(gen));

		v1.flush();
		v2.flush();
		int expected = v1(0) * v2(0);

		REQUIRE_NOTHROW(mppr(ret, v1, v2));

		ret.flush();
		CHECK(ret(0) == expected);
	}

	SECTION("size 2x2")
	{
		size_t constexpr N{2};
		skepu::Vector<int> ret(N);
		std::vector<int> expected(N,0);
		skepu::Vector<int> v1(N, rigen(gen));
		skepu::Vector<int> v2(N, rigen(gen));

		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
		{
			v1(i) = rigen(gen);
			v2(i) = rigen(gen);
		}
		skepu::cont::getParent(v1).partition();
		skepu::cont::getParent(v2).partition();
		v1.flush();
		v2.flush();
		for(size_t row(0); row < N; ++row)
			for(size_t col(0); col < N; ++col)
				expected[row] += v1(row) * v2(col);

		REQUIRE_NOTHROW(mppr(ret, v1, v2));

		ret.flush();
		for(size_t i; i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}

	SECTION("size 100x100")
	{
		size_t constexpr N{100};
		skepu::Vector<int> ret(N);
		std::vector<int> expected(N,0);
		skepu::Vector<int> v1(N, rigen(gen));
		skepu::Vector<int> v2(N, rigen(gen));

		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
		{
			v1(i) = rigen(gen);
			v2(i) = rigen(gen);
		}
		skepu::cont::getParent(v1).partition();
		skepu::cont::getParent(v2).partition();
		v1.flush();
		v2.flush();
		for(size_t row(0); row < N; ++row)
			for(size_t col(0); col < N; ++col)
				expected[row] += v1(row) * v2(col);

		REQUIRE_NOTHROW(mppr(ret, v1, v2));

		ret.flush();
		for(size_t i; i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}
}

TEST_CASE("Colwise reduce with elwise map")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapPairsReduce<1,1, mappr_elwise, add, bool, void>(
			false));
	auto mppr =
		skepu::backend::MapPairsReduce<1,1, mappr_elwise, add, bool, void>(
			false);
	REQUIRE_NOTHROW(mppr.setReduceMode(skepu::ReduceMode::ColWise));
	REQUIRE_NOTHROW(mppr.setDefaultSize(0,0));

	SECTION("size 0x0")
	{
		skepu::Vector<int> ret(0);
		skepu::Vector<int> v1(0);
		skepu::Vector<int> v2(0);
		REQUIRE_NOTHROW(mppr(ret, v1, v2));
	}

	SECTION("size 1x1")
	{
		skepu::Vector<int> ret(1);
		skepu::Vector<int> v1(1, rigen(gen));
		skepu::Vector<int> v2(1, rigen(gen));

		v1.flush();
		v2.flush();
		int expected = v1(0) * v2(0);

		REQUIRE_NOTHROW(mppr(ret, v1, v2));

		ret.flush();
		CHECK(ret(0) == expected);
	}

	SECTION("size 2x2")
	{
		size_t constexpr N{2};
		skepu::Vector<int> ret(N);
		std::vector<int> expected(N,0);
		skepu::Vector<int> v1(N, rigen(gen));
		skepu::Vector<int> v2(N, rigen(gen));

		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
		{
			v1(i) = rigen(gen);
			v2(i) = rigen(gen);
		}
		skepu::cont::getParent(v1).partition();
		skepu::cont::getParent(v2).partition();
		v1.flush();
		v2.flush();
		for(size_t col(0); col < N; ++col)
			for(size_t row(0); row < N; ++row)
				expected[col] += v1(row) * v2(col);

		REQUIRE_NOTHROW(mppr(ret, v1, v2));

		ret.flush();
		for(size_t i; i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}

	SECTION("size 100x100")
	{
		size_t constexpr N{100};
		skepu::Vector<int> ret(N);
		std::vector<int> expected(N,0);
		skepu::Vector<int> v1(N, rigen(gen));
		skepu::Vector<int> v2(N, rigen(gen));

		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
		{
			v1(i) = rigen(gen);
			v2(i) = rigen(gen);
		}
		skepu::cont::getParent(v1).partition();
		skepu::cont::getParent(v2).partition();
		v1.flush();
		v2.flush();
		for(size_t col(0); col < N; ++col)
			for(size_t row(0); row < N; ++row)
				expected[col] += v1(row) * v2(col);

		REQUIRE_NOTHROW(mppr(ret, v1, v2));

		ret.flush();
		for(size_t i; i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}
}

struct mult
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int, int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto inline static
	OMP(int a, int b)
	-> int
	{
		return a * b;
	}

	auto inline static
	CPU(int a, int b)
	-> int
	{
		return a * b;
	}
};

TEST_CASE("Multiplication reduce function")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapPairsReduce<0,0, mappr_index, mult, bool, void>(
			false));
	auto mppr =
		skepu::backend::MapPairsReduce<0,0, mappr_index, mult, bool, void>(
			false);
	size_t constexpr N{100};
	mppr.setDefaultSize(N,N);
	skepu::Vector<int> ret(N);

	SECTION("Rowwise start value of zero")
	{
		mppr.setReduceMode(skepu::ReduceMode::RowWise);
		mppr.setStartValue(0);

		mppr(ret, ret.size());
		ret.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(ret(i) == 0);
	}

	SECTION("Rowwise start value of one")
	{
		mppr.setReduceMode(skepu::ReduceMode::RowWise);
		mppr.setStartValue(1);
		std::vector<int> expected(N, 1);
		for(size_t row(0); row < N; ++row)
			for(size_t col(0); col < N; ++col)
				expected[row] *= (row * N) + col;

		mppr(ret, ret.size());
		ret.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}

	SECTION("Colwise start value of zero")
	{
		mppr.setReduceMode(skepu::ReduceMode::ColWise);
		mppr.setStartValue(0);
		mppr(ret, ret.size());

		ret.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(ret(i) == 0);
	}

	SECTION("Colwise start value of one")
	{
		mppr.setReduceMode(skepu::ReduceMode::ColWise);
		mppr.setStartValue(1);
		std::vector<int> expected(N, 1);
		for(size_t col(0); col < N; ++col)
			for(size_t row(0); row < N; ++row)
				expected[col] *= (row * N) + col;

		mppr(ret, ret.size());
		ret.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(ret(i) == expected[i]);
	}
}

struct multret_fn
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 2;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<int, int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = std::tuple<int,int>;
	constexpr static bool prefersMatrix = 1;

	auto inline static
	CPU(int a, int b)
	-> std::tuple<int,int>
	{
		return std::tuple<int,int>{a,b};
	}

	auto inline static
	OMP(int a, int b)
	-> std::tuple<int,int>
	{
		return std::tuple<int,int>{a,b};
	}
};

TEST_CASE("Multireturn rowwise")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapPairsReduce<1,1, multret_fn, add, bool, void>(
			false));
	auto mappr =
		skepu::backend::MapPairsReduce<1,1, multret_fn, add, bool, void>(
			false);
	REQUIRE_NOTHROW(
		mappr.setStartValue(std::tuple<int,int>{2,7}));

	SECTION("Rowwize")
	{
		REQUIRE_NOTHROW(
			mappr.setReduceMode(skepu::ReduceMode::RowWise));

		size_t constexpr N{100};
		skepu::Vector<int> v(100);
		skepu::Vector<int> h(100);
		std::vector<int> expected1(N, 2);
		int expected2{7};

		v.flush();
		h.flush();
		for(size_t i(0); i < N; ++i)
		{
			v(i) = rigen(gen);
			expected1[i] += N*v(i);
			h(i) = rigen(gen);
			expected2 += h(i);
		}

		skepu::Vector<int> res1(N);
		skepu::Vector<int> res2(N);
		mappr(res1, res2, v, h);
		res1.flush();
		res2.flush();
		for(size_t i(0); i < N; ++i)
		{
			REQUIRE(res1(i) == expected1[i]);
			REQUIRE(res2(i) == expected2);
		}
	}

	SECTION("Colwize")
	{
		REQUIRE_NOTHROW(
			mappr.setReduceMode(skepu::ReduceMode::ColWise));

		size_t constexpr N{100};
		skepu::Vector<int> v(100);
		skepu::Vector<int> h(100);
		int expected1{2};
		std::vector<int> expected2(N, 7);

		v.flush();
		h.flush();
		for(size_t i(0); i < N; ++i)
		{
			v(i) = rigen(gen);
			expected1 += v(i);
			h(i) = rigen(gen);
			expected2[i] += N*h(i);
		}

		skepu::Vector<int> res1(N);
		skepu::Vector<int> res2(N);
		mappr(res1, res2, v, h);
		res1.flush();
		res2.flush();
		for(size_t i(0); i < N; ++i)
		{
			REQUIRE(res1(i) == expected1);
			REQUIRE(res2(i) == expected2[i]);
		}
	}
}
