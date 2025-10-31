#include "../../../external/catch2/catch.hpp"

#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/map/mapreduce.hpp>

struct scalar_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(int i)
	-> int
	{
		return i;
	}

	auto static inline
	CPU(int i)
	-> int
	{
		return i;
	}
};

struct add_int
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

	auto static inline
	OMP(int a, int b)
	-> int
	{
		return a + b;
	}

	auto static inline
	CPU(int a, int b)
	-> int
	{
		return a + b;
	}
};

TEST_CASE("1D Scalar MapReduce")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapReduce<0, scalar_fn, add_int, bool, bool, void>(
			false, false));
	auto map_red =
		skepu::backend::MapReduce<0, scalar_fn, add_int, bool, bool, void>(
			false, false);
	int static constexpr value = 10;

	SECTION("1D size 0")
	{
		map_red.setDefaultSize(0);
		auto res = map_red(value);
		CHECK(res == 0);
	}

	SECTION("1D size 1")
	{
		map_red.setDefaultSize(1);
		auto res = map_red(value);
		CHECK(res == 10);
	}

	SECTION("1D size 2")
	{
		map_red.setDefaultSize(2);
		auto res = map_red(value);
		CHECK(res == 20);
	}

	SECTION("1D size 1000")
	{
		map_red.setDefaultSize(1000);
		auto res = map_red(value);
		CHECK(res == 10000);
	}
}

struct copy_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using IndexType = void;
	using ElwiseArgs = std::tuple<int>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	OMP(int i)
	-> int
	{
		return i;
	}

	auto static inline
	CPU(int i)
	-> int
	{
		return i;
	}
};

TEST_CASE("Elementwise Vector MapReduce")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapReduce<
			1, copy_fn, add_int, bool, bool, void>(false, false));
	auto mapreduce =	skepu::backend::MapReduce<
			1, copy_fn, add_int, bool, bool, void>(false, false);

	SECTION("vector size 1")
	{
		auto v = skepu::Vector<int>(1, 0);
		int expected = 0;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("vector size 2")
	{
		auto v = skepu::Vector<int>(2);
		v.flush();
		v(0) = 0;
		v(1) = 1;
		int expected = 1;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("vector size 10")
	{
		auto v = skepu::Vector<int>(10);
		int expected = 45;
		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("vector size 1000")
	{
		size_t constexpr N(1000);
		auto v = skepu::Vector<int>(N);
		int constexpr expected = ((N-1)*N)/2;
		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		auto res = mapreduce(v);
		CHECK(res == expected);
	}

	SECTION("with iterators")
	{
		size_t constexpr N(1000);
		auto v = skepu::Vector<int>(N);
		int constexpr expected = ((N-1)*N)/2;
		v.flush();
		for(size_t i(0); i < v.size(); ++i)
			v(i) = i;
		auto res = mapreduce(v.begin(), v.end());
		CHECK(res == expected);
	}
}

struct multi_return_fn
{
	constexpr static size_t totalArity = 1;
	constexpr static size_t outArity = 2;
	constexpr static bool indexed = 1;
	using IndexType = skepu::Index1D;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = std::tuple<int, int>;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	CPU(skepu::Index1D idx)
	-> std::tuple<int, int>
	{
			return std::tuple<int, int>{
				(idx.i % 10 ? 1 : 2),
				(idx.i % 20 ? 1 : 2)};
	}

	auto static inline
	OMP(skepu::Index1D idx)
	-> std::tuple<int, int>
	{
			return std::tuple<int, int>{
				(idx.i % 10 ? 1 : 2),
				(idx.i % 20 ? 1 : 2)};
	}
};

struct mult_red_fn
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
	constexpr static bool prefersMatrix = 0;

	auto static inline
	CPU(int a, int b)
	-> int
	{
		return a * b;
	}

	auto static inline
	OMP(int a, int b)
	-> int
	{
		return a * b;
	}
};

TEST_CASE("Multireturn map function")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapReduce<
			0, multi_return_fn, mult_red_fn, bool, bool, void>(false, false));
	auto map_red =
		skepu::backend::MapReduce<
			0, multi_return_fn, mult_red_fn, bool, bool, void>(false, false);

	SECTION("of size 1")
	{
		size_t constexpr N{1};
		REQUIRE_NOTHROW(
			map_red.setDefaultSize(N));
		REQUIRE_NOTHROW(
			map_red.setStartValue(std::tuple<int, int>{3, 3}));
		auto expected = std::tuple<int, int>{3, 3};
		auto res = map_red();
		for(size_t i(0); i < N; ++i)
		{
			std::get<0>(expected) *=
				(i % 10 ? 1 : 2);
			std::get<1>(expected) *=
				(i % 20 ? 1 : 2);
		}

		CHECK(std::get<0>(res) == std::get<0>(expected));
		CHECK(std::get<1>(res) == std::get<1>(expected));
	}

	SECTION("of size 10")
	{
		size_t constexpr N{10};
		REQUIRE_NOTHROW(
			map_red.setDefaultSize(N));
		REQUIRE_NOTHROW(
			map_red.setStartValue(std::tuple<int, int>{3, 3}));
		auto expected = std::tuple<int, int>{3, 3};
		auto res = map_red();
		for(size_t i(0); i < N; ++i)
		{
			std::get<0>(expected) *=
				(i % 10 ? 1 : 2);
			std::get<1>(expected) *=
				(i % 20 ? 1 : 2);
		}

		CHECK(std::get<0>(res) == std::get<0>(expected));
		CHECK(std::get<1>(res) == std::get<1>(expected));
	}

	SECTION("of size 100")
	{
		size_t constexpr N{100};
		REQUIRE_NOTHROW(
			map_red.setDefaultSize(N));
		REQUIRE_NOTHROW(
			map_red.setStartValue(std::tuple<int, int>{3, 3}));
		auto expected = std::tuple<int, int>{3, 3};
		auto res = map_red();
		for(size_t i(0); i < N; ++i)
		{
			std::get<0>(expected) *=
				(i % 10 ? 1 : 2);
			std::get<1>(expected) *=
				(i % 20 ? 1 : 2);
		}

		CHECK(std::get<0>(res) == std::get<0>(expected));
		CHECK(std::get<1>(res) == std::get<1>(expected));
	}
}

struct elwise_multi_return_fn
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
	using Ret = std::tuple<int, int>;
	constexpr static bool prefersMatrix = 0;

	auto static inline
	CPU(int a, int b)
	-> std::tuple<int, int>
	{
		return std::tuple<int, int>{a, a + b};
	}

	auto static inline
	OMP(int a, int b)
	-> std::tuple<int, int>
	{
		return std::tuple<int, int>{a, a + b};
	}
};

TEST_CASE("Elwise multireturn")
{
	REQUIRE_NOTHROW(
		skepu::backend::MapReduce<
			2, elwise_multi_return_fn, add_int, bool, bool, void>(
				false, false));
	auto map =
		skepu::backend::MapReduce<
			2, elwise_multi_return_fn, add_int, bool, bool, void>(
				false, false);

	SECTION("of size 1")
	{
		size_t constexpr N{1};
		skepu::Vector<int> v1(N);
		skepu::Vector<int> v2(N);
		int expected = ((N -1)*N)/2;
		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
			v1(i) = v2(i) = i;
		auto res = map(v1, v2);
		CHECK(std::get<0>(res) == expected);
		CHECK(std::get<1>(res) == 2 * expected);
	}

	SECTION("of size 2")
	{
		size_t constexpr N{2};
		skepu::Vector<int> v1(N);
		skepu::Vector<int> v2(N);
		int expected = ((N -1)*N)/2;
		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
			v1(i) = v2(i) = i;
		auto res = map(v1, v2);
		CHECK(std::get<0>(res) == expected);
		CHECK(std::get<1>(res) == 2 * expected);
	}

	SECTION("of size 10")
	{
		size_t constexpr N{10};
		skepu::Vector<int> v1(N);
		skepu::Vector<int> v2(N);
		int expected = ((N -1)*N)/2;
		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
			v1(i) = v2(i) = i;
		auto res = map(v1, v2);
		CHECK(std::get<0>(res) == expected);
		CHECK(std::get<1>(res) == 2 * expected);
	}

	SECTION("of size 100")
	{
		size_t constexpr N{100};
		skepu::Vector<int> v1(N);
		skepu::Vector<int> v2(N);
		int expected = ((N -1)*N)/2;
		v1.flush();
		v2.flush();
		for(size_t i(0); i < N; ++i)
			v1(i) = v2(i) = i;
		auto res = map(v1, v2);
		CHECK(std::get<0>(res) == expected);
		CHECK(std::get<1>(res) == 2 * expected);
	}
}
