#include <catch2/catch_test_macros.hpp>

#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/scan.hpp>

struct max_fn
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

	int static OMP(int a, int b)
	{
		return a > b ? a : b;
	}

	int static CPU(int a, int b)
	{
		return a > b ? a : b;
	}
};

TEST_CASE("Inclusive scan")
{
	skepu::backend::Scan<max_fn, bool, bool, bool, void>
		scanner(false,false,false);
	auto const N = size_t(10*skepu::cluster::mpi_size());
	int const first_val = 1;
	skepu::Vector<int> v(N);
	skepu::Vector<int> res(N);
	v.randomize(0, 100);
	v.flush();
	v(0) = first_val;

	SECTION("with initial value is zero")
	{
		scanner(res, v);

		v.flush();
		res.flush();
		int expected(first_val);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected = std::max(expected, v(i));
			REQUIRE(res(i) == expected);
		}
	}

	SECTION("with initial value is 50")
	{
		int const start_val = 50;
		scanner.setStartValue(start_val);
		scanner(res, v);

		v.flush();
		res.flush();
		int expected(first_val);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected = std::max(expected, v(i));
			REQUIRE(res(i) == expected);
		}
	}
}

TEST_CASE("Exclusive scan")
{
	skepu::backend::Scan<max_fn, bool, bool, bool, void>
		scanner(false,false,false);
	scanner.setScanMode(skepu::ScanMode::Exclusive);
	auto const N = size_t(10*skepu::cluster::mpi_size());
	skepu::Vector<int> v(N);
	skepu::Vector<int> res(N);
	v.randomize(0,98);
	if(N > 10)
	{
		v.flush();
		v(N - 12) = 99;
		v(N - 11) = 100;
	}

	SECTION("with initial value is zero")
	{
		scanner(res, v);
		v.flush();
		res.flush();

		int expected(0);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected = std::max(expected, v(i -1));
			REQUIRE(res(i) == expected);
		}
	}

	SECTION("with initial value is 50")
	{
		scanner.setStartValue(50);
		scanner(res, v);
		v.flush();
		res.flush();

		int expected(50);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected = std::max(expected, v(i -1));
			REQUIRE(res(i) == expected);
		}
	}
}

struct sum_fn
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

	int static OMP(int a, int b)
	{
		return a + b;
	}

	int static CPU(int a, int b)
	{
		return a + b;
	}
};

TEST_CASE("Inclusive prefix sum")
{
	skepu::backend::Scan<sum_fn, bool, bool, bool, void>
		scanner(false,false,false);
	auto const N = size_t(10*skepu::cluster::mpi_size());
	int const first_val = 1;
	skepu::Vector<int> v(N);
	skepu::Vector<int> res(N);
	v.randomize(0, 100);
	v.flush();
	v(0) = first_val;

	SECTION("with initial value is zero")
	{
		scanner(res, v);

		v.flush();
		res.flush();

		int expected(first_val);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected = expected +v(i);
			REQUIRE(res(i) == expected);
		}
	}

	SECTION("with initial value is 50")
	{
		int const start_val = 50;
		scanner.setStartValue(start_val);
		scanner(res, v);

		v.flush();
		res.flush();

		int expected(first_val);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected = expected + v(i);
			REQUIRE(res(i) == expected);
		}
	}
}

TEST_CASE("Exclusive prefix sum")
{
	skepu::backend::Scan<sum_fn, bool, bool, bool, void>
		scanner(false,false,false);
	scanner.setScanMode(skepu::ScanMode::Exclusive);
	auto const N = size_t(10*skepu::cluster::mpi_size());
	int const first_val = 1;
	skepu::Vector<int> v(N);
	skepu::Vector<int> res(N);
	v.randomize(0, 100);
	v.flush();
	v(0) = first_val;

	SECTION("with initial value is zero")
	{
		scanner(res, v);

		v.flush();
		res.flush();

		int expected(0);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected += v(i -1);
			REQUIRE(res(i) == expected);
		}
	}

	SECTION("with initial value is 50")
	{
		int const start_val = 50;
		scanner.setStartValue(start_val);
		scanner(res, v);

		v.flush();
		res.flush();

		int expected(start_val);
		REQUIRE(res(0) == expected);
		for(size_t i(1); i < N; ++i)
		{
			expected += v(i -1);
			REQUIRE(res(i) == expected);
		}
	}
}
