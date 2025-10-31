#include "../../../external/catch2/catch.hpp"

#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/reduce/reduce.hpp>

struct skepu_userfunction_red_mat_row_add
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int, int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	static inline int OMP(int a, int b)
	{
		return a + b;
	}

	static inline int CPU(int a, int b)
	{
		return a + b;
	}
};

struct skepu_userfunction_red_mat_col_max
{
	constexpr static size_t totalArity = 2;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<int, int>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	static inline int OMP(int a, int b)
	{
		return a < b ? b : a;
	}

	static inline int CPU(int a, int b)
	{
		return a < b ? b : a;
	}
};

TEST_CASE("Vector reduction")
{
	size_t const N{10 * skepu::cluster::mpi_size()};
	skepu::Vector<int> v(N);
	int expected(((N -1)*N)/2);
	skepu::backend::Reduce2D<
			skepu_userfunction_red_mat_row_add,
			skepu_userfunction_red_mat_col_max,
			bool, bool, void>
		sum(false,false);

	v.flush();
	for(size_t i(0); i < N; ++i)
		v(i) = i;

	int res;
	REQUIRE_NOTHROW(res = sum(v));
	CHECK(res == expected);
}

TEST_CASE("Matrix 1D reduction")
{
	size_t const N{10 * skepu::cluster::mpi_size()};
	skepu::Matrix<int> m(N, N);
	skepu::Vector<int> v(N);
	skepu::backend::Reduce2D<
			skepu_userfunction_red_mat_row_add,
			skepu_userfunction_red_mat_col_max,
			bool, bool, void>
		sum(false,false);

	m.flush();
	for(size_t i(0); i < N; ++i)
		for(size_t j(0); j < N; ++j)
			m(i,j) = j;

	SECTION("RowWise")
	{
		REQUIRE_NOTHROW(sum(v, m));
		v.flush();
		for(auto & e : v)
			REQUIRE(e == (N*(N -1))/2);
	}

	SECTION("ColWise")
	{
		sum.setReduceMode(skepu::ReduceMode::ColWise);
		REQUIRE_NOTHROW(sum(v, m));
		v.flush();
		for(size_t i(0); i < N; ++i)
			REQUIRE(v(i) == N *i);
	}
}

TEST_CASE("Matrix 2D reduction")
{
	size_t const C{10};
	size_t const R{10 * skepu::cluster::mpi_size()};
	skepu::Matrix<int> m(R, C);
	skepu::backend::Reduce2D<
			skepu_userfunction_red_mat_row_add,
			skepu_userfunction_red_mat_col_max,
			bool, bool, void>
		sum(false,false);

	m.flush();
	for(size_t i(0); i < R; ++i)
		for(size_t j(0); j < C; ++j)
			m(i,j) = i*j;

	SECTION("Rowwise")
	{
		int res;
		REQUIRE_NOTHROW(res = sum(m));
		CHECK(res == (R -1)*(C*(C -1))/2);
	}

	SECTION("Colwise")
	{
		int res;
		sum.setReduceMode(skepu::ReduceMode::ColWise);
		REQUIRE_NOTHROW(res = sum(m));
		CHECK(res == (C -1)*(R*(R -1))/2);
	}
}
