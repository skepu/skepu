#include "../../../external/catch2/catch.hpp"

#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/tensor3/tensor3.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/skeletons/reduce/reduce.hpp>

struct sum_fn
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

	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int OMP(int a, int b)
	{
		return a + b;
	}

	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int CPU(int a, int b)
	{
		return a + b;
	}
};

TEST_CASE("Sum of Vector")
{
	auto sum = skepu::backend::Reduce1D<sum_fn, bool, void>(false);
	auto constexpr N{9000ul};
	auto constexpr EXPECTED{((N -1)*N)/2};
	auto v = skepu::Vector<int>(N);
	v.flush();
	for(size_t i(0); i < N; ++i)
		v(i) = i;

	int res(0);
	REQUIRE_NOTHROW(res = sum(v));
	CHECK(res == EXPECTED);
}

TEST_CASE("Sum of matrix")
{
	auto sum = skepu::backend::Reduce1D<sum_fn, bool, void>(false);
	auto constexpr N{90ul};
	auto constexpr NN{N * N};
	auto constexpr EXPECTED{((NN -1)*NN)/2};
	auto m = skepu::Matrix<int>(N,N, 0);
	m.flush();
	for(size_t i(1); i < NN; ++i)
		m(i/N, i%N) = i;

	int res(0);
	//if(skepu::cluster::mpi_rank()) for(;;);
	REQUIRE_NOTHROW(res = sum(m));
	CHECK(res == EXPECTED);
}

TEST_CASE("Sum of Tensor3")
{
	auto sum = skepu::backend::Reduce1D<sum_fn, bool, void>(false);
	size_t constexpr N{40};
	size_t constexpr NNN{N*N*N};
	auto constexpr EXPECTED{((NNN -1)*NNN) / 2};
	skepu::Tensor3<int> tensor(N,N,N);

	tensor.flush();
	for(size_t i(1); i < NNN; ++i)
		tensor(i) = i;

	int res;
	REQUIRE_NOTHROW(res = sum(tensor));
	REQUIRE(res == EXPECTED);
}
