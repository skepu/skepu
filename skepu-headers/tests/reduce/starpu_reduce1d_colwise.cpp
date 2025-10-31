#include "skepu3/cluster/cluster.hpp"
#include "skepu3/cluster/skeletons/reduce/reduce_mode.hpp"
#include "../../../external/catch2/catch.hpp"

#include <skepu3/cluster/containers/matrix/matrix.hpp>
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

TEST_CASE("Column wise sum of matrix")
{
	size_t const N{10 * skepu::cluster::mpi_size()};
	skepu::backend::Reduce1D<sum_fn, bool, void> sum(false);
	skepu::Matrix<int> m(N,N);
	skepu::Vector<int> res(N);

	m.flush();
	for(size_t i(0); i < N; ++i)
		for(size_t j(0); j < N; ++j)
			m(i,j) = j;

	sum.setReduceMode(skepu::ReduceMode::ColWise);
	REQUIRE_NOTHROW(sum(res, m));
	res.flush();
	for(size_t i(0); i < N; ++i)
		REQUIRE(res(i) == N *i);
}
