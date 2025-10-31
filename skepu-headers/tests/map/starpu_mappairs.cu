#include "skepu3/cluster/cluster.hpp"
#include "skepu3/impl/backend.hpp"
#include "../../../external/catch2/catch.hpp"
#include <cuda.h>

#define SKEPU_CUDA
#include <skepu3/cluster/skeletons/map/mappairs.hpp>
#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>

struct HVScale
{
	constexpr static size_t totalArity = 0;
	constexpr static size_t outArity = 1;
	constexpr static bool indexed = 0;
	using ElwiseArgs = std::tuple<>;
	using ContainerArgs = std::tuple<>;
	using UniformArgs = std::tuple<>;
	typedef std::tuple<> ProxyTags;
	constexpr static skepu::AccessMode anyAccessMode[] = {};
	using Ret = int;
	constexpr static bool prefersMatrix = 0;

	__device__
	auto static
	CU(int a, int b)
	-> int
	{
		return a * b;
	}

	auto static
	OMP(int a, int b)
	-> int
	{
		return a * b;
	}

	auto static
	CPU(int a, int b)
	-> int
	{
		return a * b;
	}
};

__managed__ bool HVScale_cu_called;
__global__
void HVScaleKernel(
	int * res,
	int * a,
	int * b,
	size_t height,
	size_t width,
	size_t start)
{
	size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(!i)
		HVScale_cu_called = true;

	for(; i < height * width; i += blockDim.x * gridDim.x)
		res[i] = HVScale::CU(a[i], b[i]);
}

TEST_CASE("simple matrix init")
{
	auto map =
		skepu::backend::MapPairs<1,1, HVScalem, decltype(HVScaleKernel), void>(
			HVScaleKernel);
	skepu::setGlobalBackendSpec("cuda");

	skepu::Vector<int> a(40 * skepu::cluster::mpi_size());
	skepu::Vector<int> b(a.size());
	skepu::Matrix<int> res(a.size(), b.size());
	skepu::Matrix<int> expected(a.size(), b.size());

	a.flush();
	b.flush();
	for(size_t i{0}; i < a.size(); ++i)
	{
		a(i) = i;
		b(i) = i;

		for(size_t j{0}; j < b.size(); ++j)
			expected(i,j) = i * j;
	}

	auto old_max = skepu::max_filter_block_size;
	skepu::max_filter_block_size =
		10 * skepu::cluster::mpi_size() * sizeof(int);

	HVScale_cu_called = false;
	REQUIRE_NOTHROW(map(res, a, b));

	if(skepu::cluster::mpi_size() == 1)
		REQUIRE(HVScale_cu_called);
	REQUIRE(
		skepu::cont::getParent(a).handle_for(0)
		!= skepu::cont::getParent(a).handle_for(10));

	skepu::max_filter_block_size = old_max;

	for(size_t i{0}; i < a.size(); ++i)
		for(size_t j{0}; j < b.size(); ++j)
			REQUIRE(res(i,j) == expected(i,j));
}
