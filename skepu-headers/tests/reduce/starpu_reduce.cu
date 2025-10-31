#include "../../../external/catch2/catch.hpp"

#define SKEPU_CUDA
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

	__device__
	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int CU(int a, int b)
	{
		return a + b;
	}

	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int CPU(int a, int b)
	{
		return a + b;
	}
};

struct max_fn
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
		return a < b ? b : a;
	}

	__device__
	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int CU(int a, int b)
	{
		return a < b ? b : a;
	}

	static inline SKEPU_ATTRIBUTE_FORCE_INLINE int CPU(int a, int b)
	{
		return a < b ? b : a;
	}
};

__managed__ bool sum_kernel_called;
__global__
void sum_kernel(int * in, int * out, size_t count, size_t, bool)
{
	sum_kernel_called = true;
	int extern __shared__ smem[];
	int tid = threadIdx.x;
	size_t offset = tid + (blockIdx.x * blockDim.x);
	int grid_size = gridDim.x * blockDim.x;

	if(offset < count)
		smem[tid] = in[offset];
	for(size_t idx=offset + grid_size; idx < count; idx += grid_size)
		smem[tid] += in[idx];
	__syncthreads();

	for(int idx=blockDim.x/2; idx > 0; idx /= 2)
	{
		if(tid < idx)
			smem[tid] += smem[tid + idx];
		__syncthreads();
	}

	if(!tid)
	{
		out[blockIdx.x] = *smem;
	}
}

__managed__ bool max_kernel_called;
__global__
void max_kernel(int * in, int * out, size_t count, size_t, bool)
{
	max_kernel_called = true;
	int extern __shared__ smem[];
	int tid = threadIdx.x;
	size_t offset = tid + (blockIdx.x * blockDim.x);
	int grid_size = gridDim.x * blockDim.x;

	if(offset < count)
		smem[tid] = in[offset];
	for(size_t idx=offset + grid_size; idx < count; idx += grid_size)
		smem[tid] = max_fn::CU(smem[tid], in[idx]);
	__syncthreads();

	for(int idx=blockDim.x/2; idx > 0; idx /= 2)
	{
		if(tid < idx)
			smem[tid] = max_fn::CU(smem[tid], smem[tid + idx]);
		__syncthreads();
	}

	if(!tid)
	{
		out[blockIdx.x] = *smem;
	}
}

TEST_CASE("Elwise sum")
{
	skepu::backend::Reduce1D<sum_fn, decltype(&sum_kernel), void> red(sum_kernel);
	red.setBackend(skepu::Backend::Type::CUDA);

	SECTION("with Vector")
	{
		int const N = 11 * skepu::cluster::mpi_size();
		int const expected = (N*(N -1))/2;
		skepu::Vector<int> v(N);

		v.flush();
		for(int i(0); i < N; ++i)
			v(i) = i;

		sum_kernel_called = false;
		int res{0};
		REQUIRE_NOTHROW(res = red(v));
		if(skepu::cluster::mpi_size() == 1)
			REQUIRE(sum_kernel_called);
		CHECK(res == expected);
	}
}

TEST_CASE("Matrix 2D reduction")
{
	size_t const C{10};
	size_t const R{10 * skepu::cluster::mpi_size()};
	skepu::Matrix<int> m(R, C);
	skepu::backend::Reduce2D<
			sum_fn,
			max_fn,
			decltype(&sum_kernel),
			decltype(&max_kernel),
			void>
		sum(sum_kernel, max_kernel);
	sum.setBackend(skepu::Backend::Type::CUDA);

	m.flush();
	for(size_t i(0); i < R; ++i)
		for(size_t j(0); j < C; ++j)
			m(i,j) = i*j;

	SECTION("Rowwise")
	{
		sum_kernel_called=false;
		max_kernel_called=false;
		int res;
		REQUIRE_NOTHROW(res = sum(m));
		if(skepu::cluster::mpi_size() == 1)
		{
			REQUIRE(sum_kernel_called);
			REQUIRE(max_kernel_called);
		}
		CHECK(res == (R -1)*(C*(C -1))/2);
	}

	SECTION("Colwise")
	{
		sum_kernel_called=false;
		max_kernel_called=false;
		int res;
		sum.setReduceMode(skepu::ReduceMode::ColWise);
		REQUIRE_NOTHROW(res = sum(m));
		if(skepu::cluster::mpi_size() == 1)
		{
			REQUIRE(sum_kernel_called);
			REQUIRE(max_kernel_called);
		}
		CHECK(res == (C -1)*(R*(R -1))/2);
	}
}
