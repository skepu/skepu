#include "../../../external/catch2/catch.hpp"
#include <cuda.h>

#define SKEPU_CUDA
#include <skepu3/cluster/containers/matrix/matrix.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include <skepu3/cluster/containers/tensor3/tensor3.hpp>
#include <skepu3/cluster/containers/tensor4/tensor4.hpp>
#include <skepu3/impl/random.hpp>
#include <skepu3/impl/stride_list.hpp>
#include <skepu3/cluster/skeletons/map/map.hpp>

struct simple_map_fn
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
	CU()
	-> int
	{
		return 10;
	}

	auto static
	OMP()
	-> int
	{
		return 10;
	}

	auto static
	CPU()
	-> int
	{
		return 10;
	}
};

__managed__ bool simple_map_fn_cu_called;
__global__
void simple_map_fn_kernel(
	int * res,
	skepu::PRNG::Placeholder,
	size_t, size_t, size_t,
	size_t count,
	size_t,
	skepu::StrideList<1> skepu_strides)
{
	for(size_t i{0}; i < count; ++i)
		res[i] = simple_map_fn::CU();
	simple_map_fn_cu_called = true;
}

TEST_CASE("No args skeleton")
{
	skepu::backend::Map<0, simple_map_fn, decltype(&simple_map_fn_kernel), void>
		map(simple_map_fn_kernel);
	map.setBackend(skepu::Backend::Type::CUDA);
	simple_map_fn_cu_called = false;

	SECTION("using a vector")
	{
		skepu::Vector<int> v(10);
		map(v);
		v.flush();

		if(skepu::cluster::mpi_size() == 1)
			REQUIRE(simple_map_fn_cu_called == true);
		for(auto & e : v)
			REQUIRE(e == 10);
	}

	SECTION("using a vector")
	{
		skepu::Matrix<int> m(10,10);
		map(m);
		m.flush();

		if(skepu::cluster::mpi_size() == 1)
			REQUIRE(simple_map_fn_cu_called == true);
		for(auto & e : m)
			REQUIRE(e == 10);
	}

	SECTION("using a vector")
	{
		skepu::Tensor3<int> t3(10,10,10);
		map(t3);
		t3.flush();

		if(skepu::cluster::mpi_size() == 1)
			REQUIRE(simple_map_fn_cu_called == true);
		for(auto & e : t3)
			REQUIRE(e == 10);
	}

	SECTION("using a vector")
	{
		skepu::Tensor4<int> t4(10,10,10,10);
		map(t4);
		t4.flush();

		if(skepu::cluster::mpi_size() == 1)
			REQUIRE(simple_map_fn_cu_called == true);
		for(auto & e : t4)
			REQUIRE(e == 10);
	}

	SECTION("Small filter size")
	{
		auto bls_tmp = skepu::max_filter_block_size;
		skepu::max_filter_block_size = 10 * sizeof(int);
		skepu::Vector<int> v(1000);
		map(v);
		v.flush();
		skepu::max_filter_block_size = bls_tmp;

		if(skepu::cluster::mpi_size() == 1)
			REQUIRE(simple_map_fn_cu_called == true);
		for(auto & e : v)
			REQUIRE(e == 10);
	}
}
