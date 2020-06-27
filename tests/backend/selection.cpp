#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>

int uf()
{
#if SKEPU_USING_BACKEND_OMP
	return omp_get_max_threads();
#endif
	return 0;
}

auto skel = skepu::Map<0>(uf);

TEST_CASE("OpenMP selection")
{
	skepu::Vector<int> res(1);
	
	skepu::BackendSpec spec{skepu::Backend::Type::OpenMP};
	skel.setBackend(spec);
	
	skel(res);
	CHECK(res(0) > 1);
	
	spec.setCPUThreads(1);
	skel.setBackend(spec);
	skel(res);
	CHECK(res(0) == 1);
	
	spec.setCPUThreads(10);
	skel.setBackend(spec);
	skel(res);
	CHECK(res(0) == 10);
	
}
