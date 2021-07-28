#include <catch2/catch.hpp>

#include <skepu>
#include <math.h>

auto flatmap = skepu::Map([](int e){ return e; });

TEST_CASE("Flat map")
{
	const size_t size{10};

	skepu::Matrix<int> mat(size, size, 2);
	skepu::Tensor3<int> ten3(size, size, size, 3);
	skepu::Tensor4<int> ten4(size, size, size, size, 4);
	
	skepu::Vector<int> dest_A(size*size), dest_B(size*size*size), dest_C(size*size*size*size);
	
	
	flatmap(dest_A, mat);
	flatmap(dest_B, ten3);
	flatmap(dest_C, ten4);
	
	skepu::external(skepu::read(dest_A, dest_B, dest_C), [&]
	{
		CHECK(dest_A(0) == 2);
		CHECK(dest_A(size*size-1) == 2);
		CHECK(dest_B(0) == 3);
		CHECK(dest_B(size*size*size-1) == 3);
		CHECK(dest_C(0) == 4);
		CHECK(dest_C(size*size*size*size-1) == 4);
	});
	
}
