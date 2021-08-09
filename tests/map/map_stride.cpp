#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>
#include <skepu-lib/io.hpp>

int add(int a, int b)
{
	return a + b;
}

skepu::multiple<int, int> multi(int a, int b)
{
	return skepu::ret(a + b, a - b);
}

auto addr = skepu::Map(add);
auto addrsum = skepu::MapReduce(add, add);
auto multiple = skepu::Map(multi);


TEST_CASE("Map with stride access")
{
	constexpr size_t N{8};
	skepu::Vector<int> v1(N*4), v2(N*3), r(N*2, -1), r2(N, -1);
	skepu::Vector<int> rb = r, rc = r;
	skepu::external([&]
	{ 
		for (size_t i = 0; i < v1.size(); ++i) v1(i) = i * 100;
		for (size_t i = 0; i < v2.size(); ++i) v2(i) = i;
		std::cout << "v1: " << v1 << "\nv2: " << v2 << "\n";
	}, skepu::write(v1, v2));
	
	// MAP, POSITIVE STRIDES
	addr.setStride(2, 4, 3);
	addr(r, v1, v2);
	
	skepu::io::cout << "r: " << r << "\n";
	
	r.flush();
	for (size_t i = 0; i < N; ++i)
		CHECK(r(i*2) == (i*4*100 + i*3));
	
	
	// MAPREDUCE, POSITIVE STRIDES
	addrsum.setStride(4, 3);
	int res = addrsum(v1, v2);
	
	skepu::io::cout << "res: " << res << "\n";
	
	int expected = 0;
	for (size_t i = 0; i < N; ++i)
		expected += (i*4*100 + i*3);
	CHECK(expected == res);
	
	
	// MULTI-RETURN MAP, POSITIVE STRIDES
	multiple.setStride(2, 1, 4, 3);
	multiple(rb, r2, v1, v2);
	
	skepu::io::cout << "r:  " << rb << "\nr2: " << r2 << "\n";
	
	rb.flush();
	r2.flush();
	for (size_t i = 0; i < N; ++i)
	{
		CHECK(rb(i*2) == (i*4*100 + i*3));
		CHECK(r2(i) == (i*4*100 - i*3));
	}
	
	
	// MAP, NEGATIVE STRIDES
	addr.setStride(2, -4, -3);
	addr(rc, v1, v2);
	
	skepu::io::cout << "r: " << rc << "\n";
	
	rc.flush();
	for (size_t i = 0; i < N; ++i)
		CHECK(rc(i*2) == ((N-i-1)*4*100 + (N-i-1)*3));
	
}




TEST_CASE("Preserve non-accessed data in output containers")
{
	constexpr size_t N{8};
	skepu::Vector<int> v1(N*4), v2(N*3), r(N*2, -1);
	skepu::external([&]
	{ 
		for (size_t i = 0; i < r.size(); ++i) r(i) = i*7+3;
	}, skepu::write(r));
	
	// MAP, POSITIVE STRIDES
	addr.setStride(2, 4, 3);
	addr(r, v1, v2);
	
	r.flush();
	for (size_t i = 0; i < N; ++i)
		CHECK(r(i*2+1) == (i*2+1)*7+3);
}