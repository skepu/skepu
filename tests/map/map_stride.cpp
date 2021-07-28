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
	skepu::Vector<int> v1(32), v2(24), r(16, -1), r2(8, -1);
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
	
	// MAPREDUCE, POSITIVE STRIDES
	addrsum.setStride(4, 3);
	int res = addrsum(v1, v2);
	
	skepu::io::cout << "res: " << res << "\n";
	
	// MULTI-RETURN MAP, POSITIVE STRIDES
	multiple.setStride(2, 1, 4, 3);
	multiple(rb, r2, v1, v2);
	
	skepu::io::cout << "r:  " << rb << "\nr2: " << r2 << "\n";
	
	// MAP, NEGATIVE STRIDES
	addr.setStride(2, -4, -3);
	addr(rc, v1, v2);
	
	skepu::io::cout << "r: " << rc << "\n";
	
}

