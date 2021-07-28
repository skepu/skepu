#include <iostream>
#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/io.hpp>


int update_map(int value, int increment)
{
	return value + increment;
}

auto instance = skepu::Map<1>(update_map);

TEST_CASE("Map in place")
{
	const size_t size{100};

	skepu::Vector<int> values(size);
	values.randomize(0, 9);
	
	
	instance(values, values, 1);

}

