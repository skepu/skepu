#include "../../external/catch2/catch.hpp"

#include <algorithm>
#include <vector>
#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>

float update_mapoverlap(skepu::Region1D<float> r)
{
	return (r(-1) + r(1)) / 2;
}

auto update = skepu::MapOverlap(update_mapoverlap);

TEST_CASE("MapOverlap in place")
{
	const size_t size{100};
	
	// MapOverlap in place
	// Valid only for Red-Black update pattern with edge mode other than Cyclic
	// and as long as the result container is not also passed as a random-access input

	skepu::Vector<float> domain(size, 0);
	domain.randomize(0, 2);
	
	// The result on the host: the red elements (the odd ones) first, then the black ones (the even ones), each from the latest values
	std::vector<float> expected(size);
	for (size_t i = 0; i < size; ++i)
		expected[i] = domain(i);
	auto at = [&](long i) { return expected[std::min<long>(std::max<long>(i, 0), size - 1)]; };   // (edge mode Duplicate)
	for (size_t i = 1; i < size; i += 2)
		expected[i] = (at(i - 1) + at(i + 1)) / 2;
	for (size_t i = 0; i < size; i += 2)
		expected[i] = (at(i - 1) + at(i + 1)) / 2;
	
	update.setOverlap(1);
	update.setEdgeMode(skepu::Edge::Duplicate);
	update.setUpdateMode(skepu::UpdateMode::RedBlack);
	
	update(domain, domain);
	
	skepu::external(skepu::read(domain), [&]
	{
		for (size_t i = 0; i < size; ++i)
			CHECK(domain(i) == Approx(expected[i]));
	});
}
