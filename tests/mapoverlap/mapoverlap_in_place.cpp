#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/io.hpp>

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
	skepu::io::cout << domain << "\n";
	
	update.setOverlap(1);
	update.setEdgeMode(skepu::Edge::Duplicate);
	update.setUpdateMode(skepu::UpdateMode::RedBlack);
	
	update(domain, domain);
	skepu::io::cout << domain << "\n";
}

