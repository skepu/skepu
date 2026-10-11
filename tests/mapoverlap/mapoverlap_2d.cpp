#include "../../external/catch2/catch.hpp"

#include <algorithm>
#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>

float over_2d(skepu::Region2D<int> r, const skepu::Mat<float> stencil)
{
	return r(1,1);
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return res;
}

auto conv2 = skepu::MapOverlap(over_2d);


TEST_CASE("MapOverlap 2D fundamentals")
{
	const size_t size{25};
	
	conv2.setOverlap(2, 2);
	conv2.setPad(-5);

	skepu::Matrix<int> m(size, size);
	skepu::Matrix<float> rm_a(size-4, size-4), rm_b(size, size), rm_c(size, size), rm_d(size, size);
	skepu::external(
		skepu::read(m),
		[&]
		{
			int i = 0;
			for(size_t y = 0; y < size; ++y)
				for(size_t x = 0; x < size; ++x)
					m(y, x) = i++;
		},
		skepu::write(m)
	);

	skepu::Matrix<float> filter(2*1+1, 2*1+1, 1);
	
	// The user function gives the element at (+1, +1) from the center (the stencil in it is not reached).
	// The value of m at (y, x) is y * size + x.
	auto value = [&](long y, long x) { return float(y * long(size) + x); };
	auto clamp = [&](long v) { return std::min<long>(std::max<long>(v, 0), size - 1); };
	
	conv2.setEdgeMode(skepu::Edge::None);
	conv2(rm_a, m, filter);
	
	conv2.setEdgeMode(skepu::Edge::Cyclic);
	conv2(rm_b, m, filter);
	
	conv2.setEdgeMode(skepu::Edge::Duplicate);
	conv2(rm_c, m, filter);
	
	conv2.setEdgeMode(skepu::Edge::Pad);
	conv2(rm_d, m, filter);
	
	skepu::external(
		skepu::read(rm_a, rm_b, rm_c, rm_d),
		[&]
		{
			// None: the output has the elements that have a full neighbourhood, so its element (y, x) is at (y + 2, x + 2) of m
			for (long y = 0; y < long(size) - 4; ++y)
				for (long x = 0; x < long(size) - 4; ++x)
					CHECK(rm_a(y, x) == value(y + 3, x + 3));
			
			for (long y = 0; y < long(size); ++y)
				for (long x = 0; x < long(size); ++x)
				{
					CHECK(rm_b(y, x) == value((y + 1) % size, (x + 1) % size));
					CHECK(rm_c(y, x) == value(clamp(y + 1), clamp(x + 1)));
					CHECK(rm_d(y, x) == ((y + 1 >= long(size) || x + 1 >= long(size)) ? -5.f : value(y + 1, x + 1)));
				}
		});
}
