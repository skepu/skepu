#include "../../external/catch2/catch.hpp"

#define SKEPU_DEBUG 3
#include <algorithm>
#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>

float over_3d(skepu::Region3D<int> r, skepu::Ten3<float> stencil)
{
	return r(1,0,0);
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				res += r(i, j, k) * stencil(i + r.oi, j + r.oj, k + r.ok);
	return res;
}

auto conv3 = skepu::MapOverlap(over_3d);


TEST_CASE("MapOverlap 3D fundamentals")
{
	const size_t size{4};

	conv3.setOverlap(1, 1, 1);
	conv3.setPad(-5);

	skepu::Tensor3<int> ten3(size, size, size, "", 1);
	skepu::Tensor3<float> ret_ten3(size, size, size);
	skepu::Tensor3<float> ret_ten3_none(size-2, size-2, size-2);
	skepu::Tensor3<float> stencil3(2*1+1, 2*1+1, 2*1+1, "", 1);

	skepu::external([&] {
		auto i(0);
		for(auto & e : ten3)
			e = i++;
	}, skepu::write(ten3));
	
	// The user function gives the element at +1 in the first dimension. The value of ten3 at (a, b, c) is (a * size + b) * size + c.
	auto value = [&](long a, long b, long c) { return float((a * long(size) + b) * long(size) + c); };
	auto clamp = [&](long v) { return std::min<long>(std::max<long>(v, 0), size - 1); };
	
	conv3.setEdgeMode(skepu::Edge::None);
	conv3(ret_ten3_none, ten3, stencil3);
	
	skepu::external(skepu::read(ret_ten3_none), [&]
	{
		// None: the output has the elements that have a full neighbourhood, so its element (a, b, c) is at (a + 1, b + 1, c + 1) of ten3
		for (long a = 0; a < long(size) - 2; ++a)
			for (long b = 0; b < long(size) - 2; ++b)
				for (long c = 0; c < long(size) - 2; ++c)
					CHECK(ret_ten3_none(a, b, c) == value(a + 2, b + 1, c + 1));
	});

	auto checkAll = [&](auto expected)
	{
		skepu::external(skepu::read(ret_ten3), [&]
		{
			for (long a = 0; a < long(size); ++a)
				for (long b = 0; b < long(size); ++b)
					for (long c = 0; c < long(size); ++c)
						CHECK(ret_ten3(a, b, c) == expected(a, b, c));
		});
	};

	conv3.setEdgeMode(skepu::Edge::Cyclic);
	conv3(ret_ten3, ten3, stencil3);
	checkAll([&](long a, long b, long c) { return value((a + 1) % size, b, c); });

	conv3.setEdgeMode(skepu::Edge::Duplicate);
	conv3(ret_ten3, ten3, stencil3);
	checkAll([&](long a, long b, long c) { return value(clamp(a + 1), b, c); });

	conv3.setEdgeMode(skepu::Edge::Pad);
	conv3(ret_ten3, ten3, stencil3);
	checkAll([&](long a, long b, long c) { return a + 1 >= long(size) ? -5.f : value(a + 1, b, c); });
}
