#include "../../external/catch2/catch.hpp"

#include <algorithm>
#include <skepu>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>

float over_4d(skepu::Region4D<int> r, skepu::Ten4<float> stencil)
{
	return r(1,0,0,0);
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				for (int l = -r.ol; l <= r.ol; ++l)
					res +=
						r(i, j, k, l) * stencil(i + r.oi, j + r.oj, k + r.ok, l + r.ol);
	return res;
}

auto conv4 = skepu::MapOverlap(over_4d);


TEST_CASE("MapOverlap 4D fundamentals")
{
	const size_t size{20};
	
	conv4.setOverlap(1, 1, 1, 1);
	conv4.setPad(-5);

	skepu::Tensor4<int> ten4(size, size, size, size);
	skepu::Tensor4<float> ret_ten4(size, size, size, size);
	skepu::Tensor4<float> stencil4(2*1+1, 2*1+1, 2*1+1, 2*1+1, "", 1);

	skepu::external(
		[&]
		{
			auto i(0);
			for(auto & e : ten4)
				e = i++;
		},
		skepu::write(ten4));

	// The user function gives the element at +1 in the first dimension. The value of ten4 at (a, b, c, d) is ((a * size + b) * size + c) * size + d.
	auto value = [&](long a, long b, long c, long d) { return float(((a * long(size) + b) * long(size) + c) * long(size) + d); };
	auto clamp = [&](long v) { return std::min<long>(std::max<long>(v, 0), size - 1); };
	
	auto checkAll = [&](auto expected)
	{
		skepu::external(skepu::read(ret_ten4), [&]
		{
			for (long a = 0; a < long(size); ++a)
				for (long b = 0; b < long(size); ++b)
					for (long c = 0; c < long(size); ++c)
						for (long d = 0; d < long(size); ++d)
							CHECK(ret_ten4(a, b, c, d) == expected(a, b, c, d));
		});
	};

	conv4.setEdgeMode(skepu::Edge::Cyclic);
	conv4(ret_ten4, ten4, stencil4);
	checkAll([&](long a, long b, long c, long d) { return value((a + 1) % size, b, c, d); });

	conv4.setEdgeMode(skepu::Edge::Duplicate);
	conv4(ret_ten4, ten4, stencil4);
	checkAll([&](long a, long b, long c, long d) { return value(clamp(a + 1), b, c, d); });

	conv4.setEdgeMode(skepu::Edge::Pad);
	conv4(ret_ten4, ten4, stencil4);
	checkAll([&](long a, long b, long c, long d) { return a + 1 >= long(size) ? -5.f : value(a + 1, b, c, d); });
}
