#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/io.hpp>


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
	skepu::Tensor4<float> stencil4(2*1+1, 2*1+1, 2*1+1, 2*1+1, 1);

	skepu::external(
		[&]
		{
			auto i(0);
			for(auto & e : ten4)
				e = i++;
			std::cout << "ten4: " << ten4 << "\n";
		},
		skepu::write(ten4));

	conv4.setEdgeMode(skepu::Edge::Cyclic);
	conv4(ret_ten4, ten4, stencil4);
	
	skepu::io::cout << "Tensor4D Cyclic: " << ret_ten4 << "\n";

	conv4.setEdgeMode(skepu::Edge::Duplicate);
	conv4(ret_ten4, ten4, stencil4);
	skepu::io::cout << "Tensor4D Duplicate: " << ret_ten4 << "\n";

	conv4.setEdgeMode(skepu::Edge::Pad);
	conv4(ret_ten4, ten4, stencil4);
	skepu::io::cout << "Tensor4D Pad 0: " << ret_ten4 << "\n";
	
}

