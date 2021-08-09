#include <catch2/catch.hpp>

#define SKEPU_DEBUG 3
#include <skepu>
#include <skepu-lib/io.hpp>


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

	skepu::Tensor3<int> ten3(size, size, size, 1);
	skepu::Tensor3<float> ret_ten3(size, size, size);
	skepu::Tensor3<float> stencil3(2*1+1, 2*1+1, 2*1+1, 1);

	skepu::external([&] {
		auto i(0);
		for(auto & e : ten3)
			e = i++;
		std::cout << "ten3: " << ten3 << "\n";
	}, skepu::write(ten3));
	
	conv3.setEdgeMode(skepu::Edge::None);
	conv3(ret_ten3, ten3, stencil3);
	skepu::io::cout << "Tensor3D None: " << ret_ten3 << "\n";

	conv3.setEdgeMode(skepu::Edge::Cyclic);
	conv3(ret_ten3, ten3, stencil3);
	skepu::io::cout << "Tensor3D Cyclic: " << ret_ten3 << "\n";

	conv3.setEdgeMode(skepu::Edge::Duplicate);
	conv3(ret_ten3, ten3, stencil3);
	skepu::io::cout << "Tensor3D Duplicate: " << ret_ten3 << "\n";

	conv3.setEdgeMode(skepu::Edge::Pad);
	conv3(ret_ten3, ten3, stencil3);
	skepu::io::cout << "Tensor3D Pad 0: " << ret_ten3 << "\n";
	
}

