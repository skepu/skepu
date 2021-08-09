#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/io.hpp>


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
	skepu::Matrix<float> rm_a(size, size), rm_b(size, size), rm_c(size, size), rm_d(size, size);
	skepu::external(
		skepu::read(m),
		[&]
		{
			int i = 0;
			for(size_t y = 0; y < size; ++y)
				for(size_t x = 0; x < size; ++x)
					m(y, x) = i++;
			std::cout << "m: " << m <<"\n";
		},
		skepu::write(m)
	);

	skepu::Matrix<float> filter(2*1+1, 2*1+1, 1);
	
	conv2.setEdgeMode(skepu::Edge::None);
	conv2(rm_a, m, filter);
	skepu::io::cout << "Matrix 2D None:    rm_a= " << rm_a << "\n";

	conv2.setEdgeMode(skepu::Edge::Cyclic);
	conv2(rm_b, m, filter);
	skepu::io::cout << "Matrix 2D Cyclic:    rm_b = " << rm_b << "\n";

	conv2.setEdgeMode(skepu::Edge::Duplicate);
	conv2(rm_c, m, filter);
	skepu::io::cout << "Matrix 2D Duplicate:    rm_c = " << rm_c << "\n";

	conv2.setEdgeMode(skepu::Edge::Pad);
	conv2(rm_d, m, filter);
	skepu::io::cout << "Matrix 2D Pad:    rm_d = " << rm_d << "\n";
	
}

