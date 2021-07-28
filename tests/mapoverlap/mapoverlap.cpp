#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/io.hpp>

float over_1d(skepu::Region1D<int> r, int scale)
{
	return (r(-2)*4 + r(-1)*2 + r(0) + r(1)*2 + r(2)*4) / scale;
}

float over_2d(skepu::Region2D<int> r, const skepu::Mat<float> stencil)
{
	return r(1,0);
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return res;
}

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


auto conv = skepu::MapOverlap(over_1d);
auto conv2 = skepu::MapOverlap(over_2d);
auto conv3 = skepu::MapOverlap(over_3d);
auto conv4 = skepu::MapOverlap(over_4d);


TEST_CASE("MapOverlap fundamentals")
{
	const size_t size{20};

	conv.setOverlap(1);

	skepu::Vector<int> v(size);
	skepu::Vector<float> rv(size);
	skepu::external(
		skepu::read(v),
		[&]
		{
			for (size_t i = 0; i < size; ++i)
				v(i) = i;

			std::cout << "v: " << v <<"\n";
		},
		skepu::write(v)
	);
	
	conv.setEdgeMode(skepu::Edge::None);
	conv(rv, v, 13);
	skepu::io::cout << "Vector None:    rv = " << rv << "\n";
	
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rv, v, 13);
	skepu::io::cout << "Vector Cyclic:    rv = " << rv << "\n";

	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rv, v, 13);
	skepu::io::cout << "Vector Duplicate  rv = " << rv << "\n";

	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rv, v, 13);
	skepu::io::cout << "Vector Pad 0:     rv = " << rv << "\n";

	skepu::Matrix<int> m(size, size);
	skepu::Matrix<float> rm(size, size);
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
	
	conv.setOverlap(1);
	conv.setOverlapMode(skepu::Overlap::RowWise);
	
	conv.setEdgeMode(skepu::Edge::None);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Row-wise None:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Row-wise Cyclic:    rm = " << rm << "\n";

	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Row-wise Duplicate  rm = " << rm << "\n";

	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Row-wise Pad 0:     rm = " << rm << "\n";


	conv.setOverlap(1);
	conv.setOverlapMode(skepu::Overlap::ColWise);
	
	conv.setEdgeMode(skepu::Edge::None);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Col-wise None:    rm = " << rm << "\n";
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Col-wise Cyclic:    rm = " << rm << "\n";

	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Col-wise Duplicate  rm = " << rm << "\n";

	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	skepu::io::cout << "Matrix Col-wise Pad 0:     rm = " << rm << "\n";
	
	
	conv2.setOverlap(1, 1);
	conv2.setPad(-5);

	skepu::Matrix<float> filter(2*1+1, 2*1+1, 1);
	skepu::Matrix<float> rm2(size, size);
	
	conv2.setEdgeMode(skepu::Edge::None);
	conv2(rm2, m, filter);
	skepu::io::cout << "Matrix 2D None:    rm = " << rm2 << "\n";

	conv2.setEdgeMode(skepu::Edge::Cyclic);
	conv2(rm2, m, filter);
	skepu::io::cout << "Matrix 2D Cyclic:    rm = " << rm2 << "\n";

	conv2.setEdgeMode(skepu::Edge::Duplicate);
	conv2(rm2, m, filter);
	skepu::io::cout << "Matrix 2D Duplicate:    rm = " << rm2 << "\n";

	conv2.setEdgeMode(skepu::Edge::Pad);
	conv2(rm2, m, filter);
	skepu::io::cout << "Matrix 2D Pad 0:    rm = " << rm2 << "\n";
	
	
	// Tensor3
	conv3.setOverlap(1, 1, 1);
	conv3.setPad(-5);

	skepu::Tensor3<int> ten3(size, size, size, 1);
	skepu::Tensor3<float> ret_ten3(size, size, size);
	skepu::Tensor3<float> stencil3(2*1+1, 2*1+1, 2*1+1, 1);

	skepu::external(
		[&]
		{
			auto i(0);
			for(auto & e : ten3)
				e = i++;
			std::cout << "ten3: " << ten3 << "\n";
		},
		skepu::write(ten3));
	
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
		
	// Tensor4
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

