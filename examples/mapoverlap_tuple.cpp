#include <iostream>

#include <skepu>


skepu::multiple<float, int>
over_1d_multi(skepu::Index1D idx, skepu::Region1D<float> r, int scale)
{
	return skepu::ret((r(-2)*4 + r(-1)*2 + r(0) + r(1)*2 + r(2)*4) / scale, idx.i);
}

skepu::multiple<float, int>
over_2d_multi(skepu::Index2D idx, skepu::Region2D<float> r, const skepu::Mat<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return skepu::ret(res, idx.row + idx.col);;
}

skepu::multiple<float, int>
over_3d_multi(skepu::Index3D idx, skepu::Region3D<float> r, skepu::Ten3<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				res += r(i, j, k) * stencil(i + r.oi, j + r.oj, k + r.ok);
	return skepu::ret(res, idx.i + idx.j + idx.k);
}

skepu::multiple<float, int>
over_4d_multi(skepu::Index4D idx, skepu::Region4D<float> r, skepu::Ten4<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				for (int l = -r.ol; l <= r.ol; ++l)
					res += r(i, j, k, l) * stencil(i + r.oi, j + r.oj, k + r.ok, l + r.ol);
	return skepu::ret(res, idx.i + idx.j + idx.k + idx.l);
}




int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	skepu::setGlobalBackendSpec(spec);
	
	{
		auto conv = skepu::MapOverlap(over_1d_multi);
		conv.setOverlap(2);
		
		skepu::Vector<float> v(size, 10), rv1(size);
		skepu::Vector<int> rv2(size);
		
		for (size_t i = 0;  i < size; ++i)
			v(i) = i;
		
		std::cout << "v: " << v <<"\n";
		
		conv.setEdgeMode(skepu::Edge::Cyclic);
		conv(rv1, rv2, v, 13);
		std::cout << "Vector Cyclic:    rv1 = " << rv1 << "\n";
		std::cout << "Vector Cyclic:    rv2 = " << rv2 << "\n";
		
		conv.setEdgeMode(skepu::Edge::Duplicate);
		conv(rv1, rv2, v, 13);
		std::cout << "Vector Duplicate  rv1 = " << rv1 << "\n";
		std::cout << "Vector Duplicate  rv2 = " << rv2 << "\n";
		
		conv.setEdgeMode(skepu::Edge::Pad);
		conv.setPad(0);
		conv(rv1, rv2, v, 13);
		std::cout << "Vector Pad 0:     rv1 = " << rv1 << "\n";
		std::cout << "Vector Pad 0:     rv2 = " << rv2 << "\n";
		
		
		skepu::Matrix<float> m(size, size, 10), rm1(size, size);
		skepu::Matrix<int> rm2(size, size);
		
		
		int i = 0;
		for (int y = 0; y < size; ++y)
			for (int x = 0; x < size; ++x)
				m(y, x) = i++;
		
		std::cout << "m: " << m <<"\n";
		conv.setOverlap(2);
		conv.setOverlapMode(skepu::Overlap::RowWise);
		
		conv.setEdgeMode(skepu::Edge::Cyclic);
		conv(rm1, rm2, m, 13);
		std::cout << "Matrix Row-wise Cyclic:    rm1 = " << rm1 << "\n";
		std::cout << "Matrix Row-wise Cyclic:    rm2 = " << rm2 << "\n";
		
		conv.setEdgeMode(skepu::Edge::Duplicate);
		conv(rm1, rm2, m, 13);
		std::cout << "Matrix Row-wise Duplicate  rm1 = " << rm1 << "\n";
		std::cout << "Matrix Row-wise Duplicate  rm2 = " << rm2 << "\n";
		
		conv.setEdgeMode(skepu::Edge::Pad);
		conv.setPad(0);
		conv(rm1, rm2, m, 13);
		std::cout << "Matrix Row-wise Pad 0:     rm1 = " << rm1 << "\n";
		std::cout << "Matrix Row-wise Pad 0:     rm2 = " << rm2 << "\n";
		
		
		conv.setOverlap(2);
		conv.setOverlapMode(skepu::Overlap::ColWise);
		
		conv.setEdgeMode(skepu::Edge::Cyclic);
		conv(rm1, rm2, m, 13);
		std::cout << "Matrix Col-wise Cyclic:    rm1 = " << rm1 << "\n";
		std::cout << "Matrix Col-wise Cyclic:    rm2 = " << rm2 << "\n";
		
		conv.setEdgeMode(skepu::Edge::Duplicate);
		conv(rm1, rm2, m, 13);
		std::cout << "Matrix Col-wise Duplicate  rm1 = " << rm1 << "\n";
		std::cout << "Matrix Col-wise Duplicate  rm2 = " << rm2 << "\n";
		
		conv.setEdgeMode(skepu::Edge::Pad);
		conv.setPad(0);
		conv(rm1, rm2, m, 13);
		std::cout << "Matrix Col-wise Pad 0:     rm1 = " << rm1 << "\n";
		std::cout << "Matrix Col-wise Pad 0:     rm2 = " << rm2 << "\n";
	}
	
	// Matrix
	{
		skepu::Matrix<float> m(size, size, 10), filter(2*1+1, 2*1+1, 1), rm2(size - 2*1, size - 2*1);
		skepu::Matrix<int> ret_m_int(size - 2*1, size - 2*1);
		
		auto conv2_m = skepu::MapOverlap(over_2d_multi);
		conv2_m.setOverlap(1, 1);
		conv2_m(rm2, ret_m_int, m, filter);
		std::cout << "Tensor2D: " << rm2 << "\n" << ret_m_int << "\n";
	}
	/*
	// Tensor3
	
	skepu::Tensor3<float> ten3(size, size, size, 1), stencil3(2*1+1, 2*1+1, 2*1+1, 1), ret_ten3(size - 2*1, size - 2*1, size - 2*1);
	skepu::Tensor3<int> ret_ten3_int(size - 2*1, size - 2*1, size - 2*1);
	
	auto conv3_m = skepu::MapOverlap(over_3d_multi);
	conv3_m.setOverlap(1, 1, 1);
	conv3_m(ret_ten3, ret_ten3_int, ten3, stencil3);
	std::cout << "Tensor3D: " << ret_ten3 << "\n" << ret_ten3_int << "\n";
	
	
	// Tensor4
	
	skepu::Tensor4<float> ten4(size, size, size, size, 1), stencil4(2*1+1, 2*1+1, 2*1+1, 2*1+1, 1), ret_ten4(size - 2*1, size - 2*1, size - 2*1, size - 2*1);
	skepu::Tensor4<int> ret_ten4_int(size - 2*1, size - 2*1, size - 2*1, size - 2*1);
	
	auto conv4_m = skepu::MapOverlap(over_4d_multi);
	conv4_m.setOverlap(1, 1, 1, 1);
	std::cout << "Tensor4D: " << ret_ten4 << "\n" << ret_ten4_int << "\n";
	
	
	auto conv4_m_l = skepu::MapOverlap([](skepu::Index4D idx, skepu::Region4D<float> r, skepu::Ten4<float> stencil) -> skepu::multiple<float, int> 
	{
		float res = 0;
		for (int i = -r.oi; i <= r.oi; ++i)
			for (int j = -r.oj; j <= r.oj; ++j)
				for (int k = -r.ok; k <= r.ok; ++k)
					for (int l = -r.ol; l <= r.ol; ++l)
						res += r(i, j, k, l) * stencil(i + r.oi, j + r.oj, k + r.ok, l + r.ol);
		return skepu::ret(res, idx.i + idx.j + idx.k + idx.l);
	});*/
	
	return 0;
}

