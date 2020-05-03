#include <iostream>

#include <skepu>

float over_1d(skepu::Region1D<float> r, int scale)
{
	return (r(-2)*4 + r(-1)*2 + r(0) + r(1)*2 + r(2)*4) / scale;
}

float over_2d(skepu::Region2D<float> r, const skepu::Mat<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return res;
}

float over_3d(skepu::Region3D<float> r, skepu::Ten3<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				res += r(i, j, k) * stencil(i + r.oi, j + r.oj, k + r.ok);
	return res;
}

float over_4d(skepu::Region4D<float> r, skepu::Ten4<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				for (int l = -r.ol; l <= r.ol; ++l)
					res += r(i, j, k, l) * stencil(i + r.oi, j + r.oj, k + r.ok, l + r.ol);
	return res;
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
	
	auto conv = skepu::MapOverlap(over_1d);
	conv.setBackend(spec);
	conv.setOverlap(2);
	
	skepu::Vector<float> v(size, 10), rv(size);
	for (size_t i = 0;  i < size; ++i)
		v(i) = i;
	
	std::cout << "v: " << v <<"\n";
	
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rv, v, 13);
	std::cout << "Vector Cyclic:    rv = " << rv << "\n";
	
	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rv, v, 13);
	std::cout << "Vector Duplicate  rv = " << rv << "\n";
	
	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rv, v, 13);
	std::cout << "Vector Pad 0:     rv = " << rv << "\n";
	
	
	
	skepu::Matrix<float> m(size, size, 10), rm(size, size);
	
	
	int i = 0;
	for (int y = 0; y < size; ++y)
		for (int x = 0; x < size; ++x)
			m(y, x) = i++;
	
	std::cout << "m: " << m <<"\n";
	conv.setOverlap(2);
	conv.setOverlapMode(skepu::Overlap::RowWise);
	
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Row-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Row-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Row-wise Pad 0:     rm = " << rm << "\n";
	
	
	conv.setOverlap(2);
	conv.setOverlapMode(skepu::Overlap::ColWise);
	
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Col-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Col-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Col-wise Pad 0:     rm = " << rm << "\n";
	
	
	conv.setOverlap(2);
	conv.setOverlapMode(skepu::Overlap::RowColWise);
	
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Row-col-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Row-col-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Row-col-wise Pad 0:     rm = " << rm << "\n";
	
	
	conv.setOverlap(2);
	conv.setOverlapMode(skepu::Overlap::ColRowWise);
	
	conv.setEdgeMode(skepu::Edge::Cyclic);
	conv(rm, m, 13);
	std::cout << "Matrix Col-row-wise Cyclic:    rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Duplicate);
	conv(rm, m, 13);
	std::cout << "Matrix Col-row-wise Duplicate  rm = " << rm << "\n";
	
	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(rm, m, 13);
	std::cout << "Matrix Col-row-wise Pad 0:     rm = " << rm << "\n";
	
	
	auto conv2 = skepu::MapOverlap(over_2d);
	conv2.setBackend(spec);
	conv2.setOverlap(1, 1);
	
	skepu::Matrix<float> filter(2*1+1, 2*1+1, 1), rm2(size - 2*1, size - 2*1);
	
	conv2(rm2, m, filter);
	std::cout << "Matrix 2D Pad 0:    rm = " << rm2 << "\n";
	
	
	// Tensor3
	
/*	auto conv3 = skepu::MapOverlap(over_3d);
	conv3.setBackend(spec);
	conv3.setOverlap(1, 1, 1);
	
	skepu::Tensor3<float> ten3(size, size, size, 1), stencil3(2*1+1, 2*1+1, 2*1+1, 1), ret_ten3(size - 2*1, size - 2*1, size - 2*1);
	
	conv3(ret_ten3, ten3, stencil3);
	std::cout << "Tensor3D: " << ret_ten3 << "\n";
	*/
	
	// Tensor4
	/*
	auto conv4 = skepu::MapOverlap(over_4d);
	conv4.setBackend(spec);
	conv4.setOverlap(1, 1, 1, 1);
	
	skepu::Tensor4<float> ten4(size, size, size, size, 1), stencil4(2*1+1, 2*1+1, 2*1+1, 2*1+1, 1), ret_ten4(size - 2*1, size - 2*1, size - 2*1, size - 2*1);
	
	conv4(ret_ten4, ten4, stencil4);
//	std::cout << "Tensor4D: " << ret_ten4 << "\n";
	*/
	return 0;
}

