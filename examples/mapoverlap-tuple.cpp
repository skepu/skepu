#include <iostream>

#include <skepu>




skepu::multiple<float, int> over_2d_multi(skepu::Index2D idx, skepu::Region2D<float> r, const skepu::Mat<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return skepu::ret(res, idx.row + idx.col);;
}

skepu::multiple<float, int> over_3d_multi(skepu::Index3D idx, skepu::Region3D<float> r, skepu::Ten3<float> stencil)
{
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			for (int k = -r.ok; k <= r.ok; ++k)
				res += r(i, j, k) * stencil(i + r.oi, j + r.oj, k + r.ok);
	return skepu::ret(res, idx.i + idx.j + idx.k);;
}

skepu::multiple<float, int> over_4d_multi(skepu::Index4D idx, skepu::Region4D<float> r, skepu::Ten4<float> stencil)
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
	
	
	// Matrix
	
	skepu::Matrix<float> m(size, size, 10), filter(2*1+1, 2*1+1, 1), rm2(size - 2*1, size - 2*1);
	skepu::Matrix<int> ret_m_int(size - 2*1, size - 2*1);
	
	auto conv2_m = skepu::MapOverlap(over_2d_multi);
	conv2_m.setOverlap(1, 1);
	conv2_m(rm2, ret_m_int, m, filter);
	std::cout << "Tensor2D: " << rm2 << "\n" << ret_m_int << "\n";
	
	
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
	conv4_m(ret_ten4, ret_ten4_int, ten4, stencil4);
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
	});
	
	return 0;
}

