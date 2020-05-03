#include <iostream>
#include <skepu>
#include <math.h>

int test0(int a)
{
	return a + 1;
}

int test1(skepu::Index1D in, int a)
{
	return a * 2 + in.i;
}

int test2(skepu::Index2D in, int a)
{
//	printf("(%d %d)\n", in.row, in.col);
	return a * 2 + in.row + in.col;
}

int test3(skepu::Index3D in, int a)
{
//	printf("(%d %d %d)\n", in.i, in.j, in.k);
	return a * 2 + in.k;
}

int test4(skepu::Index4D in, int a)
{
//	printf("(%d %d %d %d)\n", in.i, in.j, in.k, in.l);
	return a * 2 + in.l;
}

int test1_ver2(skepu::Index1D in, int a)
{
	return a * 2 + in.i;
}

int test2_ver2(skepu::Index2D in, int a)
{
//	printf("(%d %d)\n", in.row, in.col);
	return a * 2 + in.row + in.col;
}

int test3_ver2(skepu::Index3D in, int a)
{
//	printf("(%d %d %d)\n", in.i, in.j, in.k);
	return a * 2 + in.k;
}

int test4_ver2(skepu::Index4D in, int a)
{
//	printf("(%d %d %d %d)\n", in.i, in.j, in.k, in.l);
	return a * 2 + in.l;
}


// TEST PROXIES

int test1_proxy(skepu::Vec<int> vec)
{
	return vec.size + 1000 * vec.data[1];
}

int test2_proxy(skepu::Mat<int> mat)
{
	return mat.cols + mat.rows + 1000 * mat.data[1];
}

int test3_proxy(skepu::Ten3<int> ten3)
{
	return ten3.size_i + ten3.size_j + ten3.size_k + 1000 * ten3.data[1];
}

int test4_proxy(skepu::Ten4<int> ten4)
{
	return ten4.size_i + ten4.size_j + ten4.size_k + ten4.size_l + 1000 * ten4.data[1];
}

int redfn(int lhs, int rhs)
{
	return lhs + rhs;
}

int test_combo()
{
	return 0;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	auto skel0 = skepu::Map<1>(test0);
	auto skel1 = skepu::Map<1>(test1);
	auto skel2 = skepu::Map<1>(test2);
	auto skel3 = skepu::Map<1>(test3);
	auto skel4 = skepu::Map<1>(test4);
	
	
	skepu::Vector<int> vec(size);
	
	skepu::Matrix<int> mat(size, size);
	
	skepu::Tensor3<int> ten3(size, size, size);
	
	skepu::Tensor4<int> ten4(size, size, size, size);
	
	
	std::cout << vec.size() << ", " << vec.size() << std::endl;
	std::cout << mat.size() << ", " << mat.total_rows() << ", " << mat.total_cols() << std::endl;
	std::cout << ten3.size() << ", " << ten3.size_i() << ", " << ten3.size_j() << ", " << ten3.size_k() << std::endl;
	std::cout << ten4.size() << ", " << ten4.size_i() << ", " << ten4.size_j() << ", " << ten4.size_k() << ", " << ten4.size_k() << std::endl;
	
	for (size_t i = 0; i < size; ++i)
		vec(i) = i;
	
	for (size_t i = 0; i < size; ++i)
		for (size_t j = 0; j < size; ++j)
			mat(i, j) = i * size + j;
	
	for (size_t i = 0; i < size; ++i)
		for (size_t j = 0; j < size; ++j)
			for (size_t k = 0; k < size; ++k)
				ten3(i, j, k) = i * size*size + j * size + k;
	
	for (size_t i = 0; i < size; ++i)
		for (size_t j = 0; j < size; ++j)
			for (size_t k = 0; k < size; ++k)
				for (size_t l = 0; l < size; ++l)
					ten4(i, j, k, l) = i * size*size*size + j * size*size + k * size + l;
	
	
	auto it1 = vec.begin();
	auto in1 = it1.getIndex();
	
	auto it2 = mat.begin() + 9;
	auto in2 = it2.getIndex();
	
	auto it3 = ten3.begin() + 99;
	auto in3 = it3.getIndex();
	
	auto it4 = ten4.begin() + 999;
	auto in4 = it4.getIndex();
	
	std::cout << in1.i << std::endl;
	std::cout << in2.row << ", " << in2.col << std::endl;
	std::cout << in3.i << ", " << in3.j << ", " << in3.k << std::endl;
	std::cout << in4.i << ", " << in4.j << ", " << in4.k << ", " << in4.l << std::endl;
	
	
	skel0(vec, vec);
	skel0(mat, mat);
	skel0(ten3, ten3);
	skel0(ten4, ten4);
	
	skel1(vec, vec);
	skel2(mat, mat);
	skel3(ten3, ten3);
	skel4(ten4, ten4);
	
//	std::cout << vec << std::endl;
//	std::cout << mat << std::endl;
//	std::cout << ten3 << std::endl;
//	std::cout << ten4 << std::endl;
	
	
	// TEST PROXIES
	
	auto skel1_p = skepu::Map<0>(test1_proxy);
	auto skel2_p = skepu::Map<0>(test2_proxy);
	auto skel3_p = skepu::Map<0>(test3_proxy);
	auto skel4_p = skepu::Map<0>(test4_proxy);
	
	skepu::Vector<int> dummy(1);
	
	std::cout << skepu::is_skepu_container_proxy<decltype(ten3.hostProxy())>::value << std::endl;
	
	skel1_p(dummy, vec);
	std::cout << dummy << std::endl;
	
	skel2_p(dummy, mat);
	std::cout << dummy << std::endl;
	
	skel3_p(dummy, ten3);
	std::cout << dummy << std::endl;
	
	skel4_p(dummy, ten4);
	std::cout << dummy << std::endl;
	
	
	// Test MapReduce
	
	auto mapred1 = skepu::MapReduce<1>(test1, redfn);
	auto mapred2 = skepu::MapReduce<1>(test2, redfn);
	auto mapred3 = skepu::MapReduce<1>(test3, redfn);
	auto mapred4 = skepu::MapReduce<1>(test4, redfn);
	
	int res1 = mapred1(vec);
	int res2 = mapred2(mat);
	int res3 = mapred3(ten3);
	int res4 = mapred4(ten4);
	
	std::cout << res1 << ", " << res2 << ", " << res3 << ", " << res4 << "\n";
	
	auto mapred0_1 = skepu::MapReduce<0>(test1_ver2, redfn);
	auto mapred0_2 = skepu::MapReduce<0>(test2_ver2, redfn);
	auto mapred0_3 = skepu::MapReduce<0>(test3_ver2, redfn);
	auto mapred0_4 = skepu::MapReduce<0>(test4_ver2, redfn);
	
	mapred0_1.setDefaultSize(2);
	mapred0_2.setDefaultSize(2, 4);
	mapred0_3.setDefaultSize(2, 4, 6);
	mapred0_4.setDefaultSize(2, 4, 6, 8);
	
	res1 = mapred0_1(1);
	res2 = mapred0_2(1);
	res3 = mapred0_3(1);
	res4 = mapred0_4(1);
	
	std::cout << res1 << ", " << res2 << ", " << res3 << ", " << res4 << "\n";
	
	
	size_t L = size;
	auto seed = skepu::Scan([](int x, int y){ return x+y; });
	skepu::Tensor4<int> temp(L, L, L, L, 1); // all entries 1
	skepu::Tensor4<int> prng_lcg(L, L, L, L);
	seed(prng_lcg, temp);
	
	std::cout << temp << ", " << prng_lcg << "\n";
	
	return 0;
}
