#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <math.h>
#include <test_helper_automatic_setup.hpp>

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


int redfn(int lhs, int rhs)
{
	return lhs + rhs;
}


auto mapred1 = skepu::MapReduce(test1, redfn);
auto mapred2 = skepu::MapReduce(test2, redfn);
auto mapred3 = skepu::MapReduce(test3, redfn);
auto mapred4 = skepu::MapReduce(test4, redfn);

auto mapred0_1 = skepu::MapReduce<0>(test1_ver2, redfn);
auto mapred0_2 = skepu::MapReduce<0>(test2_ver2, redfn);
auto mapred0_3 = skepu::MapReduce<0>(test3_ver2, redfn);
auto mapred0_4 = skepu::MapReduce<0>(test4_ver2, redfn);


TEST_CASE("MapReduce fundamentals")
{
	const size_t size{20};
	
	skepu::Vector<int> vec(size);
	skepu::Matrix<int> mat(size, size+1);
	skepu::Tensor3<int> ten3(size, size+1, size+2);
	skepu::Tensor4<int> ten4(size, size+1, size+2, size+3);
	
	for (size_t i = 0; i < vec.size(); ++i)
		vec(i) = i;
	
	for (size_t i = 0; i < mat.total_rows(); ++i)
		for (size_t j = 0; j < mat.total_cols(); ++j)
			mat(i, j) = i * mat.total_cols() + j;
	
	for (size_t i = 0; i < ten3.size_i(); ++i)
		for (size_t j = 0; j < ten3.size_j(); ++j)
			for (size_t k = 0; k < ten3.size_k(); ++k)
				ten3(i, j, k) = i * size*size + j * size + k;
	
	for (size_t i = 0; i < ten4.size_i(); ++i)
		for (size_t j = 0; j < ten4.size_j(); ++j)
			for (size_t k = 0; k < ten4.size_k(); ++k)
				for (size_t l = 0; l < ten4.size_l(); ++l)
					ten4(i, j, k, l) = i * size*size*size + j * size*size + k * size + l;
	
	
	// Test MapReduce: the sums are made on the host (modulo 2^32: the 4D sum does not fit an int)
	
	unsigned long long expected1{0}, expected2{0}, expected3{0}, expected4{0};
	
	for (size_t i = 0; i < vec.size(); ++i)
		expected1 += 2ull * vec(i) + i;
	
	for (size_t i = 0; i < mat.total_rows(); ++i)
		for (size_t j = 0; j < mat.total_cols(); ++j)
			expected2 += 2ull * mat(i, j) + i + j;
	
	for (size_t i = 0; i < ten3.size_i(); ++i)
		for (size_t j = 0; j < ten3.size_j(); ++j)
			for (size_t k = 0; k < ten3.size_k(); ++k)
				expected3 += 2ull * ten3(i, j, k) + k;
	
	for (size_t i = 0; i < ten4.size_i(); ++i)
		for (size_t j = 0; j < ten4.size_j(); ++j)
			for (size_t k = 0; k < ten4.size_k(); ++k)
				for (size_t l = 0; l < ten4.size_l(); ++l)
					expected4 += 2ull * ten4(i, j, k, l) + l;
	
	int res1 = mapred1(vec);
	int res2 = mapred2(mat);
	int res3 = mapred3(ten3);
	int res4 = mapred4(ten4);
	
	CHECK(res1 == (int)(uint32_t)expected1);
	CHECK(res2 == (int)(uint32_t)expected2);
	CHECK(res3 == (int)(uint32_t)expected3);
	CHECK(res4 == (int)(uint32_t)expected4);
	
	// MapReduce over an index space: the elements are made from the indices and a uniform argument
	
	mapred0_1.setDefaultSize(2);
	mapred0_2.setDefaultSize(2, 4);
	mapred0_3.setDefaultSize(2, 4, 6);
	mapred0_4.setDefaultSize(2, 4, 6, 8);
	
	res1 = mapred0_1(1);
	res2 = mapred0_2(1);
	res3 = mapred0_3(1);
	res4 = mapred0_4(1);
	
	int index1{0}, index2{0}, index3{0}, index4{0};
	for (int i = 0; i < 2; ++i)
		index1 += 2 + i;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 4; ++j)
			index2 += 2 + i + j;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 4; ++j)
			for (int k = 0; k < 6; ++k)
				index3 += 2 + k;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 4; ++j)
			for (int k = 0; k < 6; ++k)
				for (int l = 0; l < 8; ++l)
					index4 += 2 + l;
	
	CHECK(res1 == index1);
	CHECK(res2 == index2);
	CHECK(res3 == index3);
	CHECK(res4 == index4);
}
