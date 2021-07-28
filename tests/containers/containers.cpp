#include <catch2/catch.hpp>

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



int redfn(int lhs, int rhs)
{
	return lhs + rhs;
}

int test_combo()
{
	return 0;
}


auto skel0 = skepu::Map(test0);
auto skel1 = skepu::Map(test1);
auto skel2 = skepu::Map(test2);
auto skel3 = skepu::Map(test3);
auto skel4 = skepu::Map(test4);


TEST_CASE("Container fundamentals")
{
	const size_t size{10};
	
	skepu::Vector<int> vec(size);
	skepu::Matrix<int> mat(size, size+1);
	skepu::Tensor3<int> ten3(size, size+1, size+2);
	skepu::Tensor4<int> ten4(size, size+1, size+2, size+3);
	
	CHECK(vec.size() == size);
	
	CHECK(mat.size() == size * (size + 1));
	CHECK(mat.total_rows() == size);
	CHECK(mat.total_cols() == size + 1);
	
	CHECK(ten3.size() == size * (size + 1) * (size + 2));
	CHECK(ten3.size_i() == size);
	CHECK(ten3.size_j() == size + 1);
	CHECK(ten3.size_k() == size + 2);
	
	CHECK(ten4.size() == size  * (size + 1) * (size + 2) * (size + 3));
	CHECK(ten4.size_i() == size);
	CHECK(ten4.size_j() == size + 1);
	CHECK(ten4.size_k() == size + 2);
	CHECK(ten4.size_l() == size + 3);
	
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
	
	std::cout << "==========================================================\n";
	std::cout << "Enter copy move constructor region\n";
	std::cout << "==========================================================\n";
	{
		mat(size-1, size-1) = 56;
		
		
		// Copy assignment
		skepu::Vector<int> vecB = vec;
		skepu::Matrix<int> matB = mat;
		skepu::Tensor3<int> ten3B = ten3;
		skepu::Tensor4<int> ten4B = ten4;
		
		CHECK(vecB.size() == vec.size());
		CHECK(matB.size() == mat.size());
		CHECK(ten3B.size() == ten3.size());
		CHECK(ten4B.size() == ten4.size());
		
		CHECK(matB(size-1, size-1) == 56);
		
		// Copy constructor
		skepu::Vector<int> vecC(vec);
		skepu::Matrix<int> matC(mat);
		skepu::Tensor3<int> ten3C(ten3);
		skepu::Tensor4<int> ten4C(ten4);
		
		CHECK(vecC.size() == vec.size());
		CHECK(matC.size() == mat.size());
		CHECK(ten3C.size() == ten3.size());
		CHECK(ten4C.size() == ten4.size());
		
		
		CHECK(matC(size-1, size-1) == 56);
		
		
		// Move assignment
		skepu::Vector<int> vecD = std::move(vecB);
		skepu::Matrix<int> matD = std::move(matB);
		skepu::Tensor3<int> ten3D = std::move(ten3B);
		skepu::Tensor4<int> ten4D = std::move(ten4B);
		
		CHECK(vecD.size() == vec.size());
		CHECK(matD.size() == mat.size());
		CHECK(ten3D.size() == ten3.size());
		CHECK(ten4D.size() == ten4.size());
		
		CHECK(vecB.size() == 0);
		CHECK(matB.size() == 0);
		CHECK(ten3B.size() == 0);
		CHECK(ten4B.size() == 0);
		
		CHECK(ten4D.size_i() == size);
		
		CHECK(matD(size-1, size-1) == 56);
		
		
		// Move constructor
		skepu::Vector<int> vecE(std::move(vecC));
		skepu::Matrix<int> matE(std::move(matC));
		skepu::Tensor3<int> ten3E(std::move(ten3C));
		skepu::Tensor4<int> ten4E(std::move(ten4C));

		CHECK(vecE.size() == vec.size());
		CHECK(matE.size() == mat.size());
		CHECK(ten3E.size() == ten3.size());
		CHECK(ten4E.size() == ten4.size());
		
		CHECK(vecC.size() == 0);
		CHECK(matC.size() == 0);
		CHECK(ten3C.size() == 0);
		CHECK(ten4C.size() == 0);
		
		CHECK(matE(size-1, size-1) == 56);
		
		
		// Container conversion
		skepu::Tensor3<float> source(2,3,4);
		skepu::Vector<float> destination = std::move(source);
		CHECK(source.size() == 0);
		CHECK(destination.size() == 2*3*4);
		
		
	}
	std::cout << "==========================================================\n";
	std::cout << "Exit copy move constructor region\n";
	std::cout << "==========================================================\n";
	
	
	std::cout << "==========================================================\n";
	std::cout << "Enter pointer constructor region\n";
	std::cout << "==========================================================\n";
	{
		// Constructors from pointers
		int *vecbuf = new int[size];
		int *matbuf = new int[size * size];
		int *ten3buf = new int[size * size * size];
		int *ten4buf = new int[size * size * size * size];
		
		skepu::Vector<int> vecP(vecbuf, size, skepu::no_dealloc);
		skepu::Matrix<int> matP(matbuf, size, size, skepu::no_dealloc);
		skepu::Tensor3<int> ten3P(ten3buf, size, size, size, skepu::no_dealloc);
		skepu::Tensor4<int> ten4P(ten4buf, size, size, size, size, skepu::no_dealloc);
	}
	std::cout << "==========================================================\n";
	std::cout << "Exit pointer constructor region\n";
	std::cout << "==========================================================\n";
	
	
	skel0(vec, vec);
	skel0(mat, mat);
	skel0(ten3, ten3);
	skel0(ten4, ten4);
	
	skel1(vec, vec);
	skel2(mat, mat);
	skel3(ten3, ten3);
	skel4(ten4, ten4);
	
}
