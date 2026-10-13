// Merged from: containers.cpp, flatmap.cpp, lifecycle.cpp
// (one program instead of one for each: the build of a test program has a cost of its own; each part is in a namespace, and its test cases have its name as a prefix)
#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>
#include <math.h>
#include <iostream>


// ======== containers ========

namespace containers_part
{
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


TEST_CASE("containers: Container fundamentals")
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
}


// ======== flatmap ========

namespace flatmap_part
{
auto flatmap = skepu::Map([](int e){ return e; });

TEST_CASE("flatmap: Flat map")
{
	const size_t size{10};

	skepu::Matrix<int> mat(size, size, 2);
	skepu::Tensor3<int> ten3(size, size, size, "", 3);
	skepu::Tensor4<int> ten4(size, size, size, size, "", 4);
	
	skepu::Vector<int> dest_A(size*size), dest_B(size*size*size), dest_C(size*size*size*size);
	
	
	flatmap(dest_A, mat);
	flatmap(dest_B, ten3);
	flatmap(dest_C, ten4);
	
	skepu::external(skepu::read(dest_A, dest_B, dest_C), [&]
	{
		CHECK(dest_A(0) == 2);
		CHECK(dest_A(size*size-1) == 2);
		CHECK(dest_B(0) == 3);
		CHECK(dest_B(size*size*size-1) == 3);
		CHECK(dest_C(0) == 4);
		CHECK(dest_C(size*size*size*size-1) == 4);
	});
	
}
}


// ======== lifecycle ========

namespace lifecycle_part
{
TEST_CASE("lifecycle: Vector lifecycle")
{
	skepu::Vector<float> v1(100);
	CHECK(v1.size() == 100);
	v1(0) = 3.142f;
	CHECK(v1(0) == 3.142f);
	CHECK_THROWS(v1.init(1));
	
	// Late init
	skepu::Vector<float> v2;
	CHECK(v2.size() == 0);
	v2.init(100);
	CHECK(v2.size() == 100);
	v2(0) = 3.142f;
	CHECK(v2(0) == 3.142f);
	CHECK_THROWS(v2.init(1));

	// Default value
	skepu::Vector<float> v3(100, 3.142f);
	CHECK(v3.size() == 100);
	CHECK(v3(0) == 3.142f);
	
	// Late init + default value
	skepu::Vector<float> v4;
	CHECK(v4.size() == 0);
	v4.init(100, 3.142f);
	CHECK(v4.size() == 100);
	CHECK(v4(0) == 3.142f);
	
	// Copy constructor
	skepu::Vector<float> v5(v4);
	CHECK(v5.size() == 100);
	CHECK(v5(0) == 3.142f);
	v5(0) = 2.71f;
	CHECK(v5(0) == 2.71f);
	CHECK(v4(0) == 3.142f);
	
	// Initializer list constructor
	skepu::Vector<float> v6 = {5,4,3,2,1};
	CHECK(v6.size() == 5);
	CHECK(v6(0) == 5);
}

TEST_CASE("lifecycle: Matrix lifecycle")
{
	skepu::Matrix<float> m1(100, 100);
	CHECK(m1.size() == 100 * 100);
	m1(0, 0) = 3.142f;
	CHECK(m1(0, 0) == 3.142f);
	CHECK_THROWS(m1.init(1, 1));
	
	// Late init
	skepu::Matrix<float> m2;
	CHECK(m2.size() == 0);
	m2.init(100, 100);
	CHECK(m2.size() == 100 * 100);
	m2(0, 0) = 3.142f;
	CHECK(m2(0, 0) == 3.142f);
	CHECK_THROWS(m2.init(1, 1));

	// Default value
	skepu::Matrix<float> m3(100, 100, 3.142f);
	CHECK(m3.size() == 100 * 100);
	CHECK(m3(0, 0) == 3.142f);
	
	// Late init + default value
	skepu::Matrix<float> m4;
	CHECK(m4.size() == 0);
	m4.init(100, 100, 3.142f);
	CHECK(m4.size() == 100 * 100);
	CHECK(m4(0, 0) == 3.142f);
	
	// Copy constructor
	skepu::Matrix<float> m5(m4);
	CHECK(m5.size() == 100 * 100);
	CHECK(m5(0, 0) == 3.142f);
	m5(0, 0) = 2.71f;
	CHECK(m5(0, 0) == 2.71f);
	CHECK(m4(0, 0) == 3.142f);
}


TEST_CASE("lifecycle: Tensor3 lifecycle")
{
	skepu::Tensor3<float> t1(10, 10, 10);
	CHECK(t1.size() == 10 * 10 * 10);
	t1(0, 0, 0) = 3.142f;
	CHECK(t1(0, 0, 0) == 3.142f);
	CHECK_THROWS(t1.init(1, 1, 1));
	
	// Late init
	skepu::Tensor3<float> t2;
	CHECK(t2.size() == 0);
	CHECK_NOTHROW(t2.init(10, 10, 10));
	CHECK(t2.size() == 10 * 10 * 10);
	t2(0, 0, 0) = 3.142f;
	CHECK(t2(0, 0, 0) == 3.142f);
	CHECK_THROWS(t2.init(1, 1, 1));

	// Default value
	skepu::Tensor3<float> t3(10, 10, 10, "", 3.142f);
	CHECK(t3.size() == 10 * 10 * 10);
	CHECK(t3(0, 0, 0) == 3.142f);
	
	// Late init + default value
	skepu::Tensor3<float> t4;
	CHECK(t4.size() == 0);
	t4.init(10, 10, 10, 3.142f);
	CHECK(t4.size() == 10 * 10 * 10);
	CHECK(t4(0, 0, 0) == 3.142f);
	
	// Copy constructor
	skepu::Tensor3<float> t5(t4);
	CHECK(t5.size() == 10 * 10 * 10);
	CHECK(t5(0, 0, 0) == 3.142f);
	t5(0, 0, 0) = 2.71f;
	CHECK(t5(0, 0, 0) == 2.71f);
	CHECK(t4(0, 0, 0) == 3.142f);
}


TEST_CASE("lifecycle: Tensor4 lifecycle")
{
	skepu::Tensor4<float> t1(10, 10, 10, 10);
	CHECK(t1.size() == 10 * 10 * 10 * 10);
	t1(0, 0, 0, 0) = 3.142f;
	CHECK(t1(0, 0, 0, 0) == 3.142f);
	CHECK_THROWS(t1.init(1, 1, 1, 1));
	
	// Late init
	skepu::Tensor4<float> t2;
	CHECK(t2.size() == 0);
	t2.init(10, 10, 10, 10);
	CHECK(t2.size() == 10 * 10 * 10 * 10);
	t2(0, 0, 0, 0) = 3.142f;
	CHECK(t2(0, 0, 0, 0) == 3.142f);
	CHECK_THROWS(t2.init(1, 1, 1, 1));

	// Default value
	skepu::Tensor4<float> t3(10, 10, 10, 10, "", 3.142f);
	CHECK(t3.size() == 10 * 10 * 10 * 10);
	CHECK(t3(0, 0, 0, 0) == 3.142f);
	
	// Late init + default value
	skepu::Tensor4<float> t4;
	CHECK(t4.size() == 0);
	t4.init(10, 10, 10, 10, 3.142f);
	CHECK(t4.size() == 10 * 10 * 10 * 10);
	CHECK(t4(0, 0, 0, 0) == 3.142f);
	
	// Copy constructor
	skepu::Tensor4<float> t5(t4);
	CHECK(t5.size() == 10 * 10 * 10 * 10);
	CHECK(t5(0, 0, 0, 0) == 3.142f);
	t5(0, 0, 0, 0) = 2.71f;
	CHECK(t5(0, 0, 0, 0) == 2.71f);
	CHECK(t4(0, 0, 0, 0) == 3.142f);
}


// Move construction, move assignment and std::swap exchange the memory of the containers: nothing is copied (the pointer to the
// elements is the same afterwards). A container that wraps memory it does not own keeps it: the elements are copied into it.

TEST_CASE("lifecycle: Vector lifecycle: move construction, move assignment and std::swap")
{
	skepu::Vector<float> a(100, 1.f), b(50, 2.f);
	float *pa = a.getAddress(), *pb = b.getAddress();

	std::swap(a, b);
	CHECK(a.size() == 50);
	CHECK(b.size() == 100);
	CHECK(a.getAddress() == pb);
	CHECK(b.getAddress() == pa);
	CHECK(a(0) == 2.f);
	CHECK(b(99) == 1.f);

	// Move construction
	skepu::Vector<float> c(std::move(a));
	CHECK(c.size() == 50);
	CHECK(c.getAddress() == pb);
	CHECK(c(49) == 2.f);
	CHECK(a.size() == 0);

	// Move assignment
	skepu::Vector<float> d(7, 5.f);
	d = std::move(b);
	CHECK(d.size() == 100);
	CHECK(d.getAddress() == pa);
	CHECK(d(0) == 1.f);

	// A vector that wraps memory of the program keeps it
	float buffer[10] = {};
	skepu::Vector<float> wrapper(buffer, 10, false);
	skepu::Vector<float> source(10, 4.f);
	wrapper = std::move(source);
	CHECK(wrapper.getAddress() == buffer);
	CHECK(buffer[0] == 4.f);
	CHECK(buffer[9] == 4.f);
}

TEST_CASE("lifecycle: Matrix lifecycle: move construction, move assignment and std::swap")
{
	skepu::Matrix<float> a(10, 12, 1.f), b(5, 7, 2.f);
	float *pa = a.getAddress(), *pb = b.getAddress();

	std::swap(a, b);
	CHECK(a.size_i() == 5);
	CHECK(a.size_j() == 7);
	CHECK(b.size_i() == 10);
	CHECK(b.size_j() == 12);
	CHECK(a.getAddress() == pb);
	CHECK(b.getAddress() == pa);
	CHECK(a(0, 0) == 2.f);
	CHECK(b(9, 11) == 1.f);

	// Move construction
	skepu::Matrix<float> c(std::move(a));
	CHECK(c.size_i() == 5);
	CHECK(c.size_j() == 7);
	CHECK(c.getAddress() == pb);
	CHECK(a.size() == 0);

	// Move assignment
	skepu::Matrix<float> d(3, 3, 5.f);
	d = std::move(b);
	CHECK(d.size_i() == 10);
	CHECK(d.size_j() == 12);
	CHECK(d.getAddress() == pa);
	CHECK(d(0, 0) == 1.f);

	// A matrix that wraps memory of the program keeps it
	float buffer[6] = {};
	skepu::Matrix<float> wrapper(buffer, 2, 3, false);
	skepu::Matrix<float> source(2, 3, 4.f);
	wrapper = std::move(source);
	CHECK(wrapper.getAddress() == buffer);
	CHECK(buffer[0] == 4.f);
	CHECK(buffer[5] == 4.f);
}

TEST_CASE("lifecycle: Tensor3 lifecycle: move construction, move assignment and std::swap")
{
	skepu::Tensor3<float> a(4, 5, 6, "", 1.f), b(2, 3, 4, "", 2.f);
	float *pa = a.getAddress(), *pb = b.getAddress();

	std::swap(a, b);
	CHECK(a.size_i() == 2);
	CHECK(a.size_j() == 3);
	CHECK(a.size_k() == 4);
	CHECK(b.size_i() == 4);
	CHECK(b.size_j() == 5);
	CHECK(b.size_k() == 6);
	CHECK(a.getAddress() == pb);
	CHECK(b.getAddress() == pa);
	CHECK(a(1, 2, 3) == 2.f);
	CHECK(b(3, 4, 5) == 1.f);

	skepu::Tensor3<float> c(std::move(a));
	CHECK(c.size() == 2 * 3 * 4);
	CHECK(c.getAddress() == pb);
	CHECK(a.size() == 0);

	skepu::Tensor3<float> d(2, 2, 2, "", 5.f);
	d = std::move(b);
	CHECK(d.size_i() == 4);
	CHECK(d.size_k() == 6);
	CHECK(d.getAddress() == pa);
	CHECK(d(0, 0, 0) == 1.f);
}

TEST_CASE("lifecycle: Tensor4 lifecycle: move construction, move assignment and std::swap")
{
	skepu::Tensor4<float> a(3, 4, 5, 6, "", 1.f), b(2, 2, 3, 4, "", 2.f);
	float *pa = a.getAddress(), *pb = b.getAddress();

	std::swap(a, b);
	CHECK(a.size_i() == 2);
	CHECK(a.size_l() == 4);
	CHECK(b.size_i() == 3);
	CHECK(b.size_l() == 6);
	CHECK(a.getAddress() == pb);
	CHECK(b.getAddress() == pa);
	CHECK(a(1, 1, 2, 3) == 2.f);
	CHECK(b(2, 3, 4, 5) == 1.f);

	skepu::Tensor4<float> c(std::move(a));
	CHECK(c.size() == 2 * 2 * 3 * 4);
	CHECK(c.getAddress() == pb);
	CHECK(a.size() == 0);

	skepu::Tensor4<float> d(2, 2, 2, 2, "", 5.f);
	d = std::move(b);
	CHECK(d.size_j() == 4);
	CHECK(d.getAddress() == pa);
	CHECK(d(0, 0, 0, 0) == 1.f);
}
}
