#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"

#include <iostream>
#include <skepu>
#include <test_helper_automatic_setup.hpp>

TEST_CASE("Vector lifecycle")
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

TEST_CASE("Matrix lifecycle")
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


TEST_CASE("Tensor3 lifecycle")
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


TEST_CASE("Tensor4 lifecycle")
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

TEST_CASE("Vector lifecycle: move construction, move assignment and std::swap")
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

TEST_CASE("Matrix lifecycle: move construction, move assignment and std::swap")
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

TEST_CASE("Tensor3 lifecycle: move construction, move assignment and std::swap")
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

TEST_CASE("Tensor4 lifecycle: move construction, move assignment and std::swap")
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
