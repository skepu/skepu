#define SKEPU_ENABLE_EXCEPTIONS

#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>


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
	skepu::Tensor3<float> t3(10, 10, 10, 3.142f);
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
	skepu::Tensor4<float> t3(10, 10, 10, 10, 3.142f);
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