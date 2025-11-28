#define SKEPU_ENABLE_EXCEPTIONS

#include "../../external/catch2/catch.hpp"

#include <iostream>
#include <skepu>


TEST_CASE("Vector swap")
{
	skepu::Vector<float> v1(100);
	skepu::Vector<float> v2(313);
	
	v1.swap(v2);
	
	CHECK(v1.size() == 313);
	CHECK(v2.size() == 100);
	
}

TEST_CASE("Matrix swap")
{
	skepu::Matrix<float> m1(10, 12);
	skepu::Matrix<float> m2(13, 15);
	
	m1.swap(m2);
	
	CHECK(m1.size() == 13*15);
	CHECK(m2.size() == 10*12);
	
	CHECK(m1.size_i() == 13);
	CHECK(m1.size_j() == 15);
	
	CHECK(m2.size_i() == 10);
	CHECK(m2.size_j() == 12);
}


TEST_CASE("Tensor3 swap")
{
	skepu::Tensor3<float> t1(10, 12, 3);
	skepu::Tensor3<float> t2(13, 15, 5);
	
	t1.swap(t2);
	
	CHECK(t1.size() == 13*15*5);
	CHECK(t2.size() == 10*12*3);
	
	CHECK(t1.size_i() == 13);
	CHECK(t1.size_j() == 15);
	CHECK(t1.size_k() == 5);
	
	CHECK(t2.size_i() == 10);
	CHECK(t2.size_j() == 12);
	CHECK(t2.size_k() == 3);
}


TEST_CASE("Tensor4 swap")
{
	skepu::Tensor4<float> t1(10, 12, 3, 2);
	skepu::Tensor4<float> t2(13, 15, 5, 1);
	
	t1.swap(t2);
	
	CHECK(t1.size() == 13*15*5*1);
	CHECK(t2.size() == 10*12*3*2);
	
	CHECK(t1.size_i() == 13);
	CHECK(t1.size_j() == 15);
	CHECK(t1.size_k() == 5);
	CHECK(t1.size_l() == 1);
	
	CHECK(t2.size_i() == 10);
	CHECK(t2.size_j() == 12);
	CHECK(t2.size_k() == 3);
	CHECK(t2.size_l() == 2);
}