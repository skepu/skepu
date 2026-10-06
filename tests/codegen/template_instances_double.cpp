#include "../../external/catch2/catch.hpp"

#include <skepu>

// Skeleton instances in function templates, with double precision (not all OpenCL devices have it, so this
// test is not built for OpenCL; see template_instances_types.cpp).

template<typename T>
T axpy_uf(T x, T y, T a)
{
	return a * x + y;
}

template<typename T>
void axpy(skepu::Vector<T> &x, skepu::Vector<T> &y, T a)
{
	auto skel = skepu::Map<2>(axpy_uf<T>);
	skel(y, x, y, a);
}

template<typename T>
T prod_uf(T x, T y)
{
	return x * y;
}

template<typename T>
T sum_uf(T x, T y)
{
	return x + y;
}

// A second template parameter, defaulted from the first, that is not the element type of the vectors
template<typename T, typename R = T>
R dot(skepu::Vector<T> &x, skepu::Vector<T> &y)
{
	auto skel = skepu::MapReduce<2>(prod_uf<R>, sum_uf<R>);
	return skel(x, y);
}

TEST_CASE("Map in a function template, float and double")
{
	const size_t size{16};
	skepu::Vector<float> xf(size, 1.f), yf(size, 2.f);
	skepu::Vector<double> xd(size, 1.), yd(size, 2.);

	axpy(xf, yf, 3.f);
	axpy(xd, yd, 0.5);

	yf.updateHost(); yd.updateHost();
	for (size_t i = 0; i < size; ++i)
	{
		REQUIRE(yf(i) == 5.f);
		REQUIRE(yd(i) == 2.5);
	}
}

TEST_CASE("MapReduce in a function template, float and double")
{
	const size_t size{10};
	skepu::Vector<float> xf(size, 2.f), yf(size, 3.f);
	skepu::Vector<double> xd(size, 0.5), yd(size, 4.);

	REQUIRE(dot(xf, yf) == 60.f);
	REQUIRE(dot(xd, yd) == 20.);
}
