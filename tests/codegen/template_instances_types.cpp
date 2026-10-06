#include "../../external/catch2/catch.hpp"

#include <skepu>

// Skeleton instances in function templates that are instantiated with different types, so that
// each instantiation has its own user functions (and kernels). (No double: not all OpenCL devices have
// it; see template_instances_double.cpp.)

// ---- Map with a templated user function, element types float, unsigned and int

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

TEST_CASE("Map in a function template, three element types")
{
	const size_t size{16};
	skepu::Vector<float> xf(size, 1.f), yf(size, 2.f);
	skepu::Vector<unsigned> xu(size, 1u), yu(size, 2u);
	skepu::Vector<int> xi(size, 1), yi(size, 2);

	axpy(xf, yf, 3.f);
	axpy(xu, yu, 5u);
	axpy(xi, yi, 4);

	yf.updateHost(); yu.updateHost(); yi.updateHost();
	for (size_t i = 0; i < size; ++i)
	{
		REQUIRE(yf(i) == 5.f);
		REQUIRE(yu(i) == 7u);
		REQUIRE(yi(i) == 6);
	}
}

// ---- MapReduce with two templated user functions and a type that is not the element type

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

template<typename T>
T dot(skepu::Vector<T> &x, skepu::Vector<T> &y)
{
	auto skel = skepu::MapReduce<2>(prod_uf<T>, sum_uf<T>);
	return skel(x, y);
}

TEST_CASE("MapReduce in a function template, two element types")
{
	const size_t size{10};
	skepu::Vector<float> xf(size, 2.f), yf(size, 3.f);
	skepu::Vector<int> xi(size, 4), yi(size, 5);

	REQUIRE(dot(xf, yf) == 60.f);
	REQUIRE(dot(xi, yi) == 200);
}

// ---- A non-type template parameter that selects the user function

template<int N>
int add_n_uf(int x)
{
	return x + N;
}

template<int N>
void add_n(skepu::Vector<int> &v)
{
	auto skel = skepu::Map<1>(add_n_uf<N>);
	skel(v, v);
}

TEST_CASE("Map in a function template with a non-type parameter")
{
	skepu::Vector<int> v(8, 0);
	add_n<1>(v);
	add_n<10>(v);
	add_n<100>(v);
	add_n<1>(v);

	v.updateHost();
	for (size_t i = 0; i < v.size(); ++i)
		REQUIRE(v(i) == 112);
}

// ---- Scan and MapOverlap

template<typename T>
T plus_uf(T a, T b)
{
	return a + b;
}

template<typename T>
void prefix_sums(skepu::Vector<T> &in, skepu::Vector<T> &out)
{
	auto skel = skepu::Scan(plus_uf<T>);
	skel.setScanMode(skepu::ScanMode::Inclusive);
	skel(out, in);
}

template<typename T>
T stencil_uf(skepu::Region1D<T> r)
{
	return r(-1) + r(0) + r(1);
}

template<typename T>
void stencil(skepu::Vector<T> &in, skepu::Vector<T> &out)
{
	auto skel = skepu::MapOverlap(stencil_uf<T>);
	skel.setOverlap(1);
	skel.setEdgeMode(skepu::Edge::Pad);
	skel.setPad(T(0));
	skel(out, in);
}

TEST_CASE("Scan and MapOverlap in function templates")
{
	const size_t size{8};
	skepu::Vector<float> inf(size, 1.f), outf(size);
	skepu::Vector<int> ini(size, 2), outi(size);

	prefix_sums(inf, outf);
	prefix_sums(ini, outi);
	outf.updateHost(); outi.updateHost();
	for (size_t i = 0; i < size; ++i)
	{
		REQUIRE(outf(i) == float(i + 1));
		REQUIRE(outi(i) == int(2 * (i + 1)));
	}

	skepu::Vector<float> sf(size);
	skepu::Vector<int> si(size);
	stencil(inf, sf);
	stencil(ini, si);
	sf.updateHost(); si.updateHost();
	for (size_t i = 0; i < size; ++i)
	{
		const bool edge = (i == 0 || i == size - 1);
		REQUIRE(sf(i) == (edge ? 2.f : 3.f));
		REQUIRE(si(i) == (edge ? 4 : 6));
	}
}

// ---- A template that calls another template: the inner skeleton is instantiated from two places

template<typename T>
void twice(skepu::Vector<T> &x, skepu::Vector<T> &y, T a)
{
	axpy(x, y, a);
	axpy(x, y, a);
}

TEST_CASE("Nested function templates")
{
	skepu::Vector<float> xf(4, 1.f), yf(4, 0.f);
	skepu::Vector<int> xi(4, 1), yi(4, 0);
	twice(xf, yf, 2.f);
	twice(xi, yi, 3);
	yf.updateHost(); yi.updateHost();
	for (size_t i = 0; i < 4; ++i)
	{
		REQUIRE(yf(i) == 4.f);
		REQUIRE(yi(i) == 6);
	}
}
