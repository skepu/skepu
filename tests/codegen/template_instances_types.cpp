// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>


// ======== template_instances_types ========

namespace template_instances_types_part
{
// Skeleton instances in function templates that are instantiated with different types, so that
// each instantiation has its own user functions (and kernels). Double precision too: on an OpenCL device that does
// not have it, the skeletons with doubles run on another backend (see opencl_double_fallback.cpp).

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

TEST_CASE("template_instances_types: Map in a function template, three element types")
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

TEST_CASE("template_instances_types: MapReduce in a function template, two element types")
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

TEST_CASE("template_instances_types: Map in a function template with a non-type parameter")
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

TEST_CASE("template_instances_types: Scan and MapOverlap in function templates")
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

TEST_CASE("template_instances_types: Nested function templates")
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

// A second template parameter, defaulted from the first, that is not the element type of the vectors
template<typename T, typename R = T>
R dot_default(skepu::Vector<T> &x, skepu::Vector<T> &y)
{
	auto skel = skepu::MapReduce<2>(prod_uf<R>, sum_uf<R>);
	return skel(x, y);
}

TEST_CASE("template_instances_types: Map in a function template, float and double")
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

TEST_CASE("template_instances_types: MapReduce in a function template, float and double")
{
	const size_t size{10};
	skepu::Vector<float> xf(size, 2.f), yf(size, 3.f);
	skepu::Vector<double> xd(size, 0.5), yd(size, 4.);

	REQUIRE(dot_default(xf, yf) == 60.f);
	REQUIRE(dot_default(xd, yd) == 20.);
}
}


// ======== multiple_return_types ========

namespace multiple_return_types_part
{
// User functions that return skepu::multiple of built-in types, which have names of more than one word (signed char, unsigned int, unsigned short) or are
// not in a namespace. (No double: not all OpenCL devices have it.)

skepu::multiple<signed char, unsigned int> split(int x)
{
	return skepu::ret((signed char)(x % 100 - 50), (unsigned int)(x / 100));
}

skepu::multiple<unsigned short, unsigned char, float> three(int x)
{
	return skepu::ret((unsigned short)(x % 60000), (unsigned char)(x % 256), x * 0.5f);
}

auto skel_split = skepu::Map<1>(split);
auto skel_three = skepu::Map<1>(three);

TEST_CASE("multiple_return_types: multiple of built-in types of one or more words")
{
	const size_t n{300};
	skepu::Vector<int> in(n);
	skepu::Vector<signed char> small(n);
	skepu::Vector<unsigned int> big(n);
	for (size_t i = 0; i < n; ++i) in[i] = (int)(i * 7);

	skel_split(small, big, in);
	small.updateHost(); big.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		REQUIRE((int)small(i) == (int)(i * 7) % 100 - 50);
		REQUIRE(big(i) == (unsigned int)((i * 7) / 100));
	}

	skepu::Vector<unsigned short> wide(n);
	skepu::Vector<unsigned char> byte(n);
	skepu::Vector<float> half(n);
	skel_three(wide, byte, half, in);
	wide.updateHost(); byte.updateHost(); half.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		REQUIRE((int)wide(i) == (int)(i * 7) % 60000);
		REQUIRE((int)byte(i) == (int)(i * 7) % 256);
		REQUIRE(half(i) == (float)(i * 7) * 0.5f);
	}
}
}


// ======== template_instances ========

namespace template_instances_part
{
// A function template with a skeleton instance, instantiated twice with the same user
// function (the template parameter only changes the type of the scalar argument).
float scale_uf(float x, float factor)
{
	return x * factor;
}

template<typename S>
void scale(skepu::Vector<float> &v, S factor)
{
	auto skel = skepu::Map<1>(scale_uf);
	skel.setStride(1, 1);
	skel(v, v, factor);
}

TEST_CASE("template_instances: Skeleton in a function template, instantiated twice")
{
	const size_t size{8};
	skepu::Vector<float> v(size);
	for (size_t i = 0; i < size; ++i)
		v[i] = i + 1;

	scale(v, 2.f);   // S = float
	scale(v, 3);     // S = int

	v.updateHost();
	for (size_t i = 0; i < size; ++i)
		REQUIRE(v(i) == 6.f * (i + 1));
}
}


// ======== usertype_return ========

namespace usertype_return_part
{
// The struct only appears as a return type (no variable or parameter of the type
// in the user function), so the OpenCL source needs it from the return type.
struct Pt
{
	float x, y;
};

Pt make_pt(skepu::Index1D index, float scale)
{
	return (Pt){ index.i * scale, index.i * scale + 1 };
}

auto gen = skepu::Map<0>(make_pt);

TEST_CASE("usertype_return: User type only in the return type")
{
	const size_t size{10};

	skepu::Vector<Pt> v(size);
	gen(v, 2.f);
	v.updateHost();

	for (size_t i = 0; i < size; ++i)
	{
		REQUIRE(v(i).x == i * 2.f);
		REQUIRE(v(i).y == i * 2.f + 1);
	}
}

// ---- A random access container that is subscripted with [] in a templated user function, and cast: the subscript is on the container in
// the instantiated function too

template<typename T>
float picked_uf(skepu::Index1D index, skepu::Vec<T> values)
{
	return (float)(int)values[(index.i * 3) % 8] + 0.5f;
}

template<typename T>
void pick(skepu::Vector<float> &out, skepu::Vector<T> &values)
{
	auto skel = skepu::Map<0>(picked_uf<T>);
	skel(out, values);
}

TEST_CASE("template_instances_types: subscript on a container in a templated user function, two element types")
{
	skepu::Vector<int> ints(8);
	skepu::Vector<float> floats(8);
	skepu::Vector<float> out(8, 0.f), out_float(8, 0.f);
	for (size_t i = 0; i < 8; ++i) { ints(i) = (int)i * 10; floats(i) = (float)i * 10; }

	pick(out, ints);
	pick(out_float, floats);
	out.updateHost();
	out_float.updateHost();

	for (size_t i = 0; i < 8; ++i)
	{
		REQUIRE(out(i) == (float)(((i * 3) % 8) * 10) + 0.5f);
		REQUIRE(out_float(i) == out(i));
	}
}
}
