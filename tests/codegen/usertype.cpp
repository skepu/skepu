#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>

struct Complex
{
	float re, im;
};

auto adder = skepu::Map<2>([](Complex lhs, Complex rhs) -> Complex
{
	Complex res;
	res.re = rhs.re + lhs.re;
	res.im = rhs.im + lhs.im;
	return res;
});

auto multer = skepu::Map<2>([](Complex rhs, Complex lhs) -> Complex
{
	Complex r;
	r.re = lhs.re * rhs.re - lhs.im * rhs.im;
	r.im = lhs.im * rhs.re + lhs.re * rhs.im;
	return r;
});


Complex add_c(Complex rhs, Complex lhs)
{
	Complex res;
	res.re = rhs.re + lhs.re;
	res.im = rhs.im + lhs.im;
	return res;
}

Complex mul_c(Complex rhs, Complex lhs)
{
	Complex r;
	r.re = lhs.re * rhs.re - lhs.im * rhs.im;
	r.im = lhs.im * rhs.re + lhs.re * rhs.im;
	return r;
}

auto other = skepu::Map<2>([](Complex rhs, Complex lhs) -> Complex
{
	Complex r;
	r = add_c(lhs, rhs);
	return r;
});

/* //TODO: struct templates in OpenCL
template<typename T>
struct MyType
{
	T val;
};

auto sumthin = skepu::Map<2>([](MyType<float> a, MyType<float> b) -> MyType<float>
{
	MyType<float> res;
	res.val = a.val + b.val;
	return res;
});
*/

[[skepu::userconstant]] constexpr float
	CENTER_X = -.5f,
	CENTER_Y = 0.f,
	SCALE = 2.5f;

[[skepu::userconstant]] constexpr size_t
	MAX_ITERS = 1000;

size_t mandelbrot_f(skepu::Index2D index, float height, float width)
{
	Complex a;
	a.re = (1.0 / width) * (index.col - (width * 0.5));
	a.im = (1.0 / height) * (index.row - (height * 0.5));
	Complex c = a;
	
	for (size_t i = 0; i < 1000; ++i)
	{
		a = add_c(mul_c(a, a), c);
		if ((a.re * a.re + a.im * a.im) > 4)
			return i;
	}
	return 1000;
}


auto mandelbroter = skepu::Map<0>(mandelbrot_f);


TEST_CASE("User type")
{
	const size_t size{100};

	skepu::Vector<Complex> v1(size), v2(size), r(size);

	multer(r, v1, v2);
	
}
