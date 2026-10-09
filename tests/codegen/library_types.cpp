#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/filter.hpp>
#include <test_helper_automatic_setup.hpp>

// Types of the standard library of SkePU (skepu::filter) as elements of containers and parameters of user functions. They are declared in
// a header, not in the program: the precompiler gives the OpenCL code their names and definitions (the library does not have to).

namespace
{
	skepu::filter::RGBPixel brighten(skepu::filter::RGBPixel a, skepu::filter::RGBPixel b)
	{
		skepu::filter::RGBPixel r;
		r.r = (unsigned char)(a.r + b.r);
		r.g = (unsigned char)(a.g + b.g);
		r.b = (unsigned char)(a.b + b.b);
		return r;
	}

	float luminance(skepu::filter::HSVPixel p)
	{
		return p.v * 0.5f + p.s * 0.25f + p.h * 0.125f;
	}
}

TEST_CASE("Library types as elements of containers")
{
	const size_t n = 64;
	skepu::Vector<skepu::filter::RGBPixel> a(n), b(n), out(n);
	for (size_t i = 0; i < n; ++i)
	{
		a(i).r = (unsigned char)i; a(i).g = (unsigned char)(2 * i); a(i).b = 3;
		b(i).r = 1; b(i).g = 2; b(i).b = (unsigned char)i;
	}

	auto pixels = skepu::Map<2>(brighten);
	pixels(out, a, b);
	out.flush();
	for (size_t i = 0; i < n; ++i)
	{
		REQUIRE(out(i).r == (unsigned char)(i + 1));
		REQUIRE(out(i).g == (unsigned char)(2 * i + 2));
		REQUIRE(out(i).b == (unsigned char)(3 + i));
	}

	skepu::Vector<skepu::filter::HSVPixel> hsv(n);
	skepu::Vector<float> lum(n);
	for (size_t i = 0; i < n; ++i)
	{
		hsv(i).h = float(i); hsv(i).s = 0.5f; hsv(i).v = 2.0f;
	}
	auto light = skepu::Map<1>(luminance);
	light(lum, hsv);
	lum.flush();
	for (size_t i = 0; i < n; ++i)
		REQUIRE(lum(i) == Approx(1.0f + 0.125f + float(i) * 0.125f));
}
