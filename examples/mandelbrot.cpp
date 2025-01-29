/*!
Mandelbrot fractals. The Mandelbrot set
{
"B. B. Mandelbrot. Fractal aspects of the iteration of z → λz(1 − z) for complex λ and z.
Annals of the New York Academy of Sciences, 357:249–259, December 1980."
}
is a set of complex numbers which boundary draws a fractal in the complex numbers plane. A complex number c lies
within the Mandelbrot set, if the sequence
z_{i+1} = z_{i}2 + c
with i ∈ N, starting with z0 = 0 does not escape to infinity, otherwise c is not part of
the Mandelbrot set.

When computing a Mandelbrot fractal, the sequence in equation 3.1 is calculated
for every pixel of an image representing a section of the complex numbers plane. If a
given threshold is crossed, it is presumed that the sequence will escape to infinity and
that the pixel is not inside the Mandelbrot set. If the threshold is not crossed for a given
number of steps in the sequence, the pixel is taken as a member of the Mandelbrot
set. A pixel within the Mandelbrot set painted in black, other pixels are given a color
that corresponds to the number of sequence steps that have been calculated before
excluding the pixel from the Mandelbrot set. By setting the threshold and the number
of sequence steps accordingly, the calculation of the fractal can be a time-consuming
task. However, as all pixels are calculated independently, it is a common benchmark
application for data-parallel computations.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/filter.hpp>

#include "lodepng.h"

[[skepu::userconstant]] constexpr float
	CENTER_X = -0.5f,
	CENTER_Y = +0.5f,
	SCALE = 2,
	ANGLE_RAD = 0.5;

[[skepu::userconstant]] constexpr size_t
	MAX_ITERS = 50;


void WritePngFileBinaryMatrix(skepu::Matrix<size_t> &imageData)
{
	using namespace skepu::filter;
	auto copy = skepu::container_like<RGBPixel>(imageData);
	auto multiplier = skepu::Map([](size_t e) -> RGBPixel {
		RGBPixel res;
		res.r = 255 - (float)e / MAX_ITERS * 255;
		res.g = 255 - (float)e / MAX_ITERS * 255;
		res.b = 255 - (float)e / MAX_ITERS * 255;
		return res;
	});
	multiplier(copy, imageData);
	skepu::external(skepu::read(copy), [&]
	{
		unsigned error = lodepng::encode("mandelbrot.png", reinterpret_cast<unsigned char*>(copy.data()),
			copy.total_cols(), copy.total_rows(), LCT_RGB);
		if (error) SKEPU_ERROR("decoder error " << error << ": " << lodepng_error_text(error));
	});
}


struct cplx 
{
	float a, b;
};

cplx mult_c(cplx lhs, cplx rhs)
{
	cplx r;
	r.a = lhs.a * rhs.a - lhs.b * rhs.b;
	r.b = lhs.b * rhs.a + lhs.a * rhs.b;
	return r;
}

cplx add_c(cplx lhs, cplx rhs)
{
	cplx r;
	r.a = lhs.a + rhs.a;
	r.b = lhs.b + rhs.b;
	return r;
}

size_t mandelbrot_f(skepu::Index2D index, float height, float width)
{
	float angle = ANGLE_RAD;
	cplx a, c;
	a.a = 1.f / width * (index.col - width/2.f);
	a.b = 1.f / width * (index.row - height/2.f);
	
	a.a = SCALE * (a.a) + CENTER_X;
	a.b = SCALE * (a.b) + CENTER_Y;
	
	c.a = a.a * cos(angle) - a.b * sin(angle);
	c.b = a.a * sin(angle) + a.b * cos(angle);
	
	a.a = c.a;
	a.b = c.b;
	
	
	for (size_t i = 0; i < MAX_ITERS; ++i)
	{
		a = add_c(mult_c(a, a), c);
		if ((a.a * a.a + a.b * a.b) > 4)
			return i;
	}
	return MAX_ITERS;
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " width height backend\n";
		exit(1);
	}
	
	const size_t width = std::stoul(argv[1]);
	const size_t height = std::stoul(argv[2]);
	auto spec = skepu::BackendSpec{argv[3]};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Matrix<size_t> iterations(height, width);
	
	auto mandelbroter = skepu::Map<0>(mandelbrot_f);
	mandelbroter(iterations, height, width);
	
	WritePngFileBinaryMatrix(iterations);
}
