// Perlin noise: a texture (2D), or a stack of textures that change smoothly from one frame to the next (3D),
// made of several octaves of gradient noise.
//
// Every octave has a grid of random unit vectors (gradients), which is filled by a Map whose user function draws
// from a skepu::PRNG (the numbers are a function of the seed only, so the result is the same on every backend).
// The noise at an element is found by a second Map, over the whole texture, that reads the gradients at the corners
// of the cell that the element is in from the grid (a random-access container of another size than the texture),
// and blends their contributions with a smooth curve. The octaves are added up in the texture:
// the grid of octave o has twice the frequency and persistence^o times the amplitude of that of octave o-1.
//
// Usage: perlin height width persistence octaves base_frequency seed [backend] [--frames N] [--out PREFIX]   (--help for the details)
//   persistence:    the amplitude of an octave relative to that of the one before (e.g. 0.5)
//   octaves:        the number of octaves
//   base_frequency: the number of grid cells over the width in the first octave (e.g. 4)
//   --frames N:     make N frames (a Tensor3 of N textures) instead of one texture, with noise that is
//                   continuous also between the frames; a cell is as many frames deep as it is pixels wide
//   --out PREFIX:   write the texture to PREFIX.png (the frames to PREFIX0000.png, PREFIX0001.png, ...)
//
// The program prints the mean, the minimum and the maximum of the noise, which are a checksum of the result.
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <skepu>
#include <skepu-lib/args.hpp>
#include <skepu-lib/benchmark.hpp>
#include <skepu-lib/io.hpp>
#include <skepu-lib/util.hpp>

#include "lodepng.h"


struct vec2 { float x, y; };
struct vec3 { float x, y, z; };

// A random direction in the plane
vec2 gradient_2d(skepu::Random<1> &prng)
{
	const float angle = prng.getNormalized() * 6.28318530718f;
	vec2 g;
	g.x = cos(angle);
	g.y = sin(angle);
	return g;
}

// A random direction in space (uniform on the sphere)
vec3 gradient_3d(skepu::Random<2> &prng)
{
	const float z = prng.getNormalized() * 2.f - 1.f;
	const float angle = prng.getNormalized() * 6.28318530718f;
	const float r = sqrt(1.f - z * z);
	vec3 g;
	g.x = r * cos(angle);
	g.y = r * sin(angle);
	g.z = z;
	return g;
}

// The curve that blends the corners of a cell (the one of Perlin's improved noise: its first and second derivatives are zero at 0 and 1)
float fade(float t)
{
	return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

float interpolate(float a, float b, float t)
{
	return a + t * (b - a);
}

// The cell of the grid that a position is in: the corner with the lowest index (the last cell for a position at the very edge)
int cell_of(float position, size_t grid_size)
{
	const int cell = (int)position;
	return (cell < (int)grid_size - 2) ? cell : (int)grid_size - 2;
}

// One corner's contribution: the gradient there times the vector from the corner to the position
float corner_2d(vec2 g, float dy, float dx)
{
	return g.y * dy + g.x * dx;
}

float corner_3d(vec3 g, float dz, float dy, float dx)
{
	return g.z * dz + g.y * dy + g.x * dx;
}

// Adds an octave of 2D noise to the texture: the value of the element in [0, 1] times the amplitude. The position in the
// grid is the position in the texture divided by the size of a cell, in the same unit in both directions. The noise of the
// corners is in [-sqrt(2)/2, sqrt(2)/2], which is mapped to [0, 1].
float perlin_2d(skepu::Index2D index, float noise, skepu::Mat<vec2> grid, float cell_size, float amplitude)
{
	const float y = (float)index.row / cell_size;
	const float x = (float)index.col / cell_size;
	const int i = cell_of(y, grid.rows);
	const int j = cell_of(x, grid.cols);
	const float dy = y - i;
	const float dx = x - j;

	const float n00 = corner_2d(grid(i,     j),     dy,       dx);
	const float n01 = corner_2d(grid(i,     j + 1), dy,       dx - 1.f);
	const float n10 = corner_2d(grid(i + 1, j),     dy - 1.f, dx);
	const float n11 = corner_2d(grid(i + 1, j + 1), dy - 1.f, dx - 1.f);

	const float u = fade(dx);
	const float v = fade(dy);
	const float n = interpolate(interpolate(n00, n01, u), interpolate(n10, n11, u), v);
	return noise + amplitude * (0.5f + n * 0.70710678f);
}

// The same in 3D (the noise of the corners is in [-sqrt(3)/2, sqrt(3)/2]). The element (i, j, k) of the texture is
// the frame i, the row j and the column k.
float perlin_3d(skepu::Index3D index, float noise, skepu::Ten3<vec3> grid, float cell_size, float amplitude)
{
	const float z = (float)index.i / cell_size;
	const float y = (float)index.j / cell_size;
	const float x = (float)index.k / cell_size;
	const int h = cell_of(z, grid.size_i);
	const int i = cell_of(y, grid.size_j);
	const int j = cell_of(x, grid.size_k);
	const float dz = z - h;
	const float dy = y - i;
	const float dx = x - j;

	const float n000 = corner_3d(grid(h,     i,     j),     dz,       dy,       dx);
	const float n001 = corner_3d(grid(h,     i,     j + 1), dz,       dy,       dx - 1.f);
	const float n010 = corner_3d(grid(h,     i + 1, j),     dz,       dy - 1.f, dx);
	const float n011 = corner_3d(grid(h,     i + 1, j + 1), dz,       dy - 1.f, dx - 1.f);
	const float n100 = corner_3d(grid(h + 1, i,     j),     dz - 1.f, dy,       dx);
	const float n101 = corner_3d(grid(h + 1, i,     j + 1), dz - 1.f, dy,       dx - 1.f);
	const float n110 = corner_3d(grid(h + 1, i + 1, j),     dz - 1.f, dy - 1.f, dx);
	const float n111 = corner_3d(grid(h + 1, i + 1, j + 1), dz - 1.f, dy - 1.f, dx - 1.f);

	const float u = fade(dx);
	const float v = fade(dy);
	const float w = fade(dz);
	const float n = interpolate(
		interpolate(interpolate(n000, n001, u), interpolate(n010, n011, u), v),
		interpolate(interpolate(n100, n101, u), interpolate(n110, n111, u), v), w);
	return noise + amplitude * (0.5f + n * 0.57735027f);
}

// The gray level of an element of the texture: the noise divided by its largest possible value (the sum of the amplitudes)
unsigned char to_gray(float noise, float scale)
{
	const float level = noise * scale * 255.f;
	return (unsigned char)((level < 0.f) ? 0.f : (level > 255.f) ? 255.f : level);
}


// Writes an 8-bit grayscale PNG file (lodepng)
static void write_png(const std::string &path, const unsigned char *pixels, size_t width, size_t height)
{
	const unsigned error = lodepng::encode(path, pixels, (unsigned)width, (unsigned)height, LCT_GREY, 8);
	if (error)
		SKEPU_ERROR("cannot write " << path << ": " << lodepng_error_text(error));
}


// The mean, minimum and maximum of the noise, as a fraction of its largest possible value. The start value of
// a reduction is the identity of its operator: 0 for the sum, but not for the minimum and the maximum.
template<typename Container>
static void print_statistics(Container &noise, float max_value)
{
	auto sum = skepu::Reduce(skepu::util::add<float>);
	auto lowest = skepu::Reduce(skepu::util::min<float>);
	auto highest = skepu::Reduce(skepu::util::max<float>);
	lowest.setStartValue(std::numeric_limits<float>::max());
	highest.setStartValue(std::numeric_limits<float>::lowest());
	skepu::io::cout << std::fixed << std::setprecision(4) << "Noise: mean " << sum(noise) / noise.size() / max_value
		<< ", min " << lowest(noise) / max_value << ", max " << highest(noise) / max_value << " (of 1)\n";
}




// (a skeleton instance cannot be made in a function template that has a template template parameter, so this one is here)
static auto gray_level = skepu::Map<1>(to_gray);


// Writes the texture to PREFIX.png, or the frames to PREFIX0000.png, PREFIX0001.png, ...
static void write_images(const std::string &prefix, skepu::Matrix<unsigned char> &image)
{
	skepu::external(skepu::read(image), [&] { write_png(prefix + ".png", image.data(), image.total_cols(), image.total_rows()); });
}

static void write_images(const std::string &prefix, skepu::Tensor3<unsigned char> &images)
{
	skepu::external(skepu::read(images), [&]
	{
		for (size_t f = 0; f < images.size_i(); ++f)
		{
			std::stringstream name;
			name << prefix << std::setw(4) << std::setfill('0') << f << ".png";
			write_png(name.str(), images.data() + f * images.size_j() * images.size_k(), images.size_k(), images.size_j());
		}
	});
}

// Makes the noise of a texture, a Matrix (extents: height, width) or a Tensor3 (extents: frames, height, width), with the
// gradients and the skeletons of its dimension, and reports on it. The arguments of the program are read from the parser
// (the names that main declared are the same arguments).
template<template<typename> class Texture, typename Gradient, typename MakeGradients, typename Render, typename... Extents>
static void make_noise(skepu::args::Parser &args, MakeGradients &make_gradients, Render &render, Extents... extents)
{
	const size_t width = args.get<size_t>("width"), octaves = args.get<size_t>("octaves");
	const float persistence = args.get<float>("persistence"), base_frequency = args.get<float>("base_frequency");
	skepu::PRNG prng(args.get<unsigned long>("seed"));
	make_gradients.setPRNG(prng);

	float max_value = 0; // the sum of the amplitudes
	for (size_t o = 0; o < octaves; ++o)
		max_value += pow(persistence, o);

	// The size of a grid cell is the same in all directions, so the noise does not stretch with the shape of the texture
	auto cell_size_of = [&](size_t octave) { return (float)width / (base_frequency * pow(2.f, (float)octave)); };
	auto grid_size_of = [&](size_t octave, size_t extent) { return (size_t)ceil(extent / cell_size_of(octave)) + 1; };

	Texture<float> noise(extents..., "noise", 0.f);
	auto duration = skepu::benchmark::measure([&]
	{
		for (size_t o = 0; o < octaves; ++o)
		{
			SKEPU_TRACE_SCOPE("Octave");
			Texture<Gradient> grid(grid_size_of(o, extents)..., "gradients");
			make_gradients(grid);
			render(noise, noise, grid, cell_size_of(o), (float)pow(persistence, o));
			SKEPU_TRACE_SNAPSHOT(noise); // (if the program is built for it: make TRACE_SNAPSHOTS=1)
		}
	});
	skepu::io::cout << "Duration: " << duration.count() / 1E6 << " s.\n";

	Texture<unsigned char> image(extents..., "image", 0);
	gray_level(image, noise, 1.f / max_value);
	print_statistics(noise, max_value);

	if (const std::string out = args.named<std::string>("out", ""); !out.empty())
		write_images(out, image);
}


int main(int argc, char *argv[])
{
	skepu::args::Parser args(argc, argv, "Perlin noise: a texture, or frames of noise that change smoothly from one frame to the next");
	const size_t height        = args.get<size_t>("height");
	const size_t width         = args.get<size_t>("width");
	const float persistence    = args.get<float>("persistence");
	const size_t octaves       = args.get<size_t>("octaves");
	const float base_frequency = args.get<float>("base_frequency");
	args.get<unsigned long>("seed");   // (make_noise reads this one, and the output file, from the parser)
	const size_t frames        = args.named<size_t>("frames", 0, "make this many frames (a Tensor3) instead of one texture");
	args.named<std::string>("out", "", "write the texture to PREFIX.png (the frames to PREFIX0000.png, ...)");
	args.backend();

	if (height < 1 || width < 1 || octaves < 1 || persistence <= 0 || base_frequency <= 0)
	{
		skepu::io::cout << "The sizes, the number of octaves, the persistence and the frequency must be positive\n";
		return 1;
	}

	// (the skeletons are made in the branch that uses them: the OpenCL kernels of an instance are compiled when it is made)
	if (frames == 0)
	{
		auto gradients_2d = skepu::Map<0>(gradient_2d);
		auto render_2d = skepu::Map<1>(perlin_2d);
		make_noise<skepu::Matrix, vec2>(args, gradients_2d, render_2d, height, width);
	}
	else
	{
		auto gradients_3d = skepu::Map<0>(gradient_3d);
		auto render_3d = skepu::Map<1>(perlin_3d);
		make_noise<skepu::Tensor3, vec3>(args, gradients_3d, render_3d, frames, height, width);
	}
	return 0;
}
