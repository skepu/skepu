// The user functions of examples/perlin.cpp (keep them in step with it): the gradient grids filled by a Map with a PRNG
// (a user-defined struct as the element type), and the noise rendered by a Map that reads the grid with a random-access proxy.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>
#include <cmath>
#include <algorithm>

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


namespace
{
	// The noise of the octave: a grid of gradients for a texture, rendered with the given cell size
	skepu::Matrix<float> noise_2d(size_t rows, size_t cols, float cell_size, float amplitude, unsigned long seed)
	{
		skepu::Matrix<vec2> grid((size_t)std::ceil(rows / cell_size) + 1, (size_t)std::ceil(cols / cell_size) + 1, "gradients");
		auto make_gradients = skepu::Map<0>(gradient_2d);
		skepu::PRNG prng(seed);
		make_gradients.setPRNG(prng);
		make_gradients(grid);
		skepu::Matrix<float> noise(rows, cols, "noise", 0.f);
		skepu::Map<1>(perlin_2d)(noise, noise, grid, cell_size, amplitude);
		noise.updateHost(); // (the element access of a container does not synchronize with the device)
		return noise;
	}

	skepu::Tensor3<float> noise_3d(size_t frames, size_t rows, size_t cols, float cell_size, float amplitude, unsigned long seed)
	{
		skepu::Tensor3<vec3> grid((size_t)std::ceil(frames / cell_size) + 1, (size_t)std::ceil(rows / cell_size) + 1, (size_t)std::ceil(cols / cell_size) + 1, "gradients");
		auto make_gradients = skepu::Map<0>(gradient_3d);
		skepu::PRNG prng(seed);
		make_gradients.setPRNG(prng);
		make_gradients(grid);
		skepu::Tensor3<float> noise(frames, rows, cols, "noise", 0.f);
		skepu::Map<1>(perlin_3d)(noise, noise, grid, cell_size, amplitude);
		noise.updateHost();
		return noise;
	}
}


TEST_CASE("perlin: the gradients are unit vectors and a function of the seed")
{
	skepu::Matrix<vec2> g2(7, 9, "gradients 2D");
	skepu::Matrix<vec2> g2_again(7, 9, "gradients 2D again");
	skepu::Tensor3<vec3> g3(3, 4, 5, "gradients 3D");
	skepu::Tensor3<vec3> g3_again(3, 4, 5, "gradients 3D again");
	auto make_2d = skepu::Map<0>(gradient_2d);
	auto make_3d = skepu::Map<0>(gradient_3d);
	skepu::PRNG a(5), b(5);
	make_2d.setPRNG(a);
	make_2d(g2);
	make_3d.setPRNG(a);
	make_3d(g3);
	skepu::PRNG a_again(5);
	make_2d.setPRNG(a_again);
	make_2d(g2_again);
	make_3d.setPRNG(a_again);
	make_3d(g3_again);
	g2.updateHost();
	g2_again.updateHost();
	g3.updateHost();
	g3_again.updateHost();

	for (size_t i = 0; i < g2.size_i(); ++i)
		for (size_t j = 0; j < g2.size_j(); ++j)
		{
			REQUIRE(std::sqrt(g2(i, j).x * g2(i, j).x + g2(i, j).y * g2(i, j).y) == Approx(1.f).margin(1e-5));
			REQUIRE(g2(i, j).x == g2_again(i, j).x);
			REQUIRE(g2(i, j).y == g2_again(i, j).y);
		}
	for (size_t i = 0; i < g3.size_i(); ++i)
		for (size_t j = 0; j < g3.size_j(); ++j)
			for (size_t k = 0; k < g3.size_k(); ++k)
			{
				const vec3 v = g3(i, j, k);
				REQUIRE(std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z) == Approx(1.f).margin(1e-5));
				REQUIRE(v.x == g3_again(i, j, k).x);
				REQUIRE(v.z == g3_again(i, j, k).z);
			}
}

TEST_CASE("perlin: the noise at the points of the grid is the middle of the range, in 2D")
{
	// With a cell size of 1, every element is at a point of the grid, where the noise is zero
	skepu::Matrix<float> noise = noise_2d(5, 6, 1.f, 2.f, 3);
	for (size_t i = 0; i < noise.size_i(); ++i)
		for (size_t j = 0; j < noise.size_j(); ++j)
			REQUIRE(noise(i, j) == Approx(1.f).margin(1e-6));
}

TEST_CASE("perlin: the noise at the points of the grid is the middle of the range, in 3D")
{
	skepu::Tensor3<float> noise = noise_3d(3, 4, 5, 1.f, 2.f, 3);
	for (size_t i = 0; i < noise.size_i(); ++i)
		for (size_t j = 0; j < noise.size_j(); ++j)
			for (size_t k = 0; k < noise.size_k(); ++k)
				REQUIRE(noise(i, j, k) == Approx(1.f).margin(1e-6));
}

TEST_CASE("perlin: the noise is in its range, varies, and is continuous, in 2D (any shape of texture)")
{
	for (auto shape : {std::make_pair<size_t, size_t>(40, 50), std::make_pair<size_t, size_t>(97, 13), std::make_pair<size_t, size_t>(1, 30), std::make_pair<size_t, size_t>(30, 1)})
	{
		skepu::Matrix<float> noise = noise_2d(shape.first, shape.second, 8.f, 1.f, 11);
		float lowest = 1.f, highest = 0.f, step = 0.f;
		for (size_t i = 0; i < noise.size_i(); ++i)
			for (size_t j = 0; j < noise.size_j(); ++j)
			{
				lowest = std::min(lowest, noise(i, j));
				highest = std::max(highest, noise(i, j));
				if (j + 1 < noise.size_j()) step = std::max(step, std::abs(noise(i, j + 1) - noise(i, j)));
				if (i + 1 < noise.size_i()) step = std::max(step, std::abs(noise(i + 1, j) - noise(i, j)));
			}
		CHECK(lowest >= 0.f);
		CHECK(highest <= 1.f);
		if (shape.first * shape.second > 1000) CHECK(highest - lowest > 0.2f);
		CHECK(step < 0.3f); // a cell is 8 elements wide (the steps of a noise with discontinuities are close to 1)
	}
}

TEST_CASE("perlin: the noise is in its range, varies, and is continuous, in 3D (also between the frames)")
{
	skepu::Tensor3<float> noise = noise_3d(24, 30, 20, 8.f, 1.f, 11);
	float lowest = 1.f, highest = 0.f, step = 0.f;
	for (size_t i = 0; i < noise.size_i(); ++i)
		for (size_t j = 0; j < noise.size_j(); ++j)
			for (size_t k = 0; k < noise.size_k(); ++k)
			{
				lowest = std::min(lowest, noise(i, j, k));
				highest = std::max(highest, noise(i, j, k));
				if (i + 1 < noise.size_i()) step = std::max(step, std::abs(noise(i + 1, j, k) - noise(i, j, k)));
				if (j + 1 < noise.size_j()) step = std::max(step, std::abs(noise(i, j + 1, k) - noise(i, j, k)));
				if (k + 1 < noise.size_k()) step = std::max(step, std::abs(noise(i, j, k + 1) - noise(i, j, k)));
			}
	CHECK(lowest >= 0.f);
	CHECK(highest <= 1.f);
	CHECK(highest - lowest > 0.2f);
	CHECK(step < 0.3f);
}
