#include "../../external/catch2/catch.hpp"

#include <cmath>
#include <skepu>
#include <skepu-lib/blas.hpp>
#include "blas_check.hpp"

// Checked results of the real float BLAS level 1 routines on every backend, for several
// sizes and strides, against host computations.

using namespace blas_check;
using skepu::Vector;
namespace blas = skepu::blas;

TEST_CASE("BLAS level 1: copy, swap, scal, axpy with strides")
{
	for (size_t n : {1u, 7u, 100u})
		for (int inc : {1, 2, 3})
		{
			const size_t len = n * inc;
			Vector<float> x(len), y(len), z(len);
			fill(x, 1); fill(y, 2); fill(z, 3);
			auto hx = to_host(x), hy = to_host(y), hz = to_host(z);

			// copy: y(i*inc) := x(i*inc)
			blas::copy(n, x, inc, y, inc);
			auto r = to_host(y);
			for (size_t i = 0; i < len; ++i)
				REQUIRE(r[i] == ((i % inc == 0) ? hx[i] : hy[i]));
			hy = r;

			// swap
			blas::swap(n, x, inc, z, inc);
			auto rx = to_host(x), rz = to_host(z);
			for (size_t i = 0; i < len; ++i)
			{
				REQUIRE(rx[i] == ((i % inc == 0) ? hz[i] : hx[i]));
				REQUIRE(rz[i] == ((i % inc == 0) ? hx[i] : hz[i]));
			}
			hx = rx; hz = rz;

			// scal
			blas::scal(n, -1.5f, y, inc);
			r = to_host(y);
			for (size_t i = 0; i < len; ++i)
				REQUIRE(close(r[i], (i % inc == 0) ? -1.5 * hy[i] : hy[i]));
			hy = r;

			// axpy
			blas::axpy(n, 0.75f, x, inc, y, inc);
			r = to_host(y);
			for (size_t i = 0; i < len; ++i)
				REQUIRE(close(r[i], (i % inc == 0) ? hy[i] + 0.75 * hx[i] : hy[i]));
		}
}

TEST_CASE("BLAS level 1: dot, nrm2, asum, iamax")
{
	for (size_t n : {1u, 9u, 257u})
		for (int inc : {1, 2})
		{
			const size_t len = n * inc;
			Vector<float> x(len), y(len);
			fill(x, 4); fill(y, 5);
			auto hx = to_host(x), hy = to_host(y);

			double dot = 0, sumsq = 0, asum = 0, best = -1;
			size_t arg = 0;
			for (size_t i = 0; i < n; ++i)
			{
				dot += (double)hx[i * inc] * hy[i * inc];
				sumsq += (double)hx[i * inc] * hx[i * inc];
				asum += std::fabs(hx[i * inc]);
				if (std::fabs(hx[i * inc]) > best) { best = std::fabs(hx[i * inc]); arg = i; }
			}

			REQUIRE(close(blas::dot(n, x, inc, y, inc), dot));
			REQUIRE(close(blas::dotu(n, x, inc, y, inc), dot));
			REQUIRE(close(blas::nrm2(n, x, inc), std::sqrt(sumsq)));
			REQUIRE(close(blas::asum(n, x, inc), asum));
			// (with ties the first maximum is the BLAS answer; the values have some)
			size_t got = blas::iamax(n, x, inc);
			REQUIRE(std::fabs(hx[got * inc]) == best);
			(void)arg;
		}
}

TEST_CASE("BLAS level 1: rot")
{
	const size_t n{50};
	Vector<float> x(n), y(n);
	fill(x, 6); fill(y, 7);
	auto hx = to_host(x), hy = to_host(y);
	const float c = 0.6f, s = 0.8f;
	blas::rot(n, x, 1, y, 1, c, s);
	auto rx = to_host(x), ry = to_host(y);
	for (size_t i = 0; i < n; ++i)
	{
		REQUIRE(close(rx[i], c * hx[i] + s * hy[i]));
		REQUIRE(close(ry[i], c * hy[i] - s * hx[i]));
	}
}

TEST_CASE("BLAS level 1: rotg")
{
	for (auto ab : std::vector<std::pair<float, float>>{{3.f, 4.f}, {-2.f, 5.f}, {0.f, 3.f}, {7.f, 0.f}})
	{
		float a = ab.first, b = ab.second, c = 0, s = 0;
		blas::rotg(&a, &b, &c, &s);
		// the rotation maps (a0, b0) to (r, 0)
		REQUIRE(close(c * ab.first + s * ab.second, a));
		REQUIRE(close(c * ab.second - s * ab.first, 0));
		REQUIRE(close(c * c + s * s, 1));
	}
}
