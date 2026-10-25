// Each part is in its own namespace; its test cases are prefixed with its name.

// (an error of a routine is an exception, which the tests of the checks of the arguments catch)
#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"
#include "blas_check.hpp"
#include <skepu>
#include <cmath>
#include <skepu-lib/blas.hpp>


// ======== blas_float ========

namespace blas_float_part
{
// Checked results of the float BLAS routines that conjugate_gradient uses, on every backend.

static void require_close(float got, double want)
{
	REQUIRE(std::fabs(got - want) <= 1e-4 * std::max(1.0, std::fabs(want)));
}

TEST_CASE("blas_float: BLAS scal and axpy with a float and with a Scalar alpha")
{
	const size_t n{6};
	skepu::Vector<float> x(n), y(n);
	for (size_t i = 0; i < n; ++i)
		{ x[i] = i + 1; y[i] = 10; }

	// the same routines are used with both kinds of alpha (as in conjugate_gradient,
	// where alpha and beta come from dot products)
	skepu::blas::axpy(n, 2.f, x, 1, y, 1);
	auto d = skepu::blas::dot(n, x, 1, x, 1);
	auto alpha = 182.f / d; // 2
	skepu::blas::axpy(n, alpha, x, 1, y, 1);
	skepu::blas::scal(n, 0.5f, y, 1);
	auto beta = d / 91.f; // 1
	skepu::blas::scal(n, beta, y, 1);

	y.updateHost();
	for (size_t i = 0; i < n; ++i)
		require_close(y(i), (10 + 4.0 * (i + 1)) / 2);
}

TEST_CASE("blas_float: BLAS gemv, non-square, alpha and beta")
{
	const size_t m{5}, n{3};
	skepu::Matrix<float> A(m, n);
	skepu::Vector<float> x(n), y(m), xt(m), yt(n);
	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			A[i * n + j] = 1 + i * 3 + j * j;
	for (size_t j = 0; j < n; ++j) x[j] = 1 + j;
	for (size_t i = 0; i < m; ++i) { y[i] = 2 * i; xt[i] = 1 + i % 3; }
	for (size_t j = 0; j < n; ++j) yt[j] = 7 - j;

	skepu::blas::gemv(skepu::blas::Op::NoTrans, m, n, -1.5f, A, n, x, 1, 0.5f, y, 1);
	y.updateHost();
	for (size_t i = 0; i < m; ++i)
	{
		double want = 0.5 * 2 * i;
		for (size_t j = 0; j < n; ++j)
			want += -1.5 * (1 + i * 3 + j * j) * (1 + j);
		require_close(y(i), want);
	}

	skepu::blas::gemv(skepu::blas::Op::Trans, m, n, 2.f, A, n, xt, 1, 1.f, yt, 1);
	yt.updateHost();
	for (size_t j = 0; j < n; ++j)
	{
		double want = 7 - (double)j;
		for (size_t i = 0; i < m; ++i)
			want += 2.0 * (1 + i * 3 + j * j) * (1 + i % 3);
		require_close(yt(j), want);
	}
}
}


// ======== blas_level1 ========

namespace blas_level1_part
{
// Checked results of the real float BLAS level 1 routines on every backend, for several
// sizes and strides, against host computations.

using namespace blas_check;
using skepu::Vector;
namespace blas = skepu::blas;

TEST_CASE("blas_level1: BLAS level 1: copy, swap, scal, axpy with strides")
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

TEST_CASE("blas_level1: BLAS level 1: dot, nrm2, asum, iamax")
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

TEST_CASE("blas_level1: BLAS level 1: rot")
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

TEST_CASE("blas_level1: BLAS level 1: rotg")
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
}


// ======== blas_level23 ========

namespace blas_level23_part
{
// Checked results of the real float BLAS level 2 and 3 routines on every backend, for
// non-square sizes, all transpose variants and several alpha and beta, against host
// computations.

using namespace blas_check;
using skepu::Vector;
using skepu::Matrix;
namespace blas = skepu::blas;

static const blas::Op ops[] = { blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans };

TEST_CASE("blas_level23: BLAS gemv: all operations, non-square, alpha and beta")
{
	for (auto sizes : std::vector<std::pair<size_t, size_t>>{{5, 3}, {3, 5}, {1, 4}, {16, 16}})
		for (blas::Op op : ops)
			for (auto ab : std::vector<std::pair<float, float>>{{1.f, 0.f}, {-1.f, 1.f}, {2.f, 0.5f}, {0.f, 2.f}})
			{
				const size_t m = sizes.first, n = sizes.second;
				const bool t = op != blas::Op::NoTrans;
				const size_t lenx = t ? m : n, leny = t ? n : m;
				Matrix<float> A(m, n);
				Vector<float> x(lenx), y(leny);
				fill(A, 1); fill(x, 2); fill(y, 3);
				auto hA = to_host(A), hx = to_host(x), hy = to_host(y);

				blas::gemv(op, m, n, ab.first, A, n, x, 1, ab.second, y, 1);
				auto r = to_host(y);

				for (size_t i = 0; i < leny; ++i)
				{
					double want = (double)ab.second * hy[i];
					for (size_t j = 0; j < lenx; ++j)
						want += (double)ab.first * (t ? hA[j * n + i] : hA[i * n + j]) * hx[j];
					INFO("m " << m << " n " << n << " op " << (int)op << " alpha " << ab.first << " beta " << ab.second << " i " << i);
					REQUIRE(close(r[i], want));
				}
			}
}

TEST_CASE("blas_level23: BLAS gemv with strides")
{
	const size_t m{6}, n{4};
	for (int incx : {1, 2})
		for (int incy : {1, 3})
		{
			Matrix<float> A(m, n);
			Vector<float> x(n * incx), y(m * incy);
			fill(A, 4); fill(x, 5); fill(y, 6);
			auto hA = to_host(A), hx = to_host(x), hy = to_host(y);
			blas::gemv(blas::Op::NoTrans, m, n, 1.5f, A, n, x, incx, 0.5f, y, incy);
			auto r = to_host(y);
			for (size_t i = 0; i < m * incy; ++i)
			{
				double want = hy[i];
				if (i % incy == 0)
				{
					want *= 0.5;
					for (size_t j = 0; j < n; ++j)
						want += 1.5 * hA[(i / incy) * n + j] * hx[j * incx];
				}
				INFO("incx " << incx << " incy " << incy << " i " << i);
				REQUIRE(close(r[i], want));
			}
		}
}

TEST_CASE("blas_level23: BLAS ger and geru")
{
	const size_t m{5}, n{7};
	Vector<float> x(m), y(n);
	Matrix<float> A(m, n), B(m, n);
	fill(x, 1); fill(y, 2); fill(A, 3); fill(B, 3);
	auto hx = to_host(x), hy = to_host(y), hA = to_host(A);

	blas::ger(m, n, 0.5f, x, 1, y, 1, A, n);
	auto r = to_host(A);
	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			REQUIRE(close(r[i * n + j], hA[i * n + j] + 0.5 * hx[i] * hy[j]));

	blas::geru(m, n, -2.f, x, 1, y, 1, B, n);
	r = to_host(B);
	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < n; ++j)
			REQUIRE(close(r[i * n + j], hA[i * n + j] - 2.0 * hx[i] * hy[j]));
}

TEST_CASE("blas_level23: BLAS gemm: all operations, m, n and k different")
{
	const size_t m{5}, n{4}, k{3};
	for (blas::Op opA : ops)
		for (blas::Op opB : ops)
			for (auto ab : std::vector<std::pair<float, float>>{{1.f, 0.f}, {1.5f, -0.5f}, {0.f, 2.f}})
			{
				const bool ta = opA != blas::Op::NoTrans, tb = opB != blas::Op::NoTrans;
				// A is m x k (k x m transposed), B is k x n (n x k transposed)
				const size_t ar = ta ? k : m, ac = ta ? m : k, br = tb ? n : k, bc = tb ? k : n;
				Matrix<float> A(ar, ac), B(br, bc), C(m, n);
				fill(A, 1); fill(B, 2); fill(C, 3);
				auto hA = to_host(A), hB = to_host(B), hC = to_host(C);

				blas::gemm(opA, opB, m, n, k, ab.first, A, ac, B, bc, ab.second, C, n);
				auto r = to_host(C);

				for (size_t i = 0; i < m; ++i)
					for (size_t j = 0; j < n; ++j)
					{
						double sum = 0;
						for (size_t l = 0; l < k; ++l)
							sum += (double)(ta ? hA[l * ac + i] : hA[i * ac + l]) * (tb ? hB[j * bc + l] : hB[l * bc + j]);
						double want = ab.first * sum + ab.second * hC[i * n + j];
						INFO("opA " << (int)opA << " opB " << (int)opB << " alpha " << ab.first << " beta " << ab.second << " (" << i << "," << j << ")");
						REQUIRE(close(r[i * n + j], want));
					}
			}
}
}


// ======== blas_semantics ========

namespace blas_semantics_part
{
// The sizes, increments and leading dimensions of the routines, as in the reference BLAS: n elements of longer vectors, negative
// increments, zero sizes, the leading part of a matrix, and the values that are not read (a not-a-number where alpha or beta is zero).

using namespace blas_check;
using skepu::Vector;
using skepu::Matrix;
namespace blas = skepu::blas;

// the place in the Vector of element i of a BLAS vector of n elements and the increment inc
static size_t pos(size_t i, size_t n, int inc)
{
	return (inc > 0) ? i * inc : (n - 1 - i) * (size_t)(-inc);
}

static size_t length(size_t n, int inc, size_t extra)
{
	return 1 + (n - 1) * (size_t)std::abs(inc) + extra;
}

template<typename F>
static void for_vectors(F f)
{
	for (size_t n : {1u, 5u, 17u})
		for (int incx : {1, 2, -1, -3})
			for (int incy : {1, 3, -2})
				for (size_t extra : {0u, 4u})
					f(n, incx, incy, extra);
}

TEST_CASE("blas_semantics: BLAS level 1: n elements of longer vectors, negative increments")
{
	for_vectors([](size_t n, int incx, int incy, size_t extra)
	{
		INFO("n " << n << " incx " << incx << " incy " << incy << " extra " << extra);
		const size_t lx = length(n, incx, extra), ly = length(n, incy, extra + 1);
		Vector<float> x(lx), y(ly);
		fill(x, 1); fill(y, 2);
		const auto hx = to_host(x), hy = to_host(y);

		// copy: y(i) := x(i) for the n elements, the others are not touched
		blas::copy(n, x, incx, y, incy);
		auto r = to_host(y);
		std::vector<float> want = hy;
		for (size_t i = 0; i < n; ++i)
			want[pos(i, n, incy)] = hx[pos(i, n, incx)];
		REQUIRE(r == want);

		// axpy
		fill(y, 2);
		blas::axpy(n, 0.5f, x, incx, y, incy);
		r = to_host(y);
		want = hy;
		for (size_t i = 0; i < n; ++i)
			want[pos(i, n, incy)] += 0.5f * hx[pos(i, n, incx)];
		for (size_t i = 0; i < ly; ++i)
			REQUIRE(close(r[i], want[i]));

		// swap
		fill(y, 2);
		blas::swap(n, x, incx, y, incy);
		r = to_host(y);
		auto rx = to_host(x);
		want = hy;
		std::vector<float> wantx = hx;
		for (size_t i = 0; i < n; ++i)
		{
			want[pos(i, n, incy)] = hx[pos(i, n, incx)];
			wantx[pos(i, n, incx)] = hy[pos(i, n, incy)];
		}
		REQUIRE(r == want);
		REQUIRE(rx == wantx);

		// rot
		fill(x, 1); fill(y, 2);
		blas::rot(n, x, incx, y, incy, 0.6f, 0.8f);
		r = to_host(y);
		rx = to_host(x);
		want = hy; wantx = hx;
		for (size_t i = 0; i < n; ++i)
		{
			const float a = hx[pos(i, n, incx)], b = hy[pos(i, n, incy)];
			wantx[pos(i, n, incx)] = 0.6f * a + 0.8f * b;
			want[pos(i, n, incy)] = 0.6f * b - 0.8f * a;
		}
		for (size_t i = 0; i < lx; ++i)
			REQUIRE(close(rx[i], wantx[i]));
		for (size_t i = 0; i < ly; ++i)
			REQUIRE(close(r[i], want[i]));

		// dot
		fill(x, 1); fill(y, 2);
		double dot = 0;
		for (size_t i = 0; i < n; ++i)
			dot += (double)hx[pos(i, n, incx)] * hy[pos(i, n, incy)];
		REQUIRE(close(blas::dot(n, x, incx, y, incy), dot));
		REQUIRE(close(blas::dotu(n, x, incx, y, incy), dot));
	});
}

TEST_CASE("blas_semantics: BLAS level 1: scal, nrm2, asum and iamax on n elements, and negative increments")
{
	for (size_t n : {1u, 6u, 40u})
		for (int inc : {1, 2, 3})
			for (size_t extra : {0u, 5u})
			{
				INFO("n " << n << " inc " << inc << " extra " << extra);
				const size_t len = length(n, inc, extra);
				Vector<float> x(len);
				fill(x, 4);
				const auto hx = to_host(x);

				double sumsq = 0, asum = 0, best = -1;
				size_t arg = 0;
				for (size_t i = 0; i < n; ++i)
				{
					const double v = hx[i * inc];
					sumsq += v * v;
					asum += std::fabs(v);
					if (std::fabs(v) > best) { best = std::fabs(v); arg = i; }
				}
				REQUIRE(close(blas::nrm2(n, x, inc), std::sqrt(sumsq)));
				REQUIRE(close(blas::asum(n, x, inc), asum));
				REQUIRE(blas::iamax(n, x, inc) == arg); // (the first of equal values)

				blas::scal(n, 2.f, x, inc);
				const auto r = to_host(x);
				for (size_t i = 0; i < len; ++i)
					REQUIRE(close(r[i], (i % inc == 0 && i / inc < n) ? 2.0 * hx[i] : hx[i]));

				// the reference BLAS does nothing for a negative increment in these routines
				Vector<float> y(len);
				fill(y, 4);
				blas::scal(n, 2.f, y, -inc);
				REQUIRE(to_host(y) == hx);
				REQUIRE((float)blas::nrm2(n, y, -inc) == 0.f);
				REQUIRE((float)blas::asum(n, y, -inc) == 0.f);
				REQUIRE(blas::iamax(n, y, -inc) == 0);
			}
}

TEST_CASE("blas_semantics: BLAS level 1: zero elements")
{
	Vector<float> x(4), y(4);
	fill(x, 1); fill(y, 2);
	const auto hx = to_host(x), hy = to_host(y);
	blas::copy(0, x, 1, y, 1);
	blas::axpy(0, 2.f, x, 1, y, 1);
	blas::swap(0, x, 1, y, 1);
	blas::scal(0, 2.f, x, 1);
	blas::rot(0, x, 1, y, 1, 0.f, 1.f);
	REQUIRE(to_host(x) == hx);
	REQUIRE(to_host(y) == hy);
	REQUIRE((float)blas::dot(0, x, 1, y, 1) == 0.f);
	REQUIRE((float)blas::nrm2(0, x, 1) == 0.f);
	REQUIRE((float)blas::asum(0, x, 1) == 0.f);
	REQUIRE(blas::iamax(0, x, 1) == 0);
}

TEST_CASE("blas_semantics: BLAS level 1: the sizes of the vectors are checked")
{
	Vector<float> x(10), y(10);
	fill(x, 1); fill(y, 2);
	REQUIRE_THROWS(blas::copy(6, x, 2, y, 1));   // needs 11 elements
	REQUIRE_THROWS(blas::dot(11, x, 1, y, 1));
	REQUIRE_THROWS(blas::axpy(5, 1.f, x, 0, y, 1)); // a zero increment
	REQUIRE_NOTHROW(blas::copy(5, x, 2, y, -2));   // needs 9
}

TEST_CASE("blas_semantics: BLAS level 1: iamax finds the first of equal values")
{
	for (size_t n : {2u, 9u, 100u, 1000u})
	{
		Vector<float> x(n);
		for (size_t i = 0; i < n; ++i)
			x[i] = (i % 3 == 1) ? -5.f : 1.f;
		REQUIRE(blas::iamax(n, x, 1) == 1);
	}
}

TEST_CASE("blas_semantics: BLAS level 1: nrm2 of values whose squares are out of range")
{
	const size_t n = 100;
	Vector<float> x(n);
	for (float v : {1e30f, 3e19f, 1e-30f, 1e-22f, 1.f})
	{
		for (size_t i = 0; i < n; ++i)
			x[i] = (i % 2) ? v : -v;
		INFO("v " << v);
		REQUIRE(close(blas::nrm2(n, x, 1) / v, 10.0, 1e-5));
	}

	for (size_t i = 0; i < n; ++i)
		x[i] = 0.f;
	REQUIRE((float)blas::nrm2(n, x, 1) == 0.f);

	x[7] = INFINITY;
	REQUIRE(std::isinf((float)blas::nrm2(n, x, 1)));
	x[7] = NAN;
	REQUIRE(std::isnan((float)blas::nrm2(n, x, 1)));
}

TEST_CASE("blas_semantics: BLAS gemv: the leading part of a matrix, increments, values that are not read")
{
	const size_t M{9}, N{7};
	for (auto sizes : std::vector<std::pair<size_t, size_t>>{{9, 7}, {5, 3}, {1, 7}, {9, 1}, {4, 4}})
		for (blas::Op op : {blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans})
			for (int incx : {1, 2, -1, -2})
				for (int incy : {1, -3})
				{
					const size_t m = sizes.first, n = sizes.second;
					const bool t = op != blas::Op::NoTrans;
					const size_t lenx = t ? m : n, leny = t ? n : m;
					INFO("m " << m << " n " << n << " op " << (int)op << " incx " << incx << " incy " << incy);
					Matrix<float> A(M, N);
					Vector<float> x(length(lenx, incx, 0)), y(length(leny, incy, 2));
					fill(A, 1); fill(x, 2); fill(y, 3);
					const auto hA = to_host(A), hx = to_host(x), hy = to_host(y);

					blas::gemv(op, m, n, 1.5f, A, N, x, incx, -0.5f, y, incy);
					const auto r = to_host(y);

					std::vector<float> want = hy;
					for (size_t i = 0; i < leny; ++i)
					{
						double sum = 0;
						for (size_t j = 0; j < lenx; ++j)
							sum += (double)(t ? hA[j * N + i] : hA[i * N + j]) * hx[pos(j, lenx, incx)];
						want[pos(i, leny, incy)] = 1.5 * sum - 0.5 * hy[pos(i, leny, incy)];
					}
					for (size_t i = 0; i < r.size(); ++i)
						REQUIRE(close(r[i], want[i]));
				}

	// a not-a-number in y does not go to the result if beta is zero, and one in x does not if alpha is zero
	Matrix<float> A(4, 4);
	Vector<float> x(4), y(4);
	fill(A, 1); fill(x, 2);
	for (size_t i = 0; i < 4; ++i)
		y[i] = NAN;
	blas::gemv(blas::Op::NoTrans, 4, 4, 1.f, A, 4, x, 1, 0.f, y, 1);
	for (float v : to_host(y))
		REQUIRE(std::isfinite(v));
	x[2] = NAN;
	fill(y, 3);
	const auto hy = to_host(y);
	blas::gemv(blas::Op::NoTrans, 4, 4, 0.f, A, 4, x, 1, 2.f, y, 1);
	const auto r = to_host(y);
	for (size_t i = 0; i < 4; ++i)
		REQUIRE(close(r[i], 2.0 * hy[i]));
}

TEST_CASE("blas_semantics: BLAS ger and geru: the leading part of a matrix, increments")
{
	const size_t M{8}, N{6};
	for (auto sizes : std::vector<std::pair<size_t, size_t>>{{8, 6}, {3, 2}, {1, 6}, {8, 1}})
		for (int incx : {1, -2})
			for (int incy : {1, 3, -1})
			{
				const size_t m = sizes.first, n = sizes.second;
				INFO("m " << m << " n " << n << " incx " << incx << " incy " << incy);
				Vector<float> x(length(m, incx, 1)), y(length(n, incy, 0));
				Matrix<float> A(M, N), B(M, N);
				fill(x, 1); fill(y, 2); fill(A, 3); fill(B, 3);
				const auto hx = to_host(x), hy = to_host(y), hA = to_host(A);

				blas::ger(m, n, 0.5f, x, incx, y, incy, A, N);
				blas::geru(m, n, -2.f, x, incx, y, incy, B, N);
				const auto ra = to_host(A), rb = to_host(B);
				for (size_t i = 0; i < M; ++i)
					for (size_t j = 0; j < N; ++j)
					{
						const bool in = i < m && j < n;
						const double xy = in ? (double)hx[pos(i, m, incx)] * hy[pos(j, n, incy)] : 0;
						REQUIRE(close(ra[i * N + j], hA[i * N + j] + 0.5 * xy));
						REQUIRE(close(rb[i * N + j], hA[i * N + j] - 2.0 * xy));
					}
			}
}

TEST_CASE("blas_semantics: BLAS gemm: the leading part of the matrices, values that are not read")
{
	const size_t AR{9}, AC{8}, BR{8}, BC{9}, CR{7}, CC{6};
	const size_t m{5}, n{4}, k{3};
	for (blas::Op opA : {blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans})
		for (blas::Op opB : {blas::Op::NoTrans, blas::Op::Trans})
			for (auto ab : std::vector<std::pair<float, float>>{{1.f, 0.f}, {1.5f, -0.5f}, {0.f, 2.f}})
			{
				const bool ta = opA != blas::Op::NoTrans, tb = opB != blas::Op::NoTrans;
				INFO("opA " << (int)opA << " opB " << (int)opB << " alpha " << ab.first << " beta " << ab.second);
				Matrix<float> A(AR, AC), B(BR, BC), C(CR, CC);
				fill(A, 1); fill(B, 2); fill(C, 3);
				const auto hA = to_host(A), hB = to_host(B), hC = to_host(C);

				blas::gemm(opA, opB, m, n, k, ab.first, A, AC, B, BC, ab.second, C, CC);
				const auto r = to_host(C);
				for (size_t i = 0; i < CR; ++i)
					for (size_t j = 0; j < CC; ++j)
					{
						double want = hC[i * CC + j];
						if (i < m && j < n)
						{
							double sum = 0;
							for (size_t l = 0; l < k; ++l)
								sum += (double)(ta ? hA[l * AC + i] : hA[i * AC + l]) * (tb ? hB[j * BC + l] : hB[l * BC + j]);
							want = ab.first * sum + ab.second * hC[i * CC + j];
						}
						REQUIRE(close(r[i * CC + j], want));
					}
			}

	// k = 0 scales C by beta, and a not-a-number in C does not go to the result if beta is zero
	Matrix<float> A(3, 3), B(3, 3), C(3, 3);
	fill(A, 1); fill(B, 2); fill(C, 3);
	const auto hC = to_host(C);
	blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, 3, 3, 0, 1.f, A, 3, B, 3, 2.f, C, 3);
	const auto r = to_host(C);
	for (size_t i = 0; i < 9; ++i)
		REQUIRE(close(r[i], 2.0 * hC[i]));
	for (size_t i = 0; i < 9; ++i)
		C[i] = NAN;
	blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, 3, 3, 3, 1.f, A, 3, B, 3, 0.f, C, 3);
	for (float v : to_host(C))
		REQUIRE(std::isfinite(v));
}

TEST_CASE("blas_semantics: BLAS matrices: the sizes and the leading dimensions are checked")
{
	Matrix<float> A(4, 5), B(5, 3), C(4, 3);
	Vector<float> x(5), y(4);
	fill(A, 1); fill(B, 2); fill(C, 3); fill(x, 1); fill(y, 2);
	REQUIRE_NOTHROW(blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, 4, 3, 5, 1.f, A, 5, B, 3, 0.f, C, 3));
	REQUIRE_THROWS(blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, 4, 3, 5, 1.f, A, 6, B, 3, 0.f, C, 3)); // lda is not the columns
	REQUIRE_THROWS(blas::gemm(blas::Op::NoTrans, blas::Op::NoTrans, 4, 3, 6, 1.f, A, 5, B, 3, 0.f, C, 3)); // k is too large
	REQUIRE_THROWS(blas::gemv(blas::Op::NoTrans, 5, 5, 1.f, A, 5, x, 1, 0.f, y, 1));                      // m is too large
}
}


// ======== blas_phase1 ========

namespace blas_phase1_part
{
// The routines of the second part of the library: iamin, axpby, rotmg and rotm, and the routines of the matrices that are stored in a
// triangle (symmetric, Hermitian and triangular), against host references. The Hermitian routines are checked with real
// matrices here (they are the symmetric ones), and with complex matrices in blas_complex.

using namespace blas_check;
using skepu::Vector;
using skepu::Matrix;
namespace blas = skepu::blas;
using blas::Uplo;
using blas::Op;
using blas::Diag;
using blas::Side;

static const Op ops[] = { Op::NoTrans, Op::Trans, Op::ConjTrans };
static const Uplo uplos[] = { Uplo::Upper, Uplo::Lower };

static std::vector<double> dbl(const std::vector<float> &v) { return ref::convert<double>(v); }

// a matrix for a triangular solve: a large diagonal and small other elements, so that the system is well conditioned
static void fill_triangular(Matrix<float> &A, size_t salt)
{
	fill(A, salt);
	for (size_t i = 0; i < A.size_i(); ++i)
		for (size_t j = 0; j < A.size_j(); ++j)
			A[i * A.size_j() + j] = (i == j) ? 4.f + 0.1f * (float)(i % 3) : 0.1f * A[i * A.size_j() + j];
}

static void require_vectors(const std::vector<float> &got, const std::vector<double> &want, double tol = 1e-3)
{
	REQUIRE(got.size() == want.size());
	for (size_t i = 0; i < got.size(); ++i)
	{
		INFO("element " << i);
		REQUIRE(close(got[i], want[i], tol));
	}
}

TEST_CASE("blas_phase1: iamin and axpby")
{
	for (size_t n : {5u, 9u, 100u, 1000u})
	{
		Vector<float> x(n);
		for (size_t i = 0; i < n; ++i)
			x[i] = (i % 4 == 2) ? -0.5f : 3.f + (float)(i % 5);
		REQUIRE(blas::iamin(n, x, 1) == 2);
		REQUIRE(blas::iamax(n, x, 1) == 4); // (the first of the largest: 7 at i = 4, 9, ...)
	}
	Vector<float> none(3);
	REQUIRE(blas::iamin(0, none, 1) == 0);

	for (int incx : {1, 2, -1, -3})
		for (int incy : {1, -2})
			for (float beta : {0.f, 1.f, -0.5f})
			{
				const size_t n = 6;
				auto len = [](size_t n, int inc) { return 1 + (n - 1) * (size_t)std::abs(inc); };
				auto pos = [](size_t i, size_t n, int inc) { return (inc > 0) ? i * inc : (n - 1 - i) * (size_t)(-inc); };
				Vector<float> x(len(n, incx)), y(len(n, incy));
				fill(x, 1); fill(y, 2);
				const auto hx = to_host(x), hy = to_host(y);
				if (beta == 0.f)
					for (size_t i = 0; i < n; ++i)
						y[pos(i, n, incy)] = NAN; // (not read)
				blas::axpby(n, 2.f, x, incx, beta, y, incy);
				const auto r = to_host(y);
				for (size_t i = 0; i < r.size(); ++i)
				{
					double want = hy[i];
					for (size_t k = 0; k < n; ++k)
						if (pos(k, n, incy) == i)
							want = 2.0 * hx[pos(k, n, incx)] + beta * hy[i];
					INFO("incx " << incx << " incy " << incy << " beta " << beta << " i " << i);
					REQUIRE(close(r[i], want));
				}
			}
}

TEST_CASE("blas_phase1: rotmg and rotm")
{
	struct Case { float d1, d2, x1, y1; };
	// each kind of rotation: d2 * y1 == 0 (the identity), |d1 x1^2| > |d2 y1^2|, the other way, d1 < 0, and values that are rescaled
	const std::vector<Case> cases = {
		{1.f, 1.f, 3.f, 0.f}, {1.f, 2.f, 4.f, 1.f}, {2.f, 1.f, 1.f, 5.f}, {-1.f, 1.f, 1.f, 1.f},
		{1e10f, 1.f, 2.f, 1.f}, {1.f, 1e-10f, 1.f, 3.f}, {3e-9f, 5.f, 1.f, 2.f}, {1.f, 2e9f, 0.5f, 3.f}};
	for (const Case &c : cases)
	{
		INFO("d1 " << c.d1 << " d2 " << c.d2 << " x1 " << c.x1 << " y1 " << c.y1);
		float d1 = c.d1, d2 = c.d2, x1 = c.x1;
		float param[5] = {0, 0, 0, 0, 0};
		blas::rotmg(&d1, &d2, &x1, c.y1, param);

		Vector<float> x(1), y(1);
		x[0] = c.x1; y[0] = c.y1;
		blas::rotm(1, x, 1, y, 1, param);
		x.updateHost(); y.updateHost();

		if (param[0] == -2)
		{
			REQUIRE(x(0) == c.x1);
			REQUIRE(y(0) == c.y1);
		}
		else if (c.d1 < 0)
		{
			REQUIRE(param[0] == -1);
			REQUIRE(d1 == 0); REQUIRE(d2 == 0); REQUIRE(x1 == 0);
		}
		else
		{
			// the rotation zeroes y1, makes x1 the new x1, and keeps the weighted sum of squares
			REQUIRE(std::fabs(y(0)) <= 1e-4 * std::max(1.f, std::fabs(c.y1)));
			REQUIRE(close(x(0), x1, 1e-4));
			const double before = (double)c.d1 * c.x1 * c.x1 + (double)c.d2 * c.y1 * c.y1;
			const double after = (double)d1 * x1 * x1;
			REQUIRE(close(after, before, 1e-3));
		}
	}

	// rotm on vectors with a stride, for the flags
	for (float flag : {-1.f, 0.f, 1.f})
	{
		const size_t n = 5;
		Vector<float> x(2 * n), y(n);
		fill(x, 1); fill(y, 2);
		const auto hx = to_host(x), hy = to_host(y);
		const float param[5] = {flag, 0.5f, -0.25f, 0.75f, 1.5f};
		blas::rotm(n, x, 2, y, 1, param);
		const auto rx = to_host(x), ry = to_host(y);
		const double h11 = (flag == 0.f) ? 1 : param[1], h22 = (flag == 0.f) ? 1 : param[4];
		const double h12 = (flag == 1.f) ? 1 : param[3], h21 = (flag == 1.f) ? -1 : param[2];
		for (size_t i = 0; i < n; ++i)
		{
			REQUIRE(close(rx[2 * i], h11 * hx[2 * i] + h12 * hy[i]));
			REQUIRE(close(ry[i], h21 * hx[2 * i] + h22 * hy[i]));
			if (i + 1 < n)
				REQUIRE(rx[2 * i + 1] == hx[2 * i + 1]);
		}
	}
}

TEST_CASE("blas_phase1: symv, hemv and trmv")
{
	for (size_t n : {1u, 5u, 8u})
		for (Uplo uplo : uplos)
			for (int incx : {1, -2})
				for (int incy : {1, 3})
				{
					INFO("n " << n << " uplo " << (int)uplo << " incx " << incx << " incy " << incy);
					const size_t lx = 1 + (n - 1) * std::abs(incx), ly = 1 + (n - 1) * std::abs(incy);
					Matrix<float> A(n, n);
					Vector<float> x(lx), y(ly), y2(ly);
					fill(A, 1); fill(x, 2); fill(y, 3);
					for (size_t i = 0; i < ly; ++i)
						y2[i] = y(i);
					const auto hA = dbl(to_host(A)), hx = dbl(to_host(x)), hy = dbl(to_host(y));
					auto pos = [](size_t i, size_t n, int inc) { return (inc > 0) ? i * inc : (n - 1 - i) * (size_t)(-inc); };

					const auto M = ref::triangle(hA, n, n, 0, uplo == Uplo::Upper, false);
					std::vector<double> xl(n);
					for (size_t i = 0; i < n; ++i)
						xl[i] = hx[pos(i, n, incx)];
					const auto Mx = ref::mul(M, n, n, xl, 1);

					blas::symv(uplo, n, 1.5f, A, n, x, incx, -0.5f, y, incy);
					blas::hemv(uplo, n, 1.5f, A, n, x, incx, -0.5f, y2, incy);
					const auto ry = to_host(y), ry2 = to_host(y2);
					std::vector<double> want = hy;
					for (size_t i = 0; i < n; ++i)
						want[pos(i, n, incy)] = 1.5 * Mx[i] - 0.5 * hy[pos(i, n, incy)];
					require_vectors(ry, want);
					require_vectors(ry2, want);

					// beta zero: the old y is not read
					for (size_t i = 0; i < ly; ++i)
						y[i] = NAN;
					blas::symv(uplo, n, 1.f, A, n, x, incx, 0.f, y, incy);
					const auto rz = to_host(y);
					for (size_t i = 0; i < n; ++i)
						REQUIRE(close(rz[pos(i, n, incy)], Mx[i]));

					for (Op op : ops)
						for (Diag diag : {Diag::NonUnit, Diag::Unit})
						{
							Vector<float> t(lx);
							fill(t, 2);
							blas::trmv(uplo, op, diag, n, A, n, t, incx);
							const auto T = ref::opm(ref::triangle(hA, n, n, 2, uplo == Uplo::Upper, diag == Diag::Unit), n, n, (int)(op != Op::NoTrans));
							const auto Tx = ref::mul(T, n, n, xl, 1);
							const auto rt = to_host(t);
							std::vector<double> wt = hx;
							for (size_t i = 0; i < n; ++i)
								wt[pos(i, n, incx)] = Tx[i];
							require_vectors(rt, wt);
						}
				}
}

TEST_CASE("blas_phase1: trsv")
{
	for (size_t n : {1u, 4u, 9u})
		for (Uplo uplo : uplos)
			for (Op op : ops)
				for (Diag diag : {Diag::NonUnit, Diag::Unit})
					for (int incx : {1, -2})
					{
						INFO("n " << n << " uplo " << (int)uplo << " op " << (int)op << " diag " << (int)diag << " incx " << incx);
						Matrix<float> A(n, n);
						fill_triangular(A, 1);
						Vector<float> b(1 + (n - 1) * std::abs(incx)), x(b.size());
						fill(b, 2);
						for (size_t i = 0; i < b.size(); ++i)
							x[i] = b(i);
						const auto hA = dbl(to_host(A)), hb = dbl(to_host(b));
						auto pos = [](size_t i, size_t n, int inc) { return (inc > 0) ? i * inc : (n - 1 - i) * (size_t)(-inc); };

						blas::trsv(uplo, op, diag, n, A, n, x, incx);
						const auto hx = dbl(to_host(x));

						// op(T) x = b, and the elements between the increments are as they were
						const auto T = ref::opm(ref::triangle(hA, n, n, 2, uplo == Uplo::Upper, diag == Diag::Unit), n, n, (int)(op != Op::NoTrans));
						std::vector<double> xl(n);
						for (size_t i = 0; i < n; ++i)
							xl[i] = hx[pos(i, n, incx)];
						const auto Tx = ref::mul(T, n, n, xl, 1);
						for (size_t i = 0; i < n; ++i)
							REQUIRE(close(Tx[i], hb[pos(i, n, incx)], 1e-3));
						for (size_t i = 0; i < b.size(); ++i)
							if (i % std::abs(incx) != 0)
								REQUIRE(hx[i] == hb[i]);
					}
}

TEST_CASE("blas_phase1: syr, her, syr2 and her2")
{
	for (size_t n : {1u, 5u})
		for (Uplo uplo : uplos)
			for (int incx : {1, -2})
				for (int incy : {1, 3})
				{
					INFO("n " << n << " uplo " << (int)uplo << " incx " << incx << " incy " << incy);
					const size_t M = n + 2;
					Vector<float> x(1 + (n - 1) * std::abs(incx)), y(1 + (n - 1) * std::abs(incy));
					Matrix<float> A1(M, M), A2(M, M), A3(M, M), A4(M, M);
					fill(x, 1); fill(y, 2); fill(A1, 3); fill(A2, 3); fill(A3, 3); fill(A4, 3);
					const auto hx = dbl(to_host(x)), hy = dbl(to_host(y)), hA = dbl(to_host(A1));
					auto pos = [](size_t i, size_t n, int inc) { return (inc > 0) ? i * inc : (n - 1 - i) * (size_t)(-inc); };

					blas::syr(uplo, n, 0.5f, x, incx, A1, M);
					blas::her(uplo, n, 0.5f, x, incx, A2, M);
					blas::syr2(uplo, n, 0.5f, x, incx, y, incy, A3, M);
					blas::her2(uplo, n, 0.5f, x, incx, y, incy, A4, M);
					const auto r1 = dbl(to_host(A1)), r2 = dbl(to_host(A2)), r3 = dbl(to_host(A3)), r4 = dbl(to_host(A4));
					for (size_t i = 0; i < M; ++i)
						for (size_t j = 0; j < M; ++j)
						{
							const bool in = i < n && j < n && (uplo == Uplo::Upper ? i <= j : i >= j);
							double a1 = hA[i * M + j], a2 = a1;
							if (in)
							{
								const double xi = hx[pos(i, n, incx)], xj = hx[pos(j, n, incx)];
								const double yi = hy[pos(i, n, incy)], yj = hy[pos(j, n, incy)];
								a1 += 0.5 * xi * xj;
								a2 += 0.5 * (xi * yj + yi * xj);
							}
							REQUIRE(close(r1[i * M + j], a1));
							REQUIRE(close(r2[i * M + j], a1));
							REQUIRE(close(r3[i * M + j], a2));
							REQUIRE(close(r4[i * M + j], a2));
						}
				}
}

TEST_CASE("blas_phase1: symm, hemm, syrk, herk, syr2k and her2k")
{
	const size_t m{5}, n{4};
	for (Side side : {Side::Left, Side::Right})
		for (Uplo uplo : uplos)
		{
			INFO("side " << (int)side << " uplo " << (int)uplo);
			const size_t k = (side == Side::Left) ? m : n;
			Matrix<float> A(k + 1, k + 2), B(m + 1, n + 1), C(m + 2, n + 1), C2(m + 2, n + 1);
			fill(A, 1); fill(B, 2); fill(C, 3); fill(C2, 3);
			const auto hA = dbl(to_host(A)), hB = dbl(to_host(B)), hC = dbl(to_host(C));
			const auto M = ref::triangle(hA, A.size_j(), k, 0, uplo == Uplo::Upper, false);
			std::vector<double> Bm(m * n);
			for (size_t i = 0; i < m; ++i)
				for (size_t j = 0; j < n; ++j)
					Bm[i * n + j] = hB[i * B.size_j() + j];
			const auto P = (side == Side::Left) ? ref::mul(M, m, m, Bm, n) : ref::mul(Bm, m, n, M, n);

			blas::symm(side, uplo, m, n, 1.5f, A, A.size_j(), B, B.size_j(), -0.5f, C, C.size_j());
			blas::hemm(side, uplo, m, n, 1.5f, A, A.size_j(), B, B.size_j(), -0.5f, C2, C2.size_j());
			const auto rc = dbl(to_host(C)), rc2 = dbl(to_host(C2));
			for (size_t i = 0; i < C.size_i(); ++i)
				for (size_t j = 0; j < C.size_j(); ++j)
				{
					const double c0 = hC[i * C.size_j() + j];
					const double want = (i < m && j < n) ? 1.5 * P[i * n + j] - 0.5 * c0 : c0;
					REQUIRE(close(rc[i * C.size_j() + j], want));
					REQUIRE(close(rc2[i * C.size_j() + j], want));
				}
		}

	const size_t kk{3};
	for (Uplo uplo : uplos)
		for (Op op : {Op::NoTrans, Op::Trans, Op::ConjTrans})
		{
			INFO("uplo " << (int)uplo << " op " << (int)op);
			const bool t = (op != Op::NoTrans);
			const size_t ar = t ? kk + 1 : n + 1, ac = t ? n + 1 : kk + 1;
			Matrix<float> A(ar, ac), B(ar, ac), C1(n + 1, n + 1), C2(n + 1, n + 1), C3(n + 1, n + 1), C4(n + 1, n + 1);
			fill(A, 1); fill(B, 2); fill(C1, 3); fill(C2, 3); fill(C3, 3); fill(C4, 3);
			const auto hA = dbl(to_host(A)), hB = dbl(to_host(B)), hC = dbl(to_host(C1));
			// the n x k parts of op(A) and op(B)
			auto part = [&](const std::vector<double> &H) {
				std::vector<double> P(n * kk);
				for (size_t i = 0; i < n; ++i)
					for (size_t l = 0; l < kk; ++l)
						P[i * kk + l] = t ? H[l * ac + i] : H[i * ac + l];
				return P;
			};
			const auto PA = part(hA), PB = part(hB);
			const auto AAt = ref::mul(PA, n, kk, ref::opm(PA, n, kk, 1), n);
			const auto ABt = ref::mul(PA, n, kk, ref::opm(PB, n, kk, 1), n);
			const auto BAt = ref::mul(PB, n, kk, ref::opm(PA, n, kk, 1), n);

			blas::syrk(uplo, op, n, kk, 0.5f, A, ac, 2.f, C1, n + 1);
			blas::syr2k(uplo, op, n, kk, 0.5f, A, ac, B, ac, 2.f, C2, n + 1);
			if (op != Op::Trans)
			{
				blas::herk(uplo, op, n, kk, 0.5f, A, ac, 2.f, C3, n + 1);
				blas::her2k(uplo, op, n, kk, 0.5f, A, ac, B, ac, 2.f, C4, n + 1);
			}
			const auto r1 = dbl(to_host(C1)), r2 = dbl(to_host(C2)), r3 = dbl(to_host(C3)), r4 = dbl(to_host(C4));
			for (size_t i = 0; i < n + 1; ++i)
				for (size_t j = 0; j < n + 1; ++j)
				{
					const double c0 = hC[i * (n + 1) + j];
					const bool in = i < n && j < n && (uplo == Uplo::Upper ? i <= j : i >= j);
					const double w1 = in ? 0.5 * AAt[i * n + j] + 2 * c0 : c0;
					const double w2 = in ? 0.5 * (ABt[i * n + j] + BAt[i * n + j]) + 2 * c0 : c0;
					REQUIRE(close(r1[i * (n + 1) + j], w1));
					REQUIRE(close(r2[i * (n + 1) + j], w2));
					if (op != Op::Trans)
					{
						REQUIRE(close(r3[i * (n + 1) + j], w1));
						REQUIRE(close(r4[i * (n + 1) + j], w2));
					}
				}
		}
}

TEST_CASE("blas_phase1: trmm and trsm")
{
	const size_t m{5}, n{4};
	for (Side side : {Side::Left, Side::Right})
		for (Uplo uplo : uplos)
			for (Op op : ops)
				for (Diag diag : {Diag::NonUnit, Diag::Unit})
				{
					INFO("side " << (int)side << " uplo " << (int)uplo << " op " << (int)op << " diag " << (int)diag);
					const size_t k = (side == Side::Left) ? m : n;
					Matrix<float> A(k + 1, k + 1), B(m + 2, n + 1), B1(m + 2, n + 1);
					fill_triangular(A, 1);
					fill(B, 2); fill(B1, 2);
					const auto hA = dbl(to_host(A)), hB = dbl(to_host(B));
					const auto T = ref::opm(ref::triangle(hA, A.size_j(), k, 2, uplo == Uplo::Upper, diag == Diag::Unit), k, k, (int)(op != Op::NoTrans));
					std::vector<double> Bm(m * n);
					for (size_t i = 0; i < m; ++i)
						for (size_t j = 0; j < n; ++j)
							Bm[i * n + j] = hB[i * B.size_j() + j];
					const auto P = (side == Side::Left) ? ref::mul(T, m, m, Bm, n) : ref::mul(Bm, m, n, T, n);

					blas::trmm(side, uplo, op, diag, m, n, 2.f, A, A.size_j(), B, B.size_j());
					const auto rb = dbl(to_host(B));
					for (size_t i = 0; i < B.size_i(); ++i)
						for (size_t j = 0; j < B.size_j(); ++j)
							REQUIRE(close(rb[i * B.size_j() + j], (i < m && j < n) ? 2 * P[i * n + j] : hB[i * B.size_j() + j], 1e-3));

					// the solution X of op(T) X = alpha B (or X op(T) = alpha B) is such that the product is alpha B
					blas::trsm(side, uplo, op, diag, m, n, 0.5f, A, A.size_j(), B1, B1.size_j());
					const auto rx = dbl(to_host(B1));
					std::vector<double> X(m * n);
					for (size_t i = 0; i < m; ++i)
						for (size_t j = 0; j < n; ++j)
							X[i * n + j] = rx[i * B1.size_j() + j];
					const auto Q = (side == Side::Left) ? ref::mul(T, m, m, X, n) : ref::mul(X, m, n, T, n);
					// (the part of B outside the m x n part is as it was)
					for (size_t i = 0; i < B1.size_i(); ++i)
						for (size_t j = 0; j < B1.size_j(); ++j)
							if (i >= m || j >= n)
								REQUIRE(rx[i * B1.size_j() + j] == hB[i * B1.size_j() + j]);
					for (size_t i = 0; i < m; ++i)
						for (size_t j = 0; j < n; ++j)
							REQUIRE(close(Q[i * n + j], 0.5 * Bm[i * n + j], 1e-3));
				}
}

TEST_CASE("blas_phase1: the arguments of the triangle routines are checked")
{
	Matrix<float> A(4, 4), B(4, 4);
	Vector<float> x(4), y(4);
	fill(A, 1); fill(B, 2); fill(x, 1); fill(y, 2);
	REQUIRE_THROWS(blas::symv(Uplo::General, 4, 1.f, A, 4, x, 1, 0.f, y, 1));
	REQUIRE_THROWS(blas::trsv(Uplo::Upper, Op::NoTrans, Diag::NonUnit, 5, A, 4, x, 1));
	REQUIRE_THROWS(blas::syrk(Uplo::Lower, Op::NoTrans, 4, 4, 1.f, A, 3, 0.f, B, 4));
	REQUIRE_THROWS(blas::herk(Uplo::Lower, Op::Trans, 4, 4, 1.f, A, 4, 0.f, B, 4));
	REQUIRE_NOTHROW(blas::trsm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, 4, 4, 1.f, A, 4, B, 4));
}
}
