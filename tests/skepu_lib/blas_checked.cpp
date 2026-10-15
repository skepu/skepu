// Each part is in its own namespace; its test cases are prefixed with its name.

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
