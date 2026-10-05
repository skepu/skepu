#include "../../external/catch2/catch.hpp"

#include <cmath>
#include <skepu>
#include <skepu-lib/blas.hpp>

// Checked results of the float BLAS routines that conjugate_gradient uses, on every backend.

static void require_close(float got, double want)
{
	REQUIRE(std::fabs(got - want) <= 1e-4 * std::max(1.0, std::fabs(want)));
}

TEST_CASE("BLAS scal and axpy with a float and with a Scalar alpha")
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

TEST_CASE("BLAS gemv, non-square, alpha and beta")
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
