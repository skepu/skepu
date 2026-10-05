#include "../../external/catch2/catch.hpp"

#include <cmath>
#include <skepu>
#include <skepu-lib/blas.hpp>
#include "blas_check.hpp"

// Checked results of the real float BLAS level 2 and 3 routines on every backend, for
// non-square sizes, all transpose variants and several alpha and beta, against host
// computations.

using namespace blas_check;
using skepu::Vector;
using skepu::Matrix;
namespace blas = skepu::blas;

static const blas::Op ops[] = { blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans };

TEST_CASE("BLAS gemv: all operations, non-square, alpha and beta")
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

TEST_CASE("BLAS gemv with strides")
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

TEST_CASE("BLAS ger and geru")
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

TEST_CASE("BLAS gemm: all operations, m, n and k different")
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
