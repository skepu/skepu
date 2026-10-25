// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include "blas_check.hpp"
#include <skepu>
#include <cmath>
#include <complex>
#include <skepu-lib/blas.hpp>


// ======== blas_complex ========

namespace blas_complex_part
{
// Checked results of the BLAS routines on complex<float> (sequential, CPU and OpenMP
// backends) against std::complex computations.

using skepu::Vector;
using skepu::Matrix;
namespace blas = skepu::blas;
using C = skepu::complex::complex<float>;
using SC = std::complex<double>;

static C make(size_t i, size_t salt) { return C{blas_check::val(i, salt), blas_check::val(i * 3 + 1, salt + 7)}; }
static SC std_of(C z) { return SC(z.re, z.im); }
static bool near(C got, SC want) { return std::abs(std_of(got) - want) <= 1e-4 * std::max(1.0, std::abs(want)); }
static bool near(SC got, SC want) { return std::abs(got - want) <= 1e-4 * std::max(1.0, std::abs(want)); }

static void fill(Vector<C> &v, size_t salt) { for (size_t i = 0; i < v.size(); ++i) v[i] = make(i, salt); }
static void fill(Matrix<C> &m, size_t salt) { for (size_t i = 0; i < m.size(); ++i) m[i] = make(i, salt); }
template<typename V> static std::vector<SC> host(V &v)
{
	v.updateHost();
	std::vector<SC> h(v.size());
	for (size_t i = 0; i < h.size(); ++i) h[i] = std_of(v(i));
	return h;
}

TEST_CASE("blas_complex: BLAS complex level 1")
{
	const size_t n{37};
	Vector<C> x(n), y(n);
	fill(x, 1); fill(y, 2);
	auto hx = host(x), hy = host(y);

	SC dotc = 0, dotu = 0; double nrm = 0, asum = 0;
	for (size_t i = 0; i < n; ++i)
	{
		dotc += std::conj(hx[i]) * hy[i];
		dotu += hx[i] * hy[i];
		nrm += std::norm(hx[i]);
		asum += std::fabs(hx[i].real()) + std::fabs(hx[i].imag());
	}
	REQUIRE(near(std_of(blas::dot(n, x, 1, y, 1)), dotc));
	REQUIRE(near(std_of(blas::dotu(n, x, 1, y, 1)), dotu));
	REQUIRE(blas_check::close(blas::nrm2(n, x, 1), std::sqrt(nrm)));
	REQUIRE(blas_check::close(blas::asum(n, x, 1), asum));

	// iamax uses |re| + |im|
	size_t best = 0;
	for (size_t i = 1; i < n; ++i)
		if (std::fabs(hx[i].real()) + std::fabs(hx[i].imag()) > std::fabs(hx[best].real()) + std::fabs(hx[best].imag()))
			best = i;
	size_t got = blas::iamax(n, x, 1);
	REQUIRE(std::fabs(hx[got].real()) + std::fabs(hx[got].imag()) == std::fabs(hx[best].real()) + std::fabs(hx[best].imag()));

	blas::axpy(n, 0.5f, x, 1, y, 1);
	auto r = host(y);
	for (size_t i = 0; i < n; ++i)
		REQUIRE(near(r[i], hy[i] + 0.5 * hx[i]));
}

TEST_CASE("blas_complex: BLAS complex gemv and gemm with conjugate transpose")
{
	const size_t m{4}, n{3}, k{5};
	{
		Matrix<C> A(m, n);
		Vector<C> x(m), y(n);
		fill(A, 3); fill(x, 4); fill(y, 5);
		auto hA = host(A), hx = host(x), hy = host(y);
		blas::gemv(blas::Op::ConjTrans, m, n, 1.f, A, n, x, 1, 0.5f, y, 1);
		auto r = host(y);
		for (size_t j = 0; j < n; ++j)
		{
			SC want = 0.5 * hy[j];
			for (size_t i = 0; i < m; ++i)
				want += std::conj(hA[i * n + j]) * hx[i];
			REQUIRE(near(r[j], want));
		}
	}
	for (blas::Op opA : {blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans})
		for (blas::Op opB : {blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans})
		{
			const bool ta = opA != blas::Op::NoTrans, tb = opB != blas::Op::NoTrans;
			const size_t ar = ta ? k : m, ac = ta ? m : k, br = tb ? n : k, bc = tb ? k : n;
			Matrix<C> A(ar, ac), B(br, bc), Cm(m, n);
			fill(A, 6); fill(B, 7); fill(Cm, 8);
			auto hA = host(A), hB = host(B), hC = host(Cm);
			blas::gemm(opA, opB, m, n, k, 1.f, A, ac, B, bc, 0.5f, Cm, n);
			auto r = host(Cm);
			for (size_t i = 0; i < m; ++i)
				for (size_t j = 0; j < n; ++j)
				{
					SC want = 0.5 * hC[i * n + j];
					for (size_t l = 0; l < k; ++l)
					{
						SC a = ta ? hA[l * ac + i] : hA[i * ac + l];
						SC b = tb ? hB[j * bc + l] : hB[l * bc + j];
						if (opA == blas::Op::ConjTrans) a = std::conj(a);
						if (opB == blas::Op::ConjTrans) b = std::conj(b);
						want += a * b;
					}
					INFO("opA " << (int)opA << " opB " << (int)opB << " (" << i << "," << j << ")");
					REQUIRE(near(r[i * n + j], want));
				}
		}
}
}


// ======== blas_mixed_precision ========

namespace blas_mixed_precision_part
{
// BLAS routines in one program with float and double operands (and a mix of the two): the skeleton in each
// routine is instantiated with different types, which gives it different user functions and kernels.
// (Not built for OpenCL: not all devices have double precision.)

namespace blas = skepu::blas;
using skepu::Vector;

TEST_CASE("blas_mixed_precision: BLAS level 1 with float, double and mixed operands")
{
	const size_t n{32};
	Vector<float> xf(n), yf(n);
	Vector<double> xd(n), yd(n);
	for (size_t i = 0; i < n; ++i)
	{
		xf[i] = float(i + 1); yf[i] = 2.f;
		xd[i] = double(i + 1); yd[i] = 2.;
	}

	blas::scal(n, 2.f, xf, 1);
	blas::scal(n, 0.5, xd, 1);              // double alpha: a different instantiation than float alpha
	blas::axpy(n, 3.f, xf, 1, yf, 1);       // float, float
	blas::axpy(n, 4., xd, 1, yd, 1);        // double, double
	blas::axpy(n, 1.f, xf, 1, yd, 1);       // float into double

	xf.updateHost(); yf.updateHost(); xd.updateHost(); yd.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		const double v = double(i + 1);
		REQUIRE(xf(i) == float(2 * v));
		REQUIRE(xd(i) == 0.5 * v);
		REQUIRE(yf(i) == float(3 * 2 * v + 2));
		REQUIRE(yd(i) == 4 * 0.5 * v + 2 + 2 * v);
	}

	double sum = 0, sumsq = 0;
	for (size_t i = 0; i < n; ++i)
	{
		sum += double(xd(i)) * double(yd(i));
		sumsq += double(xd(i)) * double(xd(i));
	}
	REQUIRE(std::fabs(double(blas::dot(n, xd, 1, yd, 1)) - sum) <= 1e-9 * sum);
	REQUIRE(std::fabs(double(blas::nrm2(n, xd, 1)) - std::sqrt(sumsq)) <= 1e-9 * std::sqrt(sumsq));

	double sumf = 0, sumsqf = 0;
	for (size_t i = 0; i < n; ++i)
	{
		sumf += double(xf(i)) * double(yf(i));
		sumsqf += double(xf(i)) * double(xf(i));
	}
	REQUIRE(std::fabs(double(blas::dot(n, xf, 1, yf, 1)) - sumf) <= 1e-4 * sumf);
	REQUIRE(std::fabs(double(blas::nrm2(n, xf, 1)) - std::sqrt(sumsqf)) <= 1e-4 * std::sqrt(sumsqf));
}

TEST_CASE("blas_mixed_precision: BLAS dsdot and sdsdot sum in double precision")
{
	const size_t n{1000};
	Vector<float> x(n), y(n);
	for (size_t i = 0; i < n; ++i)
	{
		x[i] = 1.f + 1e-4f * (float)(i % 7);
		y[i] = (i % 2) ? 1e4f : -1e4f + 1.f;
	}
	x.updateHost(); y.updateHost();
	double want = 0;
	for (size_t i = 0; i < n; ++i)
		want += (double)x(i) * (double)y(i);
	REQUIRE(std::fabs(double(blas::dsdot(n, x, 1, y, 1)) - want) <= 1e-12 * std::fabs(want) + 1e-9);
	REQUIRE(std::fabs(double(blas::sdsdot(n, 2.f, x, 1, y, 1)) - (want + 2.0)) <= 1e-6 * std::max(1.0, std::fabs(want)));
	REQUIRE((float)blas::dsdot(0, x, 1, y, 1) == 0.f);
}
}


// ======== blas_complex_triangle ========

namespace blas_complex_triangle_part
{
// The routines of matrices stored in a triangle on complex<float>: Hermitian, symmetric and triangular, against std::complex
// computations.

using skepu::Vector;
using skepu::Matrix;
namespace blas = skepu::blas;
using blas::Uplo;
using blas::Op;
using blas::Diag;
using blas::Side;
using C = skepu::complex::complex<float>;
using SC = std::complex<double>;

static C make(size_t i, size_t salt) { return C{blas_check::val(i, salt), blas_check::val(i * 3 + 1, salt + 7)}; }
static SC std_of(C z) { return SC(z.re, z.im); }
static bool near(SC got, SC want, double tol = 1e-3) { return std::abs(got - want) <= tol * std::max(1.0, std::abs(want)); }

static void fill(Vector<C> &v, size_t salt) { for (size_t i = 0; i < v.size(); ++i) v[i] = make(i, salt); }
static void fill(Matrix<C> &m, size_t salt) { for (size_t i = 0; i < m.size(); ++i) m[i] = make(i, salt); }
static void fill_triangular(Matrix<C> &m, size_t salt)
{
	fill(m, salt);
	for (size_t i = 0; i < m.size_i(); ++i)
		for (size_t j = 0; j < m.size_j(); ++j)
		{
			C &v = m[i * m.size_j() + j];
			v = (i == j) ? C{4.f + 0.1f * (float)(i % 3), 0.5f} : C{0.1f * v.re, 0.1f * v.im};
		}
}
template<typename V> static std::vector<SC> host(V &v)
{
	v.updateHost();
	std::vector<SC> h(v.size());
	for (size_t i = 0; i < h.size(); ++i) h[i] = std_of(v(i));
	return h;
}
static const Uplo uplos[] = { Uplo::Upper, Uplo::Lower };
static const Op ops[] = { Op::NoTrans, Op::Trans, Op::ConjTrans };

TEST_CASE("blas_complex_triangle: hemv, symv, trmv and trsv")
{
	const size_t n = 6;
	for (Uplo uplo : uplos)
	{
		Matrix<C> A(n, n);
		Vector<C> x(n), y(n), y2(n);
		fill(A, 1); fill(x, 2); fill(y, 3); fill(y2, 3);
		const auto hA = host(A), hx = host(x), hy = host(y);
		const SC alpha(1.5, 0), beta(-0.5, 0);

		blas::hemv(uplo, n, 1.5f, A, n, x, 1, -0.5f, y, 1);
		blas::symv(uplo, n, 1.5f, A, n, x, 1, -0.5f, y2, 1);
		const auto H = blas_check::ref::triangle(hA, n, n, 1, uplo == Uplo::Upper, false);
		const auto S = blas_check::ref::triangle(hA, n, n, 0, uplo == Uplo::Upper, false);
		const auto Hx = blas_check::ref::mul(H, n, n, hx, 1), Sx = blas_check::ref::mul(S, n, n, hx, 1);
		const auto ry = host(y), ry2 = host(y2);
		for (size_t i = 0; i < n; ++i)
		{
			REQUIRE(near(ry[i], alpha * Hx[i] + beta * hy[i]));
			REQUIRE(near(ry2[i], alpha * Sx[i] + beta * hy[i]));
		}

		Matrix<C> T(n, n);
		fill_triangular(T, 4);
		const auto hT = host(T);
		for (Op op : ops)
			for (Diag diag : {Diag::NonUnit, Diag::Unit})
			{
				INFO("uplo " << (int)uplo << " op " << (int)op << " diag " << (int)diag);
				const auto M = blas_check::ref::opm(blas_check::ref::triangle(hT, n, n, 2, uplo == Uplo::Upper, diag == Diag::Unit), n, n, op == Op::NoTrans ? 0 : (op == Op::Trans ? 1 : 2));
				Vector<C> t(n), s(n);
				fill(t, 2); fill(s, 2);
				blas::trmv(uplo, op, diag, n, T, n, t, 1);
				const auto Mx = blas_check::ref::mul(M, n, n, hx, 1);
				const auto rt = host(t);
				for (size_t i = 0; i < n; ++i)
				{
					INFO("i " << i << " got " << rt[i] << " want " << Mx[i]);
					REQUIRE(near(rt[i], Mx[i]));
				}

				blas::trsv(uplo, op, diag, n, T, n, s, 1);
				const auto xs = host(s);
				const auto back = blas_check::ref::mul(M, n, n, xs, 1);
				for (size_t i = 0; i < n; ++i)
					REQUIRE(near(back[i], hx[i]));
			}
	}
}

TEST_CASE("blas_complex_triangle: her, her2, hemm, herk and her2k")
{
	const size_t n = 5, k = 3;
	for (Uplo uplo : uplos)
	{
		INFO("uplo " << (int)uplo);
		Vector<C> x(n), y(n);
		Matrix<C> A1(n, n), A2(n, n);
		fill(x, 1); fill(y, 2); fill(A1, 3); fill(A2, 3);
		const auto hx = host(x), hy = host(y), hA = host(A1);
		blas::her(uplo, n, 0.5f, x, 1, A1, n);
		blas::her2(uplo, n, SC(0.5, 0.25).real() > 0 ? C{0.5f, 0.25f} : C{0.f, 0.f}, x, 1, y, 1, A2, n);
		const auto r1 = host(A1), r2 = host(A2);
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j)
			{
				const bool in = uplo == Uplo::Upper ? i <= j : i >= j;
				SC w1 = hA[i * n + j], w2 = hA[i * n + j];
				if (in)
				{
					w1 += 0.5 * hx[i] * std::conj(hx[j]);
					w2 += SC(0.5, 0.25) * hx[i] * std::conj(hy[j]) + std::conj(SC(0.5, 0.25)) * hy[i] * std::conj(hx[j]);
					if (i == j) { w1 = w1.real(); w2 = w2.real(); }
				}
				REQUIRE(near(r1[i * n + j], w1));
				REQUIRE(near(r2[i * n + j], w2));
			}

		// hemm
		for (Side side : {Side::Left, Side::Right})
		{
			const size_t m = 4, cols = 3, kk = (side == Side::Left) ? m : cols;
			Matrix<C> A(kk, kk), B(m, cols), Cm(m, cols);
			fill(A, 5); fill(B, 6); fill(Cm, 7);
			const auto hA2 = host(A), hB = host(B), hC = host(Cm);
			const auto H = blas_check::ref::triangle(hA2, kk, kk, 1, uplo == Uplo::Upper, false);
			const auto P = (side == Side::Left) ? blas_check::ref::mul(H, m, m, hB, cols) : blas_check::ref::mul(hB, m, cols, H, cols);
			blas::hemm(side, uplo, m, cols, C{1.f, 0.5f}, A, kk, B, cols, C{0.5f, 0.f}, Cm, cols);
			const auto rc = host(Cm);
			for (size_t i = 0; i < m * cols; ++i)
				REQUIRE(near(rc[i], SC(1, 0.5) * P[i] + 0.5 * hC[i]));
		}

		// herk and her2k
		for (Op op : {Op::NoTrans, Op::ConjTrans})
		{
			const bool t = (op != Op::NoTrans);
			Matrix<C> A(t ? k : n, t ? n : k), B(t ? k : n, t ? n : k), Ck(n, n), Ck2(n, n);
			fill(A, 8); fill(B, 9); fill(Ck, 10); fill(Ck2, 10);
			const auto hAA = host(A), hBB = host(B), hCk = host(Ck);
			auto opa = [&](const std::vector<SC> &H) { std::vector<SC> P(n * k); for (size_t i = 0; i < n; ++i) for (size_t l = 0; l < k; ++l) P[i * k + l] = t ? H[l * n + i] : H[i * k + l]; return P; };
			const auto PA = opa(hAA), PB = opa(hBB);
			const auto AAh = blas_check::ref::mul(PA, n, k, blas_check::ref::opm(PA, n, k, 2), n);
			const auto ABh = blas_check::ref::mul(PA, n, k, blas_check::ref::opm(PB, n, k, 2), n);
			const auto BAh = blas_check::ref::mul(PB, n, k, blas_check::ref::opm(PA, n, k, 2), n);
			blas::herk(uplo, op, n, k, 0.5f, A, t ? n : k, 2.f, Ck, n);
			blas::her2k(uplo, op, n, k, C{0.5f, 0.25f}, A, t ? n : k, B, t ? n : k, 2.f, Ck2, n);
			const auto r3 = host(Ck), r4 = host(Ck2);
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < n; ++j)
				{
					const bool in = uplo == Uplo::Upper ? i <= j : i >= j;
					SC w3 = hCk[i * n + j], w4 = hCk[i * n + j];
					if (in)
					{
						w3 = 0.5 * AAh[i * n + j] + 2.0 * hCk[i * n + j];
						w4 = SC(0.5, 0.25) * ABh[i * n + j] + std::conj(SC(0.5, 0.25)) * BAh[i * n + j] + 2.0 * hCk[i * n + j];
						if (i == j) { w3 = w3.real(); w4 = w4.real(); }
					}
					INFO("op " << (int)op << " (" << i << "," << j << ")");
					REQUIRE(near(r3[i * n + j], w3));
					REQUIRE(near(r4[i * n + j], w4));
				}
		}
	}
}

TEST_CASE("blas_complex_triangle: trmm and trsm")
{
	const size_t m = 4, n = 3;
	for (Side side : {Side::Left, Side::Right})
		for (Uplo uplo : uplos)
			for (Op op : ops)
			{
				INFO("side " << (int)side << " uplo " << (int)uplo << " op " << (int)op);
				const size_t k = (side == Side::Left) ? m : n;
				Matrix<C> A(k, k), B(m, n), X(m, n);
				fill_triangular(A, 1); fill(B, 2); fill(X, 2);
				const auto hA = host(A), hB = host(B);
				const auto T = blas_check::ref::opm(blas_check::ref::triangle(hA, k, k, 2, uplo == Uplo::Upper, false), k, k, op == Op::NoTrans ? 0 : (op == Op::Trans ? 1 : 2));
				const auto P = (side == Side::Left) ? blas_check::ref::mul(T, m, m, hB, n) : blas_check::ref::mul(hB, m, n, T, n);
				const SC alpha(0.5, 0.25);
				blas::trmm(side, uplo, op, Diag::NonUnit, m, n, C{0.5f, 0.25f}, A, k, B, n);
				const auto rb = host(B);
				for (size_t i = 0; i < m * n; ++i)
					REQUIRE(near(rb[i], alpha * P[i]));

				blas::trsm(side, uplo, op, Diag::NonUnit, m, n, C{0.5f, 0.25f}, A, k, X, n);
				const auto hX = host(X);
				const auto Q = (side == Side::Left) ? blas_check::ref::mul(T, m, m, hX, n) : blas_check::ref::mul(hX, m, n, T, n);
				for (size_t i = 0; i < m * n; ++i)
					REQUIRE(near(Q[i], alpha * hB[i]));
			}
}
}
