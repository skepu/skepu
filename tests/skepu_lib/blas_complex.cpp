// Merged from: blas_complex.cpp, blas_mixed_precision.cpp
// (one program instead of one for each: the build of a test program has a cost of its own; each part is in a namespace, and its test cases have its name as a prefix)

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
}
