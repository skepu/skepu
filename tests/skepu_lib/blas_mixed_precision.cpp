#include "../../external/catch2/catch.hpp"

#include <cmath>
#include <skepu>
#include <skepu-lib/blas.hpp>

// BLAS routines in one program with float and double operands (and a mix of the two): the skeleton in each
// routine is instantiated with different types, which gives it different user functions and kernels.
// (Not built for OpenCL: not all devices have double precision.)

namespace blas = skepu::blas;
using skepu::Vector;

TEST_CASE("BLAS level 1 with float, double and mixed operands")
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
