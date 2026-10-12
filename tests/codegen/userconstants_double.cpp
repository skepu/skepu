#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/util.hpp>
#include <test_helper_automatic_setup.hpp>

// User constants of double precision, in a Map, a MapReduce and a Map that is a template (not built for OpenCL: not all
// devices have double precision; see userconstants.cpp).

[[skepu::userconstant]] constexpr double
	GAIN_D = 1.0 / 3.0,
	SHIFT_D = -2.5;

double map_double(double x) { return x * GAIN_D + SHIFT_D; }
double square_shifted(double x) { return x * x + SHIFT_D; }

template<typename T>
T map_template(T x) { return x * (T)GAIN_D + (T)SHIFT_D; }

template<typename T>
void apply_template(skepu::Vector<T> &in, skepu::Vector<T> &out)
{
	auto skel = skepu::Map<1>(map_template<T>);
	skel(out, in);
}

auto skel_map = skepu::Map<1>(map_double);
auto skel_mapreduce = skepu::MapReduce<1>(square_shifted, skepu::util::add<double>);

TEST_CASE("User constants of type double")
{
	const size_t n{50};
	skepu::Vector<double> x(n), y(n);
	for (size_t k = 0; k < n; ++k) x[k] = (double)k;

	skel_map(y, x);
	y.updateHost();
	for (size_t k = 0; k < n; ++k) REQUIRE(y(k) == (double)k * GAIN_D + SHIFT_D);

	double expected = 0;
	for (size_t k = 0; k < n; ++k) expected += (double)k * (double)k + SHIFT_D;
	REQUIRE((double)skel_mapreduce(x) == Approx(expected).epsilon(1e-12));
}

TEST_CASE("User constants in a Map that is a template: float and double")
{
	const size_t n{20};
	skepu::Vector<float> f(n), fr(n);
	skepu::Vector<double> d(n), dr(n);
	for (size_t k = 0; k < n; ++k) { f[k] = (float)k; d[k] = (double)k; }
	apply_template(f, fr);
	apply_template(d, dr);
	fr.updateHost(); dr.updateHost();
	for (size_t k = 0; k < n; ++k)
	{
		REQUIRE(fr(k) == (float)k * (float)GAIN_D + (float)SHIFT_D);
		REQUIRE(dr(k) == (double)k * GAIN_D + SHIFT_D);
	}
}
