#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/util.hpp>
#include <test_helper_automatic_setup.hpp>

auto abs_float = skepu::Map(skepu::util::abs<float>);
auto abs_int = skepu::Map(skepu::util::abs<int>);
auto absdiff_unsigned = skepu::Map(skepu::util::absdiff<unsigned>);
auto absdiff_float = skepu::Map(skepu::util::absdiff<float>);
auto absdiff_uniform = skepu::Map<1>(skepu::util::absdiff<float, float>);
auto sqdiff_float = skepu::Map(skepu::util::squared_difference<float>);
auto sqdiff_mixed = skepu::Map(skepu::util::squared_difference<float, int>);
auto clamp_float = skepu::Map<3>(skepu::util::clamp<float>);
auto clamp_uniform = skepu::Map<1>(skepu::util::clamp<float, float>);
auto clamp_int = skepu::Map<1>(skepu::util::clamp<int, int>);
auto lerp_float = skepu::Map<3>(skepu::util::lerp<float>);
auto lerp_uniform = skepu::Map<2>(skepu::util::lerp<float, float>);
auto muladd_float = skepu::Map<3>(skepu::util::muladd<float>);
auto muladd_uniform = skepu::Map<1>(skepu::util::muladd<float, float>);
auto sum_sqdiff = skepu::MapReduce<2>(skepu::util::squared_difference<float>, skepu::util::add<float>);

TEST_CASE("util abs and absdiff")
{
	const float fv[] = {-3.5f, 0.f, 2.f, -0.25f, 1e6f};
	skepu::Vector<float> f(5), rf(5);
	for (size_t i = 0; i < 5; ++i) f(i) = fv[i];
	abs_float(rf, f);
	rf.flush();
	for (size_t i = 0; i < 5; ++i) CHECK(rf(i) == std::abs(fv[i]));

	skepu::Vector<int> iv(4), ri(4);
	const int ii[] = {-7, 0, 5, -1};
	for (size_t i = 0; i < 4; ++i) iv(i) = ii[i];
	abs_int(ri, iv);
	ri.flush();
	for (size_t i = 0; i < 4; ++i) CHECK(ri(i) == std::abs(ii[i]));

	skepu::Vector<unsigned> ua(4), ub(4), ru(4);
	const unsigned au[] = {5, 2, 7, 0}, bu[] = {2, 5, 7, 3};
	for (size_t i = 0; i < 4; ++i) { ua(i) = au[i]; ub(i) = bu[i]; }
	absdiff_unsigned(ru, ua, ub);
	ru.flush();
	const unsigned expect_u[] = {3, 3, 0, 3};
	for (size_t i = 0; i < 4; ++i) CHECK(ru(i) == expect_u[i]);

	skepu::Vector<float> a(5), b(5, 1.5f), r(5);
	for (size_t i = 0; i < 5; ++i) a(i) = fv[i];
	absdiff_float(r, a, b);
	r.flush();
	for (size_t i = 0; i < 5; ++i) CHECK(r(i) == std::abs(fv[i] - 1.5f));
	absdiff_uniform(r, a, 1.5f);
	r.flush();
	for (size_t i = 0; i < 5; ++i) CHECK(r(i) == std::abs(fv[i] - 1.5f));
}

TEST_CASE("util squared_difference")
{
	skepu::Vector<float> a(4), b(4), r(4);
	skepu::Vector<int> c(4);
	const float av[] = {1.f, -2.f, 0.5f, 3.f}, bv[] = {4.f, 2.f, 0.5f, 1.f};
	for (size_t i = 0; i < 4; ++i) { a(i) = av[i]; b(i) = bv[i]; c(i) = (int)i; }
	sqdiff_float(r, a, b);
	r.flush();
	for (size_t i = 0; i < 4; ++i) CHECK(r(i) == (av[i] - bv[i]) * (av[i] - bv[i]));
	sqdiff_mixed(r, a, c);
	r.flush();
	for (size_t i = 0; i < 4; ++i) CHECK(r(i) == (av[i] - (float)i) * (av[i] - (float)i));
	const float total = sum_sqdiff(a, b);
	CHECK(total == 9.f + 16.f + 0.f + 4.f);
}

TEST_CASE("util clamp")
{
	const float xs[] = {-5.f, 0.f, 0.5f, 1.f, 9.f};
	skepu::Vector<float> x(5), lo(5, 0.f), hi(5, 1.f), r(5);
	for (size_t i = 0; i < 5; ++i) x(i) = xs[i];
	const float expected[] = {0.f, 0.f, 0.5f, 1.f, 1.f};
	clamp_float(r, x, lo, hi);
	r.flush();
	for (size_t i = 0; i < 5; ++i) CHECK(r(i) == expected[i]);
	clamp_uniform(r, x, 0.f, 1.f);
	r.flush();
	for (size_t i = 0; i < 5; ++i) CHECK(r(i) == expected[i]);

	skepu::Vector<int> xi(4), ri(4);
	const int is[] = {-3, 2, 5, 11};
	for (size_t i = 0; i < 4; ++i) xi(i) = is[i];
	clamp_int(ri, xi, 0, 5);
	ri.flush();
	const int expected_i[] = {0, 2, 5, 5};
	for (size_t i = 0; i < 4; ++i) CHECK(ri(i) == expected_i[i]);
}

TEST_CASE("util lerp and muladd")
{
	skepu::Vector<float> a(4), b(4), t(4), r(4);
	const float av[] = {0.f, 10.f, -2.f, 5.f}, bv[] = {10.f, 20.f, 2.f, 5.f}, tv[] = {0.f, 0.5f, 0.25f, 1.f};
	for (size_t i = 0; i < 4; ++i) { a(i) = av[i]; b(i) = bv[i]; t(i) = tv[i]; }
	lerp_float(r, a, b, t);
	r.flush();
	for (size_t i = 0; i < 4; ++i) CHECK(r(i) == av[i] + tv[i] * (bv[i] - av[i]));
	lerp_uniform(r, a, b, 0.5f);
	r.flush();
	for (size_t i = 0; i < 4; ++i) CHECK(r(i) == av[i] + 0.5f * (bv[i] - av[i]));

	muladd_float(r, a, b, t);
	r.flush();
	for (size_t i = 0; i < 4; ++i) CHECK(r(i) == av[i] * bv[i] + tv[i]);
	muladd_uniform(r, a, 2.f, 1.f);
	r.flush();
	for (size_t i = 0; i < 4; ++i) CHECK(r(i) == av[i] * 2.f + 1.f);
}
