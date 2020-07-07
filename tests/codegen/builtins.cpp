//#define SKEPU_ENABLE_EXCEPTIONS
#define PRECISION 1E-3

#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>
#include <cmath>

using namespace std;

auto skepu_exp      = skepu::Map<1>([](float f) -> float { return exp(f); });
auto skepu_exp2     = skepu::Map<1>([](float f) -> float { return exp2(f); });
auto skepu_sqrt     = skepu::Map<1>([](float f) -> float { return sqrt(f); });
auto skepu_abs      = skepu::Map<1>([](float f) -> float { return abs((int)f); });
auto skepu_fabs     = skepu::Map<1>([](float f) -> float { return fabs(f); });
auto skepu_max      = skepu::Map<1>([](float f, float f2) -> float { return max(f, f2); });
auto skepu_fmax     = skepu::Map<1>([](float f, float f2) -> float { return fmax(f, f2); });
auto skepu_pow      = skepu::Map<1>([](float f, int p) -> float { return pow(f, p); });
auto skepu_log      = skepu::Map<1>([](float f) -> float { return log(f); });
auto skepu_log2     = skepu::Map<1>([](float f) -> float { return log2(f); });
auto skepu_log10    = skepu::Map<1>([](float f) -> float { return log10(f); });
auto skepu_sin      = skepu::Map<1>([](float f) -> float { return sin(f); });
auto skepu_sinh     = skepu::Map<1>([](float f) -> float { return sinh(f); });
auto skepu_asin     = skepu::Map<1>([](float f) -> float { return asin(f); });
auto skepu_asinh    = skepu::Map<1>([](float f) -> float { return asinh(f); });
auto skepu_cos      = skepu::Map<1>([](float f) -> float { return cos(f); });
auto skepu_cosh     = skepu::Map<1>([](float f) -> float { return cosh(f); });
auto skepu_acos     = skepu::Map<1>([](float f) -> float { return acos(f); });
auto skepu_acosh    = skepu::Map<1>([](float f) -> float { return acosh(f); });
auto skepu_tan      = skepu::Map<1>([](float f) -> float { return tan(f); });
auto skepu_tanh     = skepu::Map<1>([](float f) -> float { return tanh(f); });
auto skepu_atan     = skepu::Map<1>([](float f) -> float { return atan(f); });
auto skepu_atanh    = skepu::Map<1>([](float f) -> float { return atanh(f); });
auto skepu_round    = skepu::Map<1>([](float f) -> float { return round(f); });
auto skepu_ceil     = skepu::Map<1>([](float f) -> float { return ceil(f); });
auto skepu_floor    = skepu::Map<1>([](float f) -> float { return floor(f); });
auto skepu_erf      = skepu::Map<1>([](float f) -> float { return erf(f); });
auto skepu_printf   = skepu::Map<1>([](float f, float val) -> float { printf("Hello, world!"); return val; });


void is_similar(float a, float b)
{
	CHECK((
		(isnan(a) && isnan(b)) ||
		(isinf(a) && isinf(b)) ||
		(a == Approx(b).epsilon(PRECISION)))
	);
}


TEST_CASE("Test built-in math functions")
{
	size_t constexpr N{1000};

	// Initialize input
	skepu::Vector<float> in(N), res(N);
	skepu::external([&]
	{
		for(size_t i = 0; i < N; ++i)
			in(i) = (float)i / 100.0;
	}, skepu::write(in));
	
	
	
	
	// exp
	REQUIRE_NOTHROW(skepu_exp(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::exp(in(i)));
	});
	
	
	// exp2
	REQUIRE_NOTHROW(skepu_exp2(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::exp2(in(i)));
	});
	
	
	// sqrt
	REQUIRE_NOTHROW(skepu_sqrt(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::sqrt(in(i)));
	});
	
	
	// abs
	REQUIRE_NOTHROW(skepu_abs(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::abs((int)in(i)));
	});
	
	
	// fabs
	REQUIRE_NOTHROW(skepu_fabs(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::fabs(in(i)));
	});
	
	
	// max
	REQUIRE_NOTHROW(skepu_max(res, in, (float)0.5));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::max(in(i), (float)0.5));
	});
	
	
	// fmax
	REQUIRE_NOTHROW(skepu_fmax(res, in, (float)0.5));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::fmax(in(i), (float)0.5));
	});
	
	
	// pow
	REQUIRE_NOTHROW(skepu_pow(res, in, 2));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::pow(in(i), 2));
	});
	
	
	// log
	REQUIRE_NOTHROW(skepu_log(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::log(in(i)));
	});
	
	
	// log2
	REQUIRE_NOTHROW(skepu_log2(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::log2(in(i)));
	});
	
	
	// log10
	REQUIRE_NOTHROW(skepu_log10(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::log10(in(i)));
	});
	
	
	// sin
	REQUIRE_NOTHROW(skepu_sin(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::sin(in(i)));
	});
	
	
	// sinh
	REQUIRE_NOTHROW(skepu_sinh(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::sinh(in(i)));
	});
	
	
	// asin
	REQUIRE_NOTHROW(skepu_asin(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::asin(in(i)));
	});
	
	
	// asinh
	REQUIRE_NOTHROW(skepu_asinh(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::asinh(in(i)));
	});
	
	
	// cos
	REQUIRE_NOTHROW(skepu_cos(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::cos(in(i)));
	});
	
	
	// cosh
	REQUIRE_NOTHROW(skepu_cosh(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::cosh(in(i)));
	});
	
	
	// acos
	REQUIRE_NOTHROW(skepu_acos(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::acos(in(i)));
	});
	
	
	// acosh
	REQUIRE_NOTHROW(skepu_acosh(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::acosh(in(i)));
	});
	
	
	// tan
	REQUIRE_NOTHROW(skepu_tan(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::tan(in(i)));
	});
	
	
	// tanh
	REQUIRE_NOTHROW(skepu_tanh(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::tanh(in(i)));
	});
	
	
	// atan
	REQUIRE_NOTHROW(skepu_atan(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::atan(in(i)));
	});
	
	
	// atanh
	REQUIRE_NOTHROW(skepu_atanh(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::atanh(in(i)));
	});
	
	
	// round
	REQUIRE_NOTHROW(skepu_round(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::round(in(i)));
	});
	
	
	// ceil
	REQUIRE_NOTHROW(skepu_ceil(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::ceil(in(i)));
	});
	
	
	// floor
	REQUIRE_NOTHROW(skepu_floor(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::floor(in(i)));
	});
	
	
	// erf
	REQUIRE_NOTHROW(skepu_erf(res, in));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), std::erf(in(i)));
	});
	
	
	// printf
	REQUIRE_NOTHROW(skepu_printf(res, in, 0));
	skepu::external(skepu::read(res, in), [&]
	{
		for(size_t i = 0; i < N; ++i)
			is_similar(res(i), 0);
	});
}
