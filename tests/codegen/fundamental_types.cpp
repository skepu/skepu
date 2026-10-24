// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <test_helper_automatic_setup.hpp>
#include <test_helper_functions.hpp>
#include <iostream>
#include <cmath>


// ======== fundamental_types ========

namespace fundamental_types_part
{
// see https://en.cppreference.com/w/cpp/language/types.html

int uf_int(int a)
{
    return a;
}

float uf_float(float a)
{
    return a;
}

double uf_double(double a)
{
    return a;
}

bool uf_bool(bool a)
{
    return a;
}

char uf_char(char a)
{
    return a;
}

auto mapInt = skepu::Map(uf_int);
auto mapFloat = skepu::Map(uf_float);
auto mapDouble = skepu::Map(uf_double);
auto mapBool = skepu::Map(uf_bool);
auto mapChar = skepu::Map(uf_char);

TEST_CASE("fundamental_types: Basic types")
{
    skepu::Vector<int> vInt(1, 1), rInt(1, 0);
    mapInt(rInt, vInt);
    rInt.flush();
    CHECK(rInt(0) == 1);

    skepu::Vector<float> vFloat(1, 1.0f), rFloat(1, 0.0f);
    mapFloat(rFloat, vFloat);
    rFloat.flush();
    is_similar(rFloat(0), 1.0f);

    skepu::Vector<double> vDouble(1, 1.0), rDouble(1, 0.0);
    mapDouble(rDouble, vDouble);
    rDouble.flush();
    is_similar(vDouble(0), 1.0);

    skepu::Vector<bool> vBool(1, true), rBool(1, false);
    mapBool(rBool, vBool);
    rBool.flush();
    CHECK(rBool(0) == true);

    skepu::Vector<char> vChar(1, 'a'), rChar(1, 'b');
    mapChar(rChar, vChar);
    rChar.flush();
    CHECK(rChar(0) == 'a');
}


signed char uf_signed_char(signed char a)
{
    return a;
}

unsigned char uf_unsigned_char(unsigned char a)
{
    return a;
}

auto mapSignedChar = skepu::Map(uf_signed_char);
auto mapUnsignedChar = skepu::Map(uf_unsigned_char);

TEST_CASE("fundamental_types: Char variants")
{
    skepu::Vector<signed char> vSignedChar(1, 'a'), rSignedChar(1, 'b');
    mapSignedChar(rSignedChar, vSignedChar);
    rSignedChar.flush();
    CHECK(rSignedChar(0) == 'a');

    skepu::Vector<unsigned char> vUnsignedChar(1, 'a'), rUnsignedChar(1, 'b');
    mapUnsignedChar(rUnsignedChar, vUnsignedChar);
    rUnsignedChar.flush();
    CHECK(rUnsignedChar(0) == 'a');
}

short uf_short(short a)
{
    return a;
}

short int uf_short_int(short int a)
{
    return a;
}

signed short uf_signed_short(signed short a)
{
    return a;
}

signed short int uf_signed_short_int(signed short int a)
{
    return a;
}

unsigned short uf_unsigned_short(unsigned short a)
{
    return a;
}

unsigned short int uf_unsigned_short_int(unsigned short int a)
{
    return a;
}

signed uf_signed(signed a)
{
    return a;
}

signed int uf_signed_int(signed int a)
{
    return a;
}

unsigned uf_unsigned(unsigned a)
{
    return a;
}

unsigned int uf_unsigned_int(unsigned int a)
{
    return a;
}

long uf_long(long a)
{
    return a;
}

long int uf_long_int(long int a)
{
    return a;
}

signed long uf_signed_long(signed long a)
{
    return a;
}

signed long int uf_signed_long_int(signed long int a)
{
    return a;
}

unsigned long uf_unsigned_long(unsigned long a)
{
    return a;
}

unsigned long int uf_unsigned_long_int(unsigned long int a)
{
    return a;
}

long long uf_long_long(long long a)
{
    return a;
}

long long int uf_long_long_int(long long int a)
{
    return a;
}

signed long long uf_signed_long_long(signed long long a)
{
    return a;
}

signed long long int uf_signed_long_long_int(signed long long int a)
{
    return a;
}

unsigned long long uf_unsigned_long_long(unsigned long long a)
{
    return a;
}

unsigned long long int uf_unsigned_long_long_int(unsigned long long int a)
{
    return a;
}

auto mapShort = skepu::Map(uf_short);
auto mapShortInt = skepu::Map(uf_short_int);
auto mapSignedShort = skepu::Map(uf_signed_short);
auto mapSignedShortInt = skepu::Map(uf_signed_short_int);
auto mapUnsignedShort = skepu::Map(uf_unsigned_short);
auto mapUnsignedShortInt = skepu::Map(uf_unsigned_short_int);

auto mapSigned = skepu::Map(uf_signed);
auto mapSignedInt = skepu::Map(uf_signed_int);
auto mapUnsigned = skepu::Map(uf_unsigned);
auto mapUnsignedInt = skepu::Map(uf_unsigned_int);

auto mapLong = skepu::Map(uf_long);
auto mapLongInt = skepu::Map(uf_long_int);
auto mapSignedLong = skepu::Map(uf_signed_long);
auto mapSignedLongInt = skepu::Map(uf_signed_long_int);
auto mapUnsignedLong = skepu::Map(uf_unsigned_long);
auto mapUnsignedLongInt = skepu::Map(uf_unsigned_long_int);

auto mapLongLong = skepu::Map(uf_long_long);
auto mapLongLongInt = skepu::Map(uf_long_long_int);
auto mapSignedLongLong = skepu::Map(uf_signed_long_long);
auto mapSignedLongLongInt = skepu::Map(uf_signed_long_long_int);
auto mapUnsignedLongLong = skepu::Map(uf_unsigned_long_long);
auto mapUnsignedLongLongInt = skepu::Map(uf_unsigned_long_long_int);


TEST_CASE("fundamental_types: Int variants")
{

    SECTION("Short")
    {
        skepu::Vector<short> vShort(1, 1), rShort(1, 0);
        mapShort(rShort, vShort);
        rShort.flush();
        CHECK(rShort(0) == 1);

        skepu::Vector<short int> vShortInt(1, 1), rShortInt(1, 0);
        mapShortInt(rShortInt, vShortInt);
        rShortInt.flush();
        CHECK(rShortInt(0) == 1);

        skepu::Vector<signed short> vSignedShort(1, 1), rSignedShort(1, 0);
        mapSignedShort(rSignedShort, vSignedShort);
        rSignedShort.flush();
        CHECK(rSignedShort(0) == 1);

        skepu::Vector<signed short int> vSignedShortInt(1, 1), rSignedShortInt(1, 0);
        mapSignedShort(rSignedShortInt, vSignedShortInt);
        rSignedShortInt.flush();
        CHECK(rSignedShortInt(0) == 1);

        skepu::Vector<unsigned short> vUnsignedShort(1, 1), rUnsignedShort(1, 0);
        mapUnsignedShort(rUnsignedShort, vUnsignedShort);
        rUnsignedShort.flush();
        CHECK(rUnsignedShort(0) == 1);

        skepu::Vector<unsigned short int> vUnsignedShortInt(1, 1), rUnsignedShortInt(1, 0);
        mapUnsignedShortInt(rUnsignedShortInt, vUnsignedShortInt);
        rUnsignedShortInt.flush();
        CHECK(rUnsignedShortInt(0) == 1);
    }


    SECTION("Int")
    {
        skepu::Vector<signed> vSigned(1, 1), rSigned(1, 0);
        mapSigned(rSigned, vSigned);
        rSigned.flush();
        CHECK(rSigned(0) == 1);

        skepu::Vector<signed int> vSignedInt(1, 1), rSignedInt(1, 0);
        mapSignedInt(rSignedInt, vSignedInt);
        rSignedInt.flush();
        CHECK(rSignedInt(0) == 1);

        skepu::Vector<unsigned> vUnsigned(1, 1), rUnsigned(1, 0);
        mapUnsigned(rUnsigned, vUnsigned);
        rUnsigned.flush();
        CHECK(rUnsigned(0) == 1);

        skepu::Vector<unsigned int> vUnsignedInt(1, 1), rUnsignedInt(1, 0);
        mapUnsignedInt(rUnsignedInt, vUnsignedInt);
        rUnsignedInt.flush();
        CHECK(rUnsignedInt(0) == 1);
    }

    SECTION("Long")
    {
        skepu::Vector<long> vLong(1, 1), rLong(1, 0);
        mapLong(rLong, vLong);
        rLong.flush();
        CHECK(rLong(0) == 1);

        skepu::Vector<long int> vLongInt(1, 1), rLongInt(1, 0);
        mapLongInt(rLongInt, vLongInt);
        rLongInt.flush();
        CHECK(rLongInt(0) == 1);

        skepu::Vector<signed long> vSignedLong(1, 1), rSignedLong(1, 0);
        mapSignedLong(rSignedLong, vSignedLong);
        rSignedLong.flush();
        CHECK(rSignedLong(0) == 1);

        skepu::Vector<signed long int> vSignedLongInt(1, 1), rSignedLongInt(1, 0);
        mapSignedLongInt(rSignedLongInt, vSignedLongInt);
        rSignedLongInt.flush();
        CHECK(rSignedLongInt(0) == 1);

        skepu::Vector<unsigned long> vUnsignedLong(1, 1), rUnsignedLong(1, 0);
        mapUnsignedLong(rUnsignedLong, vUnsignedLong);
        rUnsignedLong.flush();
        CHECK(rUnsignedLong(0) == 1);

        skepu::Vector<unsigned long int> vUnsignedLongInt(1, 1), rUnsignedLongInt(1, 0);
        mapUnsignedLongInt(rUnsignedLongInt, vUnsignedLongInt);
        rUnsignedLongInt.flush();
        CHECK(rUnsignedLongInt(0) == 1);
    }

    SECTION("Long long")
    {
        skepu::Vector<long long> vLongLong(1, 1), rLongLong(1, 0);
        mapLongLong(rLongLong, vLongLong);
        rLongLong.flush();
        CHECK(rLongLong(0) == 1);

        skepu::Vector<long long int> vLongLongInt(1, 1), rLongLongInt(1, 0);
        mapLongLongInt(rLongLongInt, vLongLongInt);
        rLongLongInt.flush();
        CHECK(rLongLongInt(0) == 1);

        skepu::Vector<signed long long> vSignedLongLong(1, 1), rSignedLongLong(1, 0);
        mapSignedLongLong(rSignedLongLong, vSignedLongLong);
        rSignedLongLong.flush();
        CHECK(rSignedLongLong(0) == 1);

        skepu::Vector<signed long long int> vSignedLongLongInt(1, 1), rSignedLongLongInt(1, 0);
        mapSignedLongLongInt(rSignedLongLongInt, vSignedLongLongInt);
        rSignedLongLongInt.flush();
        CHECK(rSignedLongLongInt(0) == 1);

        skepu::Vector<unsigned long long> vUnsignedLongLong(1, 1), rUnsignedLongLong(1, 0);
        mapUnsignedLongLong(rUnsignedLongLong, vUnsignedLongLong);
        rUnsignedLongLong.flush();
        CHECK(rUnsignedLongLong(0) == 1);

        skepu::Vector<unsigned long long int> vUnsignedLongLongInt(1, 1), rUnsignedLongLongInt(1, 0);
        mapUnsignedLongLongInt(rUnsignedLongLongInt, vUnsignedLongLongInt);
        rUnsignedLongLongInt.flush();
        CHECK(rUnsignedLongLongInt(0) == 1);
    }
}

// This function seems to have weird behaviour in OpenCL if you add additional logic to it such as
// a + 1.0l, a + 1.0, a - 1.0. It also doesn't work properly if you return 1.0l instead of a, rLongDouble
// doesn't get updated for some reason.
long double uf_long_double(long double a)
{
    return a;
}

auto mapLongDouble = skepu::Map(uf_long_double);

// CUDA does not support long double in device code (https://forums.developer.nvidia.com/t/support-for-long-double-data-type-in-gpu-code/313337),
// The compiler should be throwing the following warning: warning #20208-D: 'long double' is treated as 'double' in device code
// The skeletons above should still be tested for code generation purposes.
// Nor does OpenCL, where it is a double of 8 bytes: with a host long double of another size (x86: 80 bits) the elements of the containers do not match.
#if !defined(SKEPU_CUDA) && (!defined(SKEPU_OPENCL) || __LDBL_MANT_DIG__ == __DBL_MANT_DIG__)
TEST_CASE("fundamental_types: Long double")
{
    skepu::Vector<long double> vLongDouble(1, 1.0l), rLongDouble(1, 0.0l);
    mapLongDouble(rLongDouble, vLongDouble);
    rLongDouble.flush();
    is_similar(rLongDouble(0), 1.0l);
}
#endif // long double
}


// ======== builtins ========

namespace builtins_part
{
//#define SKEPU_ENABLE_EXCEPTIONS


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


TEST_CASE("builtins: Test built-in math functions")
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
			is_similar(res(i), static_cast<float>(std::abs((int)in(i))));
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
			is_similar(res(i), static_cast<float>(std::pow(in(i), 2)));
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
			is_similar(res(i), 0.0f);
	});
}


// The same functions called with a std:: qualifier (OpenCL C has them under their plain names; std::abs of a float is fabs)
auto skepu_std_math = skepu::Map<2>([](float f, float f2) -> float
{
	return std::sin(f) + std::sqrt(f) + std::pow(f, 2.f) + std::fabs(f - f2) + std::abs(f - f2) + std::max(f, f2) + std::min<float>(f, f2) + ::floor(f);
});

TEST_CASE("builtins: Test math functions called with std::")
{
	size_t constexpr N{1000};

	skepu::Vector<float> in(N), in2(N), res(N);
	skepu::external([&]
	{
		for(size_t i = 0; i < N; ++i)
		{
			in(i) = (float)i / 100.0f;
			in2(i) = 3.f;
		}
	}, skepu::write(in, in2));

	REQUIRE_NOTHROW(skepu_std_math(res, in, in2));
	skepu::external(skepu::read(res, in, in2), [&]
	{
		for(size_t i = 0; i < N; ++i)
		{
			const float f = in(i), f2 = in2(i);
			is_similar(res(i), std::sin(f) + std::sqrt(f) + std::pow(f, 2.f) + std::fabs(f - f2) + std::abs(f - f2) + std::max(f, f2) + std::min(f, f2) + std::floor(f));
		}
	});
}
}
