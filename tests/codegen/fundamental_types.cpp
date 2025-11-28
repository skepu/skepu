#include "../../external/catch2/catch.hpp"

#include <skepu>

// see https://en.cppreference.com/w/cpp/language/types.html


#include <cmath>
#define PRECISION 1E-3
template <typename T>
void is_similar(T a, T b)
{
	CHECK((
		(std::isnan(a) && std::isnan(b)) ||
		(std::isinf(a) && std::isinf(b)) ||
		(a == Approx(b).epsilon(PRECISION)))
	);
}


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

TEST_CASE("Basic types")
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

TEST_CASE("Char variants")
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


TEST_CASE("Int variants")
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
#ifndef SKEPU_CUDA
TEST_CASE("Long double")
{
    skepu::Vector<long double> vLongDouble(1, 1.0l), rLongDouble(1, 0.0l);
    mapLongDouble(rLongDouble, vLongDouble);
    rLongDouble.flush();
    is_similar(rLongDouble(0), 1.0l);
}
#endif //SKEPU_CUDA