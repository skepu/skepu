#include <iostream>

#pragma once

namespace skepu {
namespace util {

  
template<typename T>
T identity(T a)
{
	return a;
}

template<typename T>
T add(T lhs, T rhs)
{
  return lhs + rhs;
}

template<typename T, typename U>
T add(T lhs, U rhs)
{
  return lhs + rhs;
}

template<typename T>
T sub(T lhs, T rhs)
{
  return lhs - rhs;
}

template<typename T, typename U>
T sub(T lhs, U rhs)
{
  return lhs - rhs;
}

template<typename T>
T mul(T lhs, T rhs)
{
  return lhs * rhs;
}

template<typename T, typename U>
T mul(T lhs, U rhs)
{
  return lhs * rhs;
}

template<typename T>
T div(T lhs, T rhs)
{
  return lhs / rhs;
}

template<typename T, typename U>
T div(T lhs, U rhs)
{
  return lhs / rhs;
}

template<typename T>
T reciprocal(T a)
{
	return 1 / a;
}

template<typename T>
T negate(T a)
{
	return -a;
}

template<typename T>
T increment(T a)
{
	return a + 1;
}

template<typename T>
T decrement(T a)
{
	return a - 1;
}

template<typename T>
T square(T a)
{
	return a * a;
}


template<typename T>
T min(T a, T b)
{
	return a < b ? a : b;
}

template<typename T>
T max(T a, T b)
{
	return a > b ? a : b;
}

// for signed and floating-point types
template<typename T>
T abs(T a)
{
	return a < 0 ? -a : a;
}

// safe for unsigned types
template<typename T>
T absdiff(T lhs, T rhs)
{
	return lhs > rhs ? lhs - rhs : rhs - lhs;
}

template<typename T, typename U>
T absdiff(T lhs, U rhs)
{
	return lhs > rhs ? lhs - rhs : rhs - lhs;
}

template<typename T>
T squared_difference(T lhs, T rhs)
{
	T diff = lhs - rhs;
	return diff * diff;
}

template<typename T, typename U>
T squared_difference(T lhs, U rhs)
{
	T diff = lhs - rhs;
	return diff * diff;
}

template<typename T>
T clamp(T a, T lo, T hi)
{
	return a < lo ? lo : (a > hi ? hi : a);
}

template<typename T, typename U>
T clamp(T a, U lo, U hi)
{
	return a < lo ? lo : (a > hi ? hi : a);
}

// a + t * (b - a)
template<typename T>
T lerp(T a, T b, T t)
{
	return a + t * (b - a);
}

template<typename T, typename U>
T lerp(T a, T b, U t)
{
	return a + t * (b - a);
}

// a * b + c; the name fma is taken by the math function
template<typename T>
T muladd(T a, T b, T c)
{
	return a * b + c;
}

template<typename T, typename U>
T muladd(T a, U b, U c)
{
	return a * b + c;
}

// Predicates return 1 or 0 as int.
template<typename T>
int equal(T lhs, T rhs)
{
	return lhs == rhs;
}

template<typename T, typename U>
int equal(T lhs, U rhs)
{
	return lhs == rhs;
}

template<typename T>
int not_equal(T lhs, T rhs)
{
	return lhs != rhs;
}

template<typename T, typename U>
int not_equal(T lhs, U rhs)
{
	return lhs != rhs;
}

template<typename T>
int less(T lhs, T rhs)
{
	return lhs < rhs;
}

template<typename T, typename U>
int less(T lhs, U rhs)
{
	return lhs < rhs;
}

template<typename T>
int less_equal(T lhs, T rhs)
{
	return lhs <= rhs;
}

template<typename T, typename U>
int less_equal(T lhs, U rhs)
{
	return lhs <= rhs;
}

template<typename T>
int greater(T lhs, T rhs)
{
	return lhs > rhs;
}

template<typename T, typename U>
int greater(T lhs, U rhs)
{
	return lhs > rhs;
}

template<typename T>
int greater_equal(T lhs, T rhs)
{
	return lhs >= rhs;
}

template<typename T, typename U>
int greater_equal(T lhs, U rhs)
{
	return lhs >= rhs;
}

template<typename T>
int is_zero(T a)
{
	return a == 0;
}

template<typename T>
int is_nonzero(T a)
{
	return a != 0;
}

template<typename T>
int is_positive(T a)
{
	return a > 0;
}

template<typename T>
int is_negative(T a)
{
	return a < 0;
}

template<typename T>
int is_nan(T a)
{
	return a != a;
}

// not NaN and not infinite: inf - inf is NaN
template<typename T>
int is_finite(T a)
{
	return a - a == 0;
}

template<typename T>
int in_range(T a, T lo, T hi)
{
	return lo <= a && a <= hi;
}

template<typename T, typename U>
int in_range(T a, U lo, U hi)
{
	return lo <= a && a <= hi;
}

// |a - b| <= rtol * max(1, |b|)
template<typename T>
int is_close(T a, T b, T rtol)
{
	T diff = a - b;
	diff = diff < 0 ? -diff : diff;
	T scale = b < 0 ? -b : b;
	scale = scale < 1 ? 1 : scale;
	return diff <= rtol * scale;
}

template<typename T, typename U>
int is_close(T a, T b, U rtol)
{
	T diff = a - b;
	diff = diff < 0 ? -diff : diff;
	T scale = b < 0 ? -b : b;
	scale = scale < 1 ? 1 : scale;
	return diff <= rtol * scale;
}

template<typename T>
int logical_and(T lhs, T rhs)
{
	return lhs && rhs;
}

template<typename T>
int logical_or(T lhs, T rhs)
{
	return lhs || rhs;
}

template<typename T>
int logical_not(T a)
{
	return !a;
}

}} // skepu::util
