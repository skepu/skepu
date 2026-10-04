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

}} // skepu::util
