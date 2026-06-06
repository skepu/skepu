#include <iostream>

#pragma once

namespace skepu {
namespace complex {

template<typename T>
struct complex
{
  T re;
  T im;
  
  using value_type = T;
  
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
  constexpr complex()
  : re{0}, im{0}
  {}
  
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
  constexpr complex(T real)
  : re{real}, im{0}
  {}
    
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
  constexpr complex(T real, T im)
  : re{real}, im{0}
  {}
  
  template<typename T1, typename T2>
#ifdef SKEPU_CUDA
__host__ __device__
#endif
  constexpr complex(T1 real, T2 imaginary) 
  : re(real), im(imaginary)
  {}
};

using DComplex = complex<double>;
using FComplex = complex<float>;


// Type traits

template<typename T>
struct value_type {};

template<>
struct value_type<DComplex> { using type = double; };

template<>
struct value_type<FComplex> { using type = float; };

template<typename T>
struct value_type<complex<T>> { using type = T; };


template<typename T>
struct is_skepu_complex: std::false_type {};

template<>
struct is_skepu_complex<DComplex>: std::true_type {};

template<>
struct is_skepu_complex<FComplex>: std::true_type {};


// Constants

// Cannot be used inside userfunctions, usable e.g. for default values to Reduce skeleton invocations
constexpr DComplex DZero = {0.0, 0.0};
constexpr DComplex DOne  = {1.0, 0.0};

constexpr FComplex FZero = {0.0, 0.0};
constexpr FComplex FOne = {1.0, 0.0};


// Functions

template<typename C>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
inline
C add(C lhs, C rhs)
{
  C res;
  res.re = lhs.re + rhs.re;
  res.im = lhs.im + rhs.im;
  return res;
}

template<typename C>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
inline
C sub(C lhs, C rhs)
{
  C res;
  res.re = lhs.re - rhs.re;
  res.im = lhs.im - rhs.im;
  return res;
}

template<typename C>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
inline
C mul(C lhs, C rhs)
{
  C res;
  res.re = lhs.re * rhs.re - lhs.im * rhs.im;
  res.im = lhs.im * rhs.re + lhs.re * rhs.im;
  return res;
}

template<typename C>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
inline
C real_div(C z, typename value_type<C>::type div)
{
  C res;
  res.re = z.re / div;
  res.im = z.im / div;
  return res;
}

template<typename R, typename std::enable_if<!is_skepu_complex<R>::value, bool>::type = 0>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
inline
R conj(R x)
{
  return x;
}

template<typename C, typename std::enable_if<is_skepu_complex<C>::value, bool>::type = 0>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
inline
C conj(C z)
{
  C res;
  res.re = z.re;
  res.im = -z.im;
  return res;
}

template<typename C>
inline
typename value_type<C>::type sq_norm(C z)
{
  return z.re * z.re + z.im * z.im;
}

template<typename C>
inline
typename value_type<C>::type real(C z)
{
  return z.re;
}

template<typename C>
inline
typename value_type<C>::type imag(C z)
{
  return z.im;
}

// 1-norm
template<typename T>
T abs1(T x)
{
  return fabs(x);
}

// 1-norm
template<typename T>
T abs1(complex<T> x)
{
  return std::abs(real(x)) + std::abs(imag(x));
}




// For use outside skeletons

template<typename C>
inline 
typename std::enable_if<is_skepu_complex<C>::value, std::ostream &>::type
operator<<(std::ostream &o, C z)
{
  o << z.re << (z.im >= 0 ? "+" : "") << z.im << "i";
  return o;
}




// OPERATOR +

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator+(complex<T> lhs, complex<T> rhs)
{
  return add(lhs, rhs);
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator+(complex<T> lhs, T rhs)
{
  return {lhs.re + rhs, lhs.im};
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator+(T lhs, complex<T> rhs)
{
  return {rhs.re + lhs, rhs.im};
}


// OPERATOR +=

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T>& operator+=(complex<T> &lhs, complex<T> rhs)
{
  lhs.re += rhs.re;
  lhs.im += rhs.im;
  return lhs;
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T>& operator+=(complex<T> &lhs, T rhs)
{
  lhs.re += rhs;
  return lhs;
}


// OPERATOR -

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator-(complex<T> lhs, complex<T> rhs)
{
  return sub(lhs, rhs);
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator-(complex<T> lhs, T rhs)
{
  return {lhs.re - rhs, lhs.im};
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator-(T lhs, complex<T> rhs)
{
  return {lhs - rhs.re, -rhs.im};
}


// OPERATOR -=

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T>& operator-=(complex<T> &lhs, complex<T> rhs)
{
  lhs.re -= rhs.re;
  lhs.im -= rhs.im;
  return lhs;
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T>& operator-=(complex<T> &lhs, T rhs)
{
  lhs.re -= rhs;
  return lhs;
}


// OPERATOR *

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator*(complex<T> lhs, complex<T> rhs)
{
  return mul(lhs, rhs);
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator*(complex<T> lhs, T rhs)
{
  return {lhs.re * rhs, lhs.im * rhs};
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator*(T lhs, complex<T> rhs)
{
  return {rhs.re * lhs, rhs.im * lhs};
}

// OPERATOR /

/*template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator/(complex<T> lhs, complex<T> rhs)
{
  return mul(lhs, rhs);
}*/

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
complex<T> operator/(complex<T> lhs, T rhs)
{
  return real_div(lhs, rhs);
}


// OPERATOR ==

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
bool operator==(complex<T> lhs, complex<T> rhs)
{
  return (lhs.re == rhs.re) && (lhs.im == rhs.im);
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
bool operator==(complex<T> lhs, T rhs)
{
  return lhs.im == 0 && (lhs.re == rhs);
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
bool operator==(T lhs, complex<T> rhs)
{
  return rhs.im == 0 && (rhs.re == lhs);
}


// OPERATOR !=

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
bool operator!=(complex<T> lhs, complex<T> rhs)
{
  return (lhs.re != rhs.re) || (lhs.im != rhs.im);
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
bool operator!=(complex<T> lhs, T rhs)
{
  return lhs.im != 0 || (lhs.re != rhs);
}

template<typename T>
#ifdef SKEPU_CUDA
	__host__ __device__
#endif
bool operator!=(T lhs, complex<T> rhs)
{
  return rhs.im != 0 || (rhs.re != lhs);
}



}} // skepu::complex

#ifdef SKEPU_OPENCL

namespace skepu
{
	template<> inline std::string getDataTypeCL<complex::complex<float>>         () { return "complex_float";          }
	template<> inline std::string getDataTypeCL<complex::complex<double>>        () { return "complex_double";         }
}

#endif