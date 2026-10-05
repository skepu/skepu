#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOfBlasHPP;

#include <skepu-lib/complex.hpp>

#define BLAS_CONST // const

namespace cplx = skepu::complex;
using FComplex = cplx::complex<float>;
using DComplex = cplx::complex<double>;


// Can't have types defined in an enclosed namespace (yet)
// Can't have type templates in SkepU (yet)
struct iamax_tmp_float
{
	float val;
	size_t index;
};

struct iamax_tmp_double
{
	double val;
	size_t index;
};

template<typename T>
struct iamax_tmp_type_helper {};

template<>
struct iamax_tmp_type_helper<float>
{
	using type = iamax_tmp_float;
};

template<>
struct iamax_tmp_type_helper<double>
{
	using type = iamax_tmp_double;
};

template<typename T>
using iamax_type = typename iamax_tmp_type_helper<T>::type;





namespace skepu {

namespace blas {

	using stride_type = int;
	using size_type = size_t;

#define SKEPU_BLAS_STRIDE_TYPE_UF int
#define SKEPU_BLAS_SIZE_TYPE_UF size_t

	// the plain type of a scalar argument that may be a Scalar<T> (the result of dot, ...)
	template< typename T >
	struct unwrap_scalar { using type = typename std::decay<T>::type; };

	template< typename T >
	struct unwrap_scalar< skepu::Scalar<T> > { using type = T; };

	template< typename T >
	using unwrap_scalar_t = typename unwrap_scalar< typename std::decay<T>::type >::type;

	// for zero types
	template< typename... Types >
	struct scalar_type_traits;

	// define scalar_type<> type alias
	template< typename... Types >
	using scalar_type = typename scalar_type_traits< Types... >::type;

	// for one type
	template< typename T >
	struct scalar_type_traits< T >
	{
	  using type = typename std::decay<T>::type;
	};

	// for two types
	// relies on type of ?: operator being the common type of its two arguments
	template< typename T1, typename T2 >
	struct scalar_type_traits< T1, T2 >
	{
		using type = typename std::decay< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >::type;
	};

	// for either or both complex,
	// find common type of associated real types, then add complex
	template<> struct scalar_type_traits<FComplex, float> { using type = FComplex; };
	template<> struct scalar_type_traits<FComplex, double> { using type = DComplex; };
	template<> struct scalar_type_traits<DComplex, float> { using type = DComplex; };
	template<> struct scalar_type_traits<DComplex, double> { using type = DComplex; };

	template<> struct scalar_type_traits<float, FComplex> { using type = FComplex; };
	template<> struct scalar_type_traits<double, FComplex> { using type = DComplex; };
	template<> struct scalar_type_traits<float, DComplex> { using type = DComplex; };
	template<> struct scalar_type_traits<double, DComplex> { using type = DComplex; };

	// for three or more types
	template< typename T1, typename T2, typename... Types >
	struct scalar_type_traits< T1, T2, Types... >
	{
	    using type = scalar_type< scalar_type< T1, T2 >, Types... >;
	};




	// for zero types
	template< typename... Types >
	struct real_type_traits;

	// define real_type<> type alias
	template< typename... Types >
	using real_type = typename real_type_traits< Types... >::real_t;

	// define complex_type<> type alias
	template< typename... Types >
	using complex_type = skepu::complex::complex< real_type< Types... > >;

	// for one type
	template< typename T >
	struct real_type_traits<T>
	{
	    using real_t = T;
	};

	// for one complex type, strip complex
	template< typename T >
	struct real_type_traits< skepu::complex::complex<T> >
	{
	    using real_t = T;
	};

	// for two or more types
	template< typename T1, typename... Types >
	struct real_type_traits< T1, Types... >
	{
	    using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
	};



	using namespace skepu::complex;

// ================================================================================
// ================================================================================
// ==========================       LEVEL 1 BLAS
// ================================================================================
// ================================================================================



// ----------------------------------------------------------
//   SWAP
// ----------------------------------------------------------


template<typename TX, typename TY>
skepu::multiple<TY, TX> swap_uf(TX x, TY y)
{
	return skepu::ret(y, x);
}

template<typename TX, typename TY>
void swap(
	size_type                        n,
	Vector<TX> &                     x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	// if (incx == incy == 1) AND (TX == TY) then pointer-swap
	auto skel = Map<2>(swap_uf<TX, TY>);
	skel.setStride(incx, incy, incx, incy);
	skel.setLabel("SWAP");
	skel(x, y, x, y);
}


// ----------------------------------------------------------
//   SCAL
// ----------------------------------------------------------

template<typename TX, typename TS>
TX scal_uf(TX x, TS alpha)
{
	return x * alpha;
}

template<typename TX, typename TS>
void scal(
	size_type                        n,
	TS                               alpha,
	Vector<TX> &                     x,
	stride_type                      incx
)
{
	// (the user function is the same for a plain and a Scalar<> alpha: a skeleton in a
	// function template can only have one user function per instantiation of the user function)
	auto skel = Map<1>(scal_uf<TX, scalar_type<TX, unwrap_scalar_t<TS>>>);
	skel.setStride(incx, incx);
	skel.setLabel("SCAL");
	skel(x, x, alpha);
}


// ----------------------------------------------------------
//   COPY
// ----------------------------------------------------------

template<typename TX, typename TY>
TY copy_uf(TX x)
{
	return x;
}

template<typename TX, typename TY>
void copy(
	size_type                        n,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	auto skel = Map<1>(copy_uf<TX, TY>);
	skel.setStride(incy, incx);
	skel.setLabel("COPY");
	skel(y, x);
}


// ----------------------------------------------------------
//   AXPY
// ----------------------------------------------------------

template<typename TX, typename TY, typename TA = scalar_type<TX, TY>>
TY axpy_uf(TX x, TY y, TA alpha)
{
	return alpha * x + y;
}

template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
void axpy(
	size_type                        n,
	TS                               alpha,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	auto skel = Map<2>(axpy_uf<TX, TY>);
	skel.setStride(incy, incx, incy);
	skel.setLabel("AXPY");
	skel(y, x, y, alpha);
}


// ----------------------------------------------------------
//   DOT
// ----------------------------------------------------------

template<typename TX, typename TY, typename TR = scalar_type<TX, TY>>
TR dot_uf_1(TX x, TY y)
{
	return skepu::complex::conj(x) * y;
}

template<typename T>
T dot_uf_2(T lhs, T rhs)
{
	return lhs + rhs;
}

template<typename TX, typename TY>
Scalar<scalar_type<TX, TY>> dot(
	size_type                        n,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> BLAS_CONST&           y,
	stride_type                      incy
)
{
	auto skel = MapReduce<2>(dot_uf_1<TX, TY>, dot_uf_2<scalar_type<TX, TY>>);
	skel.setStride(incx, incy);
	skel.setLabel("DOT");
	return skel(x, y);
}



// ----------------------------------------------------------
//   DOTU
// ----------------------------------------------------------

template<typename TX, typename TY, typename TR = scalar_type<TX, TY>>
TR dotu_uf_1(TX x, TY y)
{
	return x * y;
}

template<typename TX, typename TY>
Scalar<scalar_type<TX, TY>> dotu(
	size_type                        n,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> BLAS_CONST&           y,
	stride_type                      incy
)
{
	auto skel = MapReduce<2>(dotu_uf_1<TX, TY>, dot_uf_2<scalar_type<TX, TY>>); // ???
	skel.setStride(incx, incy);
	skel.setLabel("DOTU");
	return skel(x, y);
}



// ----------------------------------------------------------
//   NRM2
// ----------------------------------------------------------

template<typename T, typename TR = real_type<T>>
TR nrm2_uf_1(T x)
{
	return real(x) * real(x) + imag(x) * imag(x);
}

template<typename T>
T nrm2_uf_2(T lhs, T rhs)
{
	return lhs + rhs;
}

template<typename T>
real_type<T> nrm2(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
)
{
	auto skel = MapReduce<1>(nrm2_uf_1<T>, nrm2_uf_2<real_type<T>>);
	skel.setStride(incx);
	skel.setLabel("NRM2");
	return sqrt(skel(x)); // TODO: Scalar<T>
}


// ----------------------------------------------------------
//   ASUM
// ----------------------------------------------------------

template<typename T>
real_type<T> asum_uf_1(T x)
{
	return abs1(x);
}

template<typename T>
Scalar<real_type<T>> asum(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
)
{
	auto skel = MapReduce<1>(asum_uf_1<T>, dot_uf_2<real_type<T>>);
	skel.setStride(incx);
	skel.setLabel("ASUM");
	return skel(x);
}


// ----------------------------------------------------------
//   IAMAX
// ----------------------------------------------------------

template<typename T, typename R = real_type<T>, typename H = iamax_type<R>>
H iamax_uf_1(skepu::Index1D index, T xi)
{
	H tmp;
	tmp.val = abs1(xi);
	tmp.index = index.i;
	return tmp;
}

template<typename H>
H iamax_uf_2(H lhs, H rhs)
{
	return (lhs.val > rhs.val) ? lhs : rhs;
}

template<typename T>
size_type iamax(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
)
{
	auto skel = MapReduce<1>(iamax_uf_1<T>, iamax_uf_2<iamax_type<real_type<T>>>);
	skel.setStride(incx);
	skel.setLabel("IAMAX");
	iamax_type<real_type<T>> res = skel(x);
	return res.index;
}


// ----------------------------------------------------------
//   ROTG
// ----------------------------------------------------------

template<typename T>
T sign(T a, T b)
{
	T x = (a >= 0 ? a : - a);
	return b >= 0 ? x : -x;
}

template<typename T>
void rotg(T *a, T *b, T *c, T *s)
{
	const T c_b4 = 1.;
  T d__1, d__2;
  T r, scale, z, roe;

  roe = *b;
  if (abs(*a) > abs(*b))
		roe = *a;

  scale = abs(*a) + abs(*b);

	if (scale != 0.)
	{
		/* Computing 2nd power */
	  d__1 = *a / scale;
		/* Computing 2nd power */
	  d__2 = *b / scale;
	  r = scale * sqrt(d__1 * d__1 + d__2 * d__2);
	  r = sign(c_b4, roe) * r;
	  *c = *a / r;
	  *s = *b / r;
	  z = 1.;
	  if (abs(*a) > abs(*b))
			z = *s;

		if (abs(*b) >= abs(*a) && *c != 0.)
			z = 1. / *c;

	  *a = r;
	  *b = z;

	  return;
  }

  *c = 1.;
  *s = 0.;
  r = 0.;
  z = 0.;
	*a = r;
  *b = z;

  return;
}


// ----------------------------------------------------------
//   ROT
// ----------------------------------------------------------

template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
skepu::multiple<TX, TY> rot_uf(TX x, TY y, TS c, TS s)
{
	TS new_x = c * x + s * y;
	TS new_y = c * y - s * x;
	return skepu::ret(new_x, new_y);
}

template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
void rot(
	size_type                        n,
	Vector<TX> &                     x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy,
	TS                               c,
	TS                               s
)
{
  auto skel = Map<2>(rot_uf<TX, TY>);
	skel.setStride(incx, incy, incx, incy);
	skel.setLabel("ROT");
	skel(x, y, x, y, c, s);
}


// ----------------------------------------------------------
//   ROTMG
// ----------------------------------------------------------

// TODO implement

// ----------------------------------------------------------
//   ROTM
// ----------------------------------------------------------

// TODO implement




// ================================================================================
// ================================================================================
// ==========================       LEVEL 2 BLAS
// ================================================================================
// ================================================================================


// ----------------------------------------------------------
//   GEMV
// ----------------------------------------------------------

enum class Op     : char { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper    = 'U', Lower    = 'L', General   = 'G' };
enum class Diag   : char { NonUnit  = 'N', Unit     = 'U' };
enum class Side   : char { Left     = 'L', Right    = 'R' };


// y := y + alpha * A * x, one row of A (n elements) per output element
template<typename TA , typename TX , typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_notrans_noconj(TY y, MatRow<TA> Arow, Vec<TX> x, stride_type incx, TS alpha, size_type n) // TODO: reference for proxies
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF i = 0; i < n; ++i)
		tmp += x(i * incx) * Arow(i); // TODO negative incx
	return y + alpha * tmp;
}

// y := y + alpha * A^T * x, one column of A (m elements) per output element
template<typename TA , typename TX , typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_trans_noconj(TY y, MatCol<TA> Acol, Vec<TX> x, stride_type incx, TS alpha, size_type m) // TODO: reference for proxies
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF i = 0; i < m; ++i)
		tmp += x(i * incx) * Acol(i); // TODO negative incx
	return y + alpha * tmp;
}

// y := y + alpha * A^H * x
template<typename TA , typename TX , typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_trans_conj(TY y, MatCol<TA> Acol, Vec<TX> x, stride_type incx, TS alpha, size_type m) // TODO: reference for proxies
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF i = 0; i < m; ++i)
		tmp += x(i * incx) * skepu::complex::conj(Acol(i)); // TODO negative incx
	return y + alpha * tmp;
}


template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
void gemv(
	blas::Op 	                       trans,
	size_type                        m,
	size_type                        n,
	TS                               alpha,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	TS                               beta,
	Vector<TY> &                     y,
	stride_type                      incy
)
{
	const TS zero = 0, one  = 1;

  // quick return
  if (m == 0 || n == 0 || (alpha == zero && beta == one))
    return;

//  int64_t lenx = (trans == Op::NoTrans ? n : m);
//  int64_t leny = (trans == Op::NoTrans ? m : n);
//  int64_t kx = (incx > 0 ? 0 : (-lenx + 1)*incx);
//  int64_t ky = (incy > 0 ? 0 : (-leny + 1)*incy);

  // form y = beta*y
  if (beta != one)
		scal(m, beta, y, incy);

  if (alpha == zero)
    return;

  if (trans == Op::NoTrans)
	{
    auto skel = Map<1>(gemv_uf_notrans_noconj<TA, TX, TY>);
		skel.setStride(incy, incy);
		skel.setLabel("GEMV");
		skel(y, y, A, x, incx, alpha, n);
  }
  else if (trans == Op::Trans)
	{
		auto skel = Map<1>(gemv_uf_trans_noconj<TA, TX, TY>);
		skel.setStride(incy, incy);
		skel.setLabel("GEMV");
		skel(y, y, A, x, incx, alpha, m);
  }
  else // trans == Op::ConjTrans
	{
		auto skel = Map<1>(gemv_uf_trans_conj<TA, TX, TY>);
		skel.setStride(incy, incy);
		skel.setLabel("GEMV");
		skel(y, y, A, x, incx, alpha, m);
  }
}


// ----------------------------------------------------------
//   GER
// ----------------------------------------------------------

template<typename TA>
TA ger_add_uf(TA a, TA b)
{
	return a + b;
}

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
TA ger_uf(TX x, TY y, TS alpha)
{
	return alpha * x * conj(y);
}

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
void ger(
	size_type                         m,
	size_type                         n,
	TS                                alpha,
	Vector<TX> BLAS_CONST&            x,
	stride_type                       incx,
	Vector<TY> BLAS_CONST&            y,
	stride_type                       incy,
	Matrix<TA>&                       A,
	stride_type                       lda
)
{
	// (MapPairs' in place mode, adding the result to the old element, is not implemented by
	// the CUDA backend: the outer product goes to a temporary, which is then added)
	Matrix<TA> outer(A.size_i(), A.size_j());
	auto skel = skepu::MapPairs<1, 1>(ger_uf<TX, TY, TA>);
	// skel.setStride(incx, incy); // TODO implement
	skel.setLabel("GER");
	skel(outer, x, y, alpha);

	auto add = skepu::Map<2>(ger_add_uf<TA>);
	add.setLabel("GER add");
	add(A, A, outer);
}


// ----------------------------------------------------------
//   GERU
// ----------------------------------------------------------

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
TA geru_uf(TX x, TY y, TS alpha)
{
	return alpha * x * y;
}

template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
void geru(
	size_type                         m,
	size_type                         n,
	TS                                alpha,
	Vector<TX> BLAS_CONST&            x,
	stride_type                       incx,
	Vector<TY> BLAS_CONST&            y,
	stride_type                       incy,
	Matrix<TA>&                       A,
	stride_type                       lda
)
{
	// (MapPairs' in place mode, adding the result to the old element, is not implemented by
	// the CUDA backend: the outer product goes to a temporary, which is then added)
	Matrix<TA> outer(A.size_i(), A.size_j());
	auto skel = skepu::MapPairs<1, 1>(geru_uf<TX, TY, TA>);
	// skel.setStride(incx, incy); // TODO implement
	skel.setLabel("GERU");
	skel(outer, x, y, alpha);

	auto add = skepu::Map<2>(ger_add_uf<TA>);
	add.setLabel("GERU add");
	add(A, A, outer);
}





// ================================================================================
// ================================================================================
// ==========================       LEVEL 3 BLAS
// ================================================================================
// ================================================================================


// ----------------------------------------------------------
//   GEMM
// ----------------------------------------------------------

template<typename T1, typename T2>
T1 gemm_uf_matrix_scale(T1 x, T2 beta)
{
	return x * beta;
}

// C(i, j) := alpha * sum_l op(A)(i, l) * op(B)(l, j) + beta * C(i, j)
// op is 0 (no transpose), 1 (transpose) or 2 (conjugate transpose); the elements of A and B
// are read through random access proxies since a row of a transposed matrix is not the
// row (or column) that goes with the output element.
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf(skepu::Index2D index, TC c, const skepu::Mat<TA> A, const skepu::Mat<TB> B, size_type k, int opa, int opb, TS alpha, TS beta)
{
	TC res = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < k; ++l)
	{
		TA a = (opa == 0) ? A(index.row, l) : A(l, index.row);
		TB b = (opb == 0) ? B(l, index.col) : B(index.col, l);
		if (opa == 2) a = skepu::complex::conj(a);
		if (opb == 2) b = skepu::complex::conj(b);
		res += a * b;
	}
	return alpha * res + beta * c;
}


template<typename TA, typename TB, typename TC>
void gemm(
	blas::Op                         transA,
	blas::Op                         transB,
	size_type                        m,
	size_type                        n,
	size_type                        k,
	scalar_type<TA, TB, TC>          alpha,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Matrix<TB> BLAS_CONST&           B,
	size_type                        ldb,
	scalar_type<TA, TB, TC>          beta,
	Matrix<TC>&                      C,
	size_type                        ldc
)
{
	typedef blas::scalar_type<TA, TB, TC> scalar_t;

  // constants
  const scalar_t zero = 0;
  const scalar_t one  = 1;

//  blas_error_if( lda < ((transA != Op::NoTrans) ? k : m) );
//  blas_error_if( ldb < ((transB != Op::NoTrans) ? n : k) );
//  blas_error_if( ldc < m );

  // quick return
  if (m == 0 || n == 0 || k == 0)
      return;

  if (alpha == zero)
	{
		auto skel_scale = Map<1>(gemm_uf_matrix_scale<TC, scalar_t>);
		skel_scale.setLabel("GEMM scale");
		skel_scale(C, C, beta);
    return;
  }

  // alpha != zero
	auto skel = skepu::Map<1>(gemm_uf<TA, TB, TC>);
	skel.setLabel("GEMM");
	const int opa = (transA == Op::NoTrans) ? 0 : (transA == Op::Trans ? 1 : 2);
	const int opb = (transB == Op::NoTrans) ? 0 : (transB == Op::Trans ? 1 : 2);
	skel(C, C, A, B, k, opa, opb, alpha, beta);
}
/*

// option A
map = Map<>(...);
map.setTriangular(skepu::Upper);
map(my_matrix);


O O O O
O O O X
O O X X
O X X X

// option B
my_matrix = skepu::TriangleMatrix()

*/

#undef SKEPU_BLAS_STRIDE_TYPE_UF
#undef SKEPU_BLAS_SIZE_TYPE_UF
}}

static skepu::PrecompilerMarker endOfBlasHPP;
#endif // SKEPU_PRECOMPILED
