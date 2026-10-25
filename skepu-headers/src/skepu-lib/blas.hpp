#pragma once
#include <skepu>
#ifndef SKEPU_PRECOMPILED
static skepu::PrecompilerMarker startOfBlasHPP;

#include <limits>
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
// ==========================       CONVENTIONS
// ================================================================================
// ================================================================================
//
// Vectors: a routine on n elements of a vector x with the increment incx works on the elements x[0], x[incx], ..., x[(n-1) * incx]
// of the Vector. A negative increment starts from the far end (the first element of the operation is x[(n-1) * |incx|]), as in
// the reference BLAS. The Vector must have at least 1 + (n-1) * |incx| elements, and may have more. The routines that are not
// defined for a negative increment in the reference BLAS (scal, nrm2, asum, iamax) do nothing for one.
//
// Matrices: the matrices are row-major SkePU Matrix containers, with their own leading dimension (the number of columns), so that
// lda (ldb, ldc) must be the number of columns of the Matrix. A routine on a matrix of m rows and n columns works on its
// leading m x n part, which is the whole Matrix when it has m rows and n columns.
//
// Zero sizes: a routine with a size of zero does nothing, and a scalar result is zero. Index results (iamax) are zero-based.


// ----------------------------------------------------------
//   Helpers: the elements of a BLAS vector
// ----------------------------------------------------------

inline int abs_stride(stride_type s)
{
	return s < 0 ? -s : s;
}

// The scalar value of a result that is a Scalar when the program is traced and a plain value otherwise
template<typename R>
R scalar_value(const Scalar<R> &s)
{
	return s.m_value;
}

template<typename R>
R scalar_value(R v)
{
	return v;
}

inline void check_vector(const char *routine, const char *name, size_t size, size_type n, stride_type inc)
{
	if (inc == 0)
		SKEPU_ERROR("blas::" << routine << ": the increment of " << name << " is zero");
	if (n > 0 && size < 1 + (n - 1) * (size_type)abs_stride(inc))
		SKEPU_ERROR("blas::" << routine << ": the vector " << name << " has " << size << " elements, which is too few for n = " << n
			<< " and increment " << inc);
}

template<typename TA>
void check_matrix(const char *routine, const char *name, const Matrix<TA> &A, size_type rows, size_type cols, size_type ld)
{
	if (A.size_i() < rows || A.size_j() < cols)
		SKEPU_ERROR("blas::" << routine << ": the matrix " << name << " is " << A.size_i() << " x " << A.size_j()
			<< ", which is too small for " << rows << " x " << cols);
	if (ld != A.size_j())
		SKEPU_ERROR("blas::" << routine << ": the leading dimension of " << name << " is " << ld << ", but the Matrix has " << A.size_j()
			<< " columns (the leading dimension of a SkePU Matrix is its number of columns)");
}

// Element i of the vector that is gathered by (first, inc) is src(first + i * inc)
template<typename T>
T gather_uf(skepu::Index1D index, skepu::Vec<T> src, int first, int inc)
{
	return src(first + (int)index.i * inc);
}

template<typename T>
void gather(Vector<T> &dst, Vector<T> &src, int first, int inc SKEPU_CALL_SITE_PARAM)
{
	auto skel = Map<0>(gather_uf<T>);
	skel.setLabel("BLAS gather");
	skel(dst, src, first, inc);
}

// The elements of src that gather takes are replaced by those of values (n of them), the others are kept
template<typename T>
T scatter_uf(skepu::Index1D index, T old, skepu::Vec<T> values, int first, int inc, int n)
{
	const int d = (int)index.i - first;
	if (d % inc != 0)
		return old;
	const int q = d / inc;
	return (q >= 0 && q < n) ? values(q) : old;
}

template<typename T>
void scatter(Vector<T> &dst, Vector<T> &values, int first, int inc, size_type n SKEPU_CALL_SITE_PARAM)
{
	auto skel = Map<1>(scatter_uf<T>);
	skel.setLabel("BLAS scatter");
	skel(dst, dst, values, first, inc, (int)n);
}

// The n elements of a BLAS vector (a Vector and an increment) for a skeleton: either the Vector itself, if the elements are all of
// it and in the direction of the Vector (the skeleton then uses the increment as its stride), or a copy of the elements in a
// Vector of n elements, which is written back to the Vector by commit()
template<typename T>
class vec_view
{
public:
	vec_view(const char *routine, const char *name, Vector<T> &x, size_type n, stride_type inc, bool read, bool write)
	: m_src(x), m_n(n), m_inc(inc), m_write(write),
	  m_direct(inc > 0 && x.size() == n * (size_type)inc),
	  m_first(inc > 0 ? 0 : (int)((n - 1) * (size_type)abs_stride(inc)))
	{
		check_vector(routine, name, x.size(), n, inc);
		if (!m_direct)
		{
			// (an empty Vector makes no allocation and no event: only a view that needs a copy has one)
			m_copy = Vector<T>(n, std::string(name) + " (gathered)");
			if (read)
				gather(m_copy, m_src, m_first, m_inc);
		}
	}

	// The vector for the skeleton, and the stride that goes with it
	Vector<T> &vec() { return m_direct ? m_src : m_copy; }
	stride_type stride() const { return m_direct ? m_inc : 1; }

	void commit()
	{
		if (!m_direct && m_write)
			scatter(m_src, m_copy, m_first, m_inc, m_n);
	}

private:
	Vector<T> &m_src;
	size_type m_n;
	stride_type m_inc;
	bool m_write, m_direct;
	int m_first;
	Vector<T> m_copy;
};


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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.swap");
	if (n == 0)
		return;
	vec_view<TX> vx("swap", "x", x, n, incx, true, true);
	vec_view<TY> vy("swap", "y", y, n, incy, true, true);
	// if (incx == incy == 1) AND (TX == TY) then pointer-swap
	auto skel = Map<2>(swap_uf<TX, TY>);
	skel.setStride(vx.stride(), vy.stride(), vx.stride(), vy.stride());
	skel.setLabel("SWAP");
	skel(vx.vec(), vy.vec(), vx.vec(), vy.vec());
	vx.commit();
	vy.commit();
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.scal");
	if (n == 0 || incx <= 0)
		return;
	vec_view<TX> vx("scal", "x", x, n, incx, true, true);
	// (the user function is the same for a plain and a Scalar<> alpha, so those instantiations share a kernel)
	auto skel = Map<1>(scal_uf<TX, scalar_type<TX, unwrap_scalar_t<TS>>>);
	skel.setStride(vx.stride(), vx.stride());
	skel.setLabel("SCAL");
	skel(vx.vec(), vx.vec(), alpha);
	vx.commit();
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.copy");
	if (n == 0)
		return;
	vec_view<TX> vx("copy", "x", x, n, incx, true, false);
	vec_view<TY> vy("copy", "y", y, n, incy, false, true);
	auto skel = Map<1>(copy_uf<TX, TY>);
	skel.setStride(vy.stride(), vx.stride());
	skel.setLabel("COPY");
	skel(vy.vec(), vx.vec());
	vy.commit();
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.axpy");
	if (n == 0)
		return;
	vec_view<TX> vx("axpy", "x", x, n, incx, true, false);
	vec_view<TY> vy("axpy", "y", y, n, incy, true, true);
	auto skel = Map<2>(axpy_uf<TX, TY>);
	skel.setStride(vy.stride(), vx.stride(), vy.stride());
	skel.setLabel("AXPY");
	skel(vy.vec(), vx.vec(), vy.vec(), alpha);
	vy.commit();
}


// ----------------------------------------------------------
//   DOT
// ----------------------------------------------------------

// (the operands are converted to the type of the sum before they are multiplied: it is the type of the result of the product,
// unless the sum is of a higher precision)
template<typename TX, typename TY, typename TR = scalar_type<TX, TY>>
TR dot_uf_1(TX x, TY y)
{
	return (TR)skepu::complex::conj(x) * (TR)y;
}

template<typename T>
T dot_uf_2(T lhs, T rhs)
{
	return lhs + rhs;
}

// The dot product (with the first vector conjugated) of n elements with products and sum of the type TR
template<typename TR, typename TX, typename TY>
Scalar<TR> dot_accumulate(const char *routine, const char *label, size_type n, Vector<TX> &x, stride_type incx, Vector<TY> &y, stride_type incy)
{
	if (n == 0)
		return Scalar<TR>(TR(0));
	vec_view<TX> vx(routine, "x", x, n, incx, true, false);
	vec_view<TY> vy(routine, "y", y, n, incy, true, false);
	auto skel = MapReduce<2>(dot_uf_1<TX, TY, TR>, dot_uf_2<TR>);
	skel.setStride(vx.stride(), vy.stride());
	skel.setLabel(label);
	return skel(vx.vec(), vy.vec());
}

template<typename TX, typename TY>
Scalar<scalar_type<TX, TY>> dot(
	size_type                        n,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	Vector<TY> BLAS_CONST&           y,
	stride_type                      incy
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.dot");
	return dot_accumulate<scalar_type<TX, TY>>("dot", "DOT", n, x, incx, y, incy);
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.dotu");
	using R = scalar_type<TX, TY>;
	if (n == 0)
		return Scalar<R>(R(0));
	vec_view<TX> vx("dotu", "x", x, n, incx, true, false);
	vec_view<TY> vy("dotu", "y", y, n, incy, true, false);
	auto skel = MapReduce<2>(dotu_uf_1<TX, TY>, dot_uf_2<R>);
	skel.setStride(vx.stride(), vy.stride());
	skel.setLabel("DOTU");
	return skel(vx.vec(), vy.vec());
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

// For a vector where the sum of squares is out of range (the fallback of nrm2): the largest component and the sum of the squares
// of the components divided by it
template<typename T, typename TR = real_type<T>>
TR nrm2_absmax(T x)
{
	const TR r = real(x), i = imag(x);
	const TR ar = (r < 0) ? -r : r, ai = (i < 0) ? -i : i;
	return (ar != ar) ? ar : ((ai != ai) ? ai : ((ar > ai) ? ar : ai));
}

template<typename R>
R nrm2_max(R lhs, R rhs)
{
	return (lhs != lhs) ? lhs : ((rhs != rhs) ? rhs : ((lhs > rhs) ? lhs : rhs));
}

template<typename T, typename TR = real_type<T>>
TR nrm2_uf_scaled(T x, TR scale)
{
	const TR r = real(x) / scale, i = imag(x) / scale;
	return r * r + i * i;
}

// The square root, on the host, of the result of a reduction, as a Scalar that has the ids of what it came from (the result of a
// reduction is a plain value when the program is not traced)
template<typename R>
Scalar<R> sqrt_scalar(Scalar<R> sum)
{
	return Scalar<R>(sqrt((R)sum), sum.m_ids);
}

template<typename R>
Scalar<R> sqrt_scalar(R sum)
{
	return Scalar<R>(sqrt(sum));
}

template<typename R>
Scalar<R> scaled_sqrt_scalar(R scale, Scalar<R> sum)
{
	return Scalar<R>(scale * sqrt((R)sum), sum.m_ids);
}

template<typename R>
Scalar<R> scaled_sqrt_scalar(R scale, R sum)
{
	return Scalar<R>(scale * sqrt(sum));
}

template<typename T>
Scalar<real_type<T>> nrm2(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.nrm2");
	using R = real_type<T>;
	if (n == 0 || incx <= 0)
		return Scalar<R>(R(0));
	vec_view<T> vx("nrm2", "x", x, n, incx, true, false);
	auto skel = MapReduce<1>(nrm2_uf_1<T>, nrm2_uf_2<R>);
	skel.setStride(vx.stride());
	skel.setLabel("NRM2");
	// (the square root of the sum is done on the host: the result is a Scalar with the ids of the sum, so that the trace knows what it came from)
	auto sum = skel(vx.vec());

	// The sum of squares is the norm squared, unless it has overflowed or its small terms have underflowed (or an element is not a number)
	const R s = scalar_value(sum);
	if (s >= std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon() && s <= std::numeric_limits<R>::max())
		return sqrt_scalar(sum);

	// Otherwise the norm is the largest component times the norm of the vector divided by it
	auto skel_max = MapReduce<1>(nrm2_absmax<T>, nrm2_max<R>);
	skel_max.setStride(vx.stride());
	skel_max.setLabel("NRM2 scale");
	const R scale = scalar_value(skel_max(vx.vec()));
	if (!(scale > 0) || !(scale <= std::numeric_limits<R>::max())) // zero, infinite or not a number
		return Scalar<R>(scale);

	auto skel_scaled = MapReduce<1>(nrm2_uf_scaled<T>, nrm2_uf_2<R>);
	skel_scaled.setStride(vx.stride());
	skel_scaled.setLabel("NRM2 scaled");
	return scaled_sqrt_scalar(scale, skel_scaled(vx.vec(), scale));
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.asum");
	using R = real_type<T>;
	if (n == 0 || incx <= 0)
		return Scalar<R>(R(0));
	vec_view<T> vx("asum", "x", x, n, incx, true, false);
	auto skel = MapReduce<1>(asum_uf_1<T>, dot_uf_2<R>);
	skel.setStride(vx.stride());
	skel.setLabel("ASUM");
	return skel(vx.vec());
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

// The largest value, and the first of equal values (so that the result does not depend on the order of the reduction)
template<typename H>
H iamax_uf_2(H lhs, H rhs)
{
	return (lhs.val > rhs.val || (lhs.val == rhs.val && lhs.index < rhs.index)) ? lhs : rhs;
}

template<typename T>
size_type iamax(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.iamax");
	if (n == 0 || incx <= 0)
		return 0;
	vec_view<T> vx("iamax", "x", x, n, incx, true, false);
	auto skel = MapReduce<1>(iamax_uf_1<T>, iamax_uf_2<iamax_type<real_type<T>>>);
	skel.setStride(vx.stride());
	skel.setLabel("IAMAX");
	iamax_type<real_type<T>> res = skel(vx.vec());
	return res.index;
}


// ----------------------------------------------------------
//   IAMIN
// ----------------------------------------------------------

// The smallest value, and the first of equal values
template<typename H>
H iamin_uf_2(H lhs, H rhs)
{
	return (lhs.val < rhs.val || (lhs.val == rhs.val && lhs.index < rhs.index)) ? lhs : rhs;
}

template<typename T>
size_type iamin(
	size_type                        n,
	Vector<T> BLAS_CONST&            x,
	stride_type                      incx
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.iamin");
	using H = iamax_type<real_type<T>>;
	if (n == 0 || incx <= 0)
		return 0;
	vec_view<T> vx("iamin", "x", x, n, incx, true, false);
	auto skel = MapReduce<1>(iamax_uf_1<T>, iamin_uf_2<H>);
	skel.setStride(vx.stride());
	skel.setLabel("IAMIN");
	// (the start value of a reduction is zero, which is the smallest value)
	H start;
	start.val = std::numeric_limits<real_type<T>>::max();
	start.index = std::numeric_limits<size_t>::max();
	skel.setStartValue(start);
	H res = skel(vx.vec());
	return res.index;
}


// ----------------------------------------------------------
//   AXPBY
// ----------------------------------------------------------

// y := alpha * x + beta * y (y is not read if beta is zero)
template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
TY axpby_uf(TX x, TY y, TS alpha, TS beta)
{
	const TS zero = 0;
	return (beta == zero) ? alpha * x : alpha * x + beta * y;
}

template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
void axpby(
	size_type                        n,
	TS                               alpha,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	TS                               beta,
	Vector<TY> &                     y,
	stride_type                      incy
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.axpby");
	if (n == 0)
		return;
	const TS zero = 0;
	vec_view<TX> vx("axpby", "x", x, n, incx, true, false);
	vec_view<TY> vy("axpby", "y", y, n, incy, !(beta == zero), true);
	auto skel = Map<2>(axpby_uf<TX, TY>);
	skel.setStride(vy.stride(), vx.stride(), vy.stride());
	skel.setLabel("AXPBY");
	skel(vy.vec(), vx.vec(), vy.vec(), alpha, beta);
	vy.commit();
}


// ----------------------------------------------------------
//   DSDOT, SDSDOT
// ----------------------------------------------------------

// The dot product of two vectors of floats with the products and the sum in double precision
inline Scalar<double> dsdot(
	size_type                        n,
	Vector<float> BLAS_CONST&        x,
	stride_type                      incx,
	Vector<float> BLAS_CONST&        y,
	stride_type                      incy
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.dsdot");
	return dot_accumulate<double>("dsdot", "DSDOT", n, x, incx, y, incy);
}

// sb plus the dot product of two vectors of floats, the sum in double precision, as a float
inline Scalar<float> sdsdot(
	size_type                        n,
	float                            sb,
	Vector<float> BLAS_CONST&        x,
	stride_type                      incx,
	Vector<float> BLAS_CONST&        y,
	stride_type                      incy
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.sdsdot");
	return Scalar<float>((float)((double)sb + scalar_value(dsdot(n, x, incx, y, incy))));
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
void rotg(T *a, T *b, T *c, T *s SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.rotg");
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.rot");
	if (n == 0)
		return;
	vec_view<TX> vx("rot", "x", x, n, incx, true, true);
	vec_view<TY> vy("rot", "y", y, n, incy, true, true);
	auto skel = Map<2>(rot_uf<TX, TY>);
	skel.setStride(vx.stride(), vy.stride(), vx.stride(), vy.stride());
	skel.setLabel("ROT");
	skel(vx.vec(), vy.vec(), vx.vec(), vy.vec(), c, s);
	vx.commit();
	vy.commit();
}


// ----------------------------------------------------------
//   ROTMG
// ----------------------------------------------------------

// Sets up the modified Givens rotation: param[0] is the flag (-2: H is the identity, -1: all the elements of H are in param,
// 0: H21 and H12 are, 1: H11 and H22 are; the others are 1, -1 or 1 as the flag says), param[1..4] are H11, H21, H12, H22.
// d1, d2 and x1 are scaled (the rotation zeroes y1 of the vector (x1 sqrt(d1), y1 sqrt(d2))).
template<typename T>
void rotmg(T *d1, T *d2, T *x1, T y1, T *param SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.rotmg");
	const T gam = 4096, gamsq = 16777216, rgamsq = 5.9604645e-8;
	T flag, h11 = 0, h12 = 0, h21 = 0, h22 = 0;

	if (*d1 < 0)
	{
		flag = -1;
		*d1 = 0; *d2 = 0; *x1 = 0;
	}
	else
	{
		const T p2 = *d2 * y1;
		if (p2 == 0)
		{
			param[0] = -2;
			return;
		}
		const T p1 = *d1 * *x1;
		const T q2 = p2 * y1;
		const T q1 = p1 * *x1;
		if (std::abs(q1) > std::abs(q2))
		{
			h21 = -y1 / *x1;
			h12 = p2 / p1;
			const T u = 1 - h12 * h21;
			flag = 0;
			*d1 = *d1 / u;
			*d2 = *d2 / u;
			*x1 = *x1 * u;
		}
		else if (q2 < 0)
		{
			flag = -1;
			h11 = h12 = h21 = h22 = 0;
			*d1 = 0; *d2 = 0; *x1 = 0;
		}
		else
		{
			flag = 1;
			h11 = p1 / p2;
			h22 = *x1 / y1;
			const T u = 1 + h11 * h22;
			const T temp = *d2 / u;
			*d2 = *d1 / u;
			*d1 = temp;
			*x1 = y1 * u;
		}

		// Rescale d1 and d2 so that they are in range
		if (*d1 != 0)
		{
			while (*d1 <= rgamsq || *d1 >= gamsq)
			{
				if (flag == 0) { h11 = 1; h22 = 1; flag = -1; }
				else           { h21 = -1; h12 = 1; flag = -1; }
				if (*d1 <= rgamsq) { *d1 = *d1 * gamsq; *x1 = *x1 / gam; h11 = h11 / gam; h12 = h12 / gam; }
				else               { *d1 = *d1 / gamsq; *x1 = *x1 * gam; h11 = h11 * gam; h12 = h12 * gam; }
			}
		}
		if (*d2 != 0)
		{
			while (std::abs(*d2) <= rgamsq || std::abs(*d2) >= gamsq)
			{
				if (flag == 0) { h11 = 1; h22 = 1; flag = -1; }
				else           { h21 = -1; h12 = 1; flag = -1; }
				if (std::abs(*d2) <= rgamsq) { *d2 = *d2 * gamsq; h21 = h21 / gam; h22 = h22 / gam; }
				else                    { *d2 = *d2 / gamsq; h21 = h21 * gam; h22 = h22 * gam; }
			}
		}
	}

	param[0] = flag;
	if (flag < 0)
	{
		param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22;
	}
	else if (flag == 0)
	{
		param[2] = h21; param[3] = h12;
	}
	else
	{
		param[1] = h11; param[4] = h22;
	}
}


// ----------------------------------------------------------
//   ROTM
// ----------------------------------------------------------

// (x, y) := H (x, y)
template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
skepu::multiple<TX, TY> rotm_uf(TX x, TY y, TS h11, TS h21, TS h12, TS h22)
{
	TS new_x = h11 * x + h12 * y;
	TS new_y = h21 * x + h22 * y;
	return skepu::ret(new_x, new_y);
}

// Applies the modified Givens rotation that rotmg sets up (param has 5 elements)
template<typename TX, typename TY, typename TS = scalar_type<TX, TY>>
void rotm(
	size_type                        n,
	Vector<TX> &                     x,
	stride_type                      incx,
	Vector<TY> &                     y,
	stride_type                      incy,
	const TS *                       param
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.rotm");
	if (n == 0 || param[0] == -2)
		return;
	TS h11 = param[1], h21 = param[2], h12 = param[3], h22 = param[4];
	if (param[0] == 0)
	{
		h11 = 1; h22 = 1;
	}
	else if (param[0] == 1)
	{
		h21 = -1; h12 = 1;
	}
	vec_view<TX> vx("rotm", "x", x, n, incx, true, true);
	vec_view<TY> vy("rotm", "y", y, n, incy, true, true);
	auto skel = Map<2>(rotm_uf<TX, TY>);
	skel.setStride(vx.stride(), vy.stride(), vx.stride(), vy.stride());
	skel.setLabel("ROTM");
	skel(vx.vec(), vy.vec(), vx.vec(), vy.vec(), h11, h21, h12, h22);
	vx.commit();
	vy.commit();
}





// ================================================================================
// ================================================================================
// ==========================       LEVEL 2 BLAS
// ================================================================================
// ================================================================================


enum class Op     : char { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper    = 'U', Lower    = 'L', General   = 'G' };
enum class Diag   : char { NonUnit  = 'N', Unit     = 'U' };
enum class Side   : char { Left     = 'L', Right    = 'R' };


// ----------------------------------------------------------
//   GEMV
// ----------------------------------------------------------

// y(i) := alpha * sum_l op(A)(i, l) * x(l) + beta * y(i), where len is the number of terms and the row (or column) of A is the
// one of y(i). y is not read if beta is zero (so that a not-a-number in it does not go to the result), as in the reference BLAS.
template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_notrans(TY y, MatRow<TA> Arow, Vec<TX> x, stride_type incx, TS alpha, TS beta, size_type len)
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < len; ++l)
		tmp += x(l * incx) * Arow(l);
	const TS zero = 0;
	return (beta == zero) ? alpha * tmp : alpha * tmp + beta * y;
}

template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_trans(TY y, MatCol<TA> Acol, Vec<TX> x, stride_type incx, TS alpha, TS beta, size_type len)
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < len; ++l)
		tmp += x(l * incx) * Acol(l);
	const TS zero = 0;
	return (beta == zero) ? alpha * tmp : alpha * tmp + beta * y;
}

template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
TY gemv_uf_conj(TY y, MatCol<TA> Acol, Vec<TX> x, stride_type incx, TS alpha, TS beta, size_type len)
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < len; ++l)
		tmp += x(l * incx) * skepu::complex::conj(Acol(l));
	const TS zero = 0;
	return (beta == zero) ? alpha * tmp : alpha * tmp + beta * y;
}


template<typename TY, typename TS>
TY gemv_scale_uf(TY y, TS beta)
{
	const TS zero = 0;
	return (beta == zero) ? (TY)0 : (TY)(beta * y);
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.gemv");
	const TS zero = 0, one  = 1;

	// quick return
	if (m == 0 || n == 0 || (alpha == zero && beta == one))
		return;

	const bool t = (trans != Op::NoTrans);
	const size_type lenx = t ? m : n, leny = t ? n : m;
	check_matrix("gemv", "A", A, m, n, lda);

	vec_view<TX> vx("gemv", "x", x, lenx, incx, !(alpha == zero), false);
	vec_view<TY> vy("gemv", "y", y, leny, incy, !(beta == zero), true);

	if (alpha == zero)
	{
		// (the terms are not read: only y is scaled)
		auto scale = Map<1>(gemv_scale_uf<TY, TS>);
		scale.setStride(vy.stride(), vy.stride());
		scale.setLabel("GEMV scale");
		scale(vy.vec(), vy.vec(), beta);
	}
	else if (trans == Op::NoTrans)
	{
		auto skel = Map<1>(gemv_uf_notrans<TA, TX, TY>);
		skel.setStride(vy.stride(), vy.stride());
		skel.setLabel("GEMV");
		skel(vy.vec(), vy.vec(), A, vx.vec(), vx.stride(), alpha, beta, lenx);
	}
	else if (trans == Op::Trans)
	{
		auto skel = Map<1>(gemv_uf_trans<TA, TX, TY>);
		skel.setStride(vy.stride(), vy.stride());
		skel.setLabel("GEMV");
		skel(vy.vec(), vy.vec(), A, vx.vec(), vx.stride(), alpha, beta, lenx);
	}
	else
	{
		auto skel = Map<1>(gemv_uf_conj<TA, TX, TY>);
		skel.setStride(vy.stride(), vy.stride());
		skel.setLabel("GEMV");
		skel(vy.vec(), vy.vec(), A, vx.vec(), vx.stride(), alpha, beta, lenx);
	}
	vy.commit();
}


// ----------------------------------------------------------
//   GER, GERU
// ----------------------------------------------------------

// A(i, j) := A(i, j) + alpha * x(i) * y(j), in the leading m x n part of A, with the conjugate of y(j) for ger
template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
TA ger_uf(skepu::Index2D index, TA a, const skepu::Vec<TX> x, const skepu::Vec<TY> y, stride_type incx, stride_type incy, TS alpha, size_type m, size_type n, int conjugate)
{
	if (index.row >= m || index.col >= n)
		return a;
	TY yj = y(index.col * incy);
	if (conjugate)
		yj = skepu::complex::conj(yj);
	return a + alpha * x(index.row * incx) * yj;
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.ger");
	const TS zero = 0;
	if (m == 0 || n == 0 || alpha == zero)
		return;
	check_matrix("ger", "A", A, m, n, lda);
	vec_view<TX> vx("ger", "x", x, m, incx, true, false);
	vec_view<TY> vy("ger", "y", y, n, incy, true, false);

	auto skel = Map<1>(ger_uf<TX, TY, TA>);
	skel.setLabel("GER");
	skel(A, A, vx.vec(), vy.vec(), vx.stride(), vy.stride(), alpha, m, n, 1);
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.geru");
	const TS zero = 0;
	if (m == 0 || n == 0 || alpha == zero)
		return;
	check_matrix("geru", "A", A, m, n, lda);
	vec_view<TX> vx("geru", "x", x, m, incx, true, false);
	vec_view<TY> vy("geru", "y", y, n, incy, true, false);

	auto skel = Map<1>(ger_uf<TX, TY, TA>);
	skel.setLabel("GERU");
	skel(A, A, vx.vec(), vy.vec(), vx.stride(), vy.stride(), alpha, m, n, 0);
}





// ----------------------------------------------------------
//   Matrices that are stored in a triangle: SYMV, HEMV, TRMV, TRSV
// ----------------------------------------------------------

inline void check_uplo(const char *routine, Uplo uplo)
{
	if (uplo != Uplo::Upper && uplo != Uplo::Lower)
		SKEPU_ERROR("blas::" << routine << ": uplo must be Upper or Lower");
}

// Element (r, c) of the matrix that is stored in a triangle of A (upper: r <= c, otherwise r >= c), where the matrix is
//   kind 0: symmetric (the other triangle is the transpose of the stored one),
//   kind 1: Hermitian (the conjugate transpose, and a real diagonal),
//   kind 2: triangular (zero in the other triangle, and a diagonal of ones if unit).
template<typename T>
T triangle_elem(const skepu::Mat<T> A, size_type r, size_type c, int kind, int upper, int unit)
{
	const bool inside = upper ? (r <= c) : (r >= c);
	T v = 0;
	if (kind == 2)
	{
		if (inside)
			v = (unit && r == c) ? (T)1 : A(r, c);
	}
	else if (inside)
	{
		v = A(r, c);
	}
	else
	{
		v = A(c, r);
		if (kind == 1)
			v = skepu::complex::conj(v);
	}
	if (kind == 1 && r == c)
		v = skepu::complex::real(v);
	return v;
}

// y(i) := alpha * sum_l op(M)(i, l) * x(l) + beta * y(i) for a matrix M that is stored in a triangle of A (see triangle_elem). op is
// 0, 1 or 2 (none, transpose or conjugate transpose), used for a triangular matrix. y is not read if beta is zero.
template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
TY triangle_mv_uf(skepu::Index1D index, TY y, const skepu::Mat<TA> A, const skepu::Vec<TX> x, stride_type incx, TS alpha, TS beta, size_type len, int kind, int upper, int unit, int op)
{
	TS tmp = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < len; ++l)
	{
		TA a = (op == 0) ? triangle_elem<TA>(A, index.i, l, kind, upper, unit) : triangle_elem<TA>(A, l, index.i, kind, upper, unit);
		if (op == 2)
			a = skepu::complex::conj(a);
		tmp += x(l * incx) * a;
	}
	const TS zero = 0;
	return (beta == zero) ? alpha * tmp : alpha * tmp + beta * y;
}

// y := alpha * M * x + beta * y, with M stored in the triangle of A (the kind is symmetric or Hermitian)
template<typename TA, typename TX, typename TY, typename TS>
void triangle_mv(const char *routine, int kind, Uplo uplo, size_type n, TS alpha, Matrix<TA> &A, size_type lda, Vector<TX> &x, stride_type incx, TS beta, Vector<TY> &y, stride_type incy)
{
	check_uplo(routine, uplo);
	const TS zero = 0, one = 1;
	if (n == 0 || (alpha == zero && beta == one))
		return;
	check_matrix(routine, "A", A, n, n, lda);

	vec_view<TX> vx(routine, "x", x, n, incx, !(alpha == zero), false);
	vec_view<TY> vy(routine, "y", y, n, incy, !(beta == zero), true);
	if (alpha == zero)
	{
		auto scale = Map<1>(gemv_scale_uf<TY, TS>);
		scale.setStride(vy.stride(), vy.stride());
		scale.setLabel("scale");
		scale(vy.vec(), vy.vec(), beta);
	}
	else
	{
		auto skel = Map<1>(triangle_mv_uf<TA, TX, TY>);
		skel.setStride(vy.stride(), vy.stride());
		skel.setLabel(kind == 1 ? "HEMV" : "SYMV");
		skel(vy.vec(), vy.vec(), A, vx.vec(), vx.stride(), alpha, beta, n, kind, uplo == Uplo::Upper, 0, 0);
	}
	vy.commit();
}

// y := alpha * A * x + beta * y for a symmetric A, stored in a triangle
template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
void symv(
	Uplo                             uplo,
	size_type                        n,
	TS                               alpha,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	TS                               beta,
	Vector<TY> &                     y,
	stride_type                      incy
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.symv");
	triangle_mv("symv", 0, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

// y := alpha * A * x + beta * y for a Hermitian A, stored in a triangle
template<typename TA, typename TX, typename TY, typename TS = scalar_type<TA, TX, TY>>
void hemv(
	Uplo                             uplo,
	size_type                        n,
	TS                               alpha,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Vector<TX> BLAS_CONST&           x,
	stride_type                      incx,
	TS                               beta,
	Vector<TY> &                     y,
	stride_type                      incy
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.hemv");
	triangle_mv("hemv", 1, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

// x := op(A) * x for a triangular A
template<typename TA, typename TX>
void trmv(
	Uplo                             uplo,
	blas::Op                         trans,
	Diag                             diag,
	size_type                        n,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Vector<TX> &                     x,
	stride_type                      incx
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.trmv");
	using TS = scalar_type<TA, TX>;
	check_uplo("trmv", uplo);
	if (n == 0)
		return;
	check_matrix("trmv", "A", A, n, n, lda);

	// (the product goes to a new vector, since each element of it needs all of x)
	Vector<TX> result(n);
	vec_view<TX> vx("trmv", "x", x, n, incx, true, false);
	auto skel = Map<1>(triangle_mv_uf<TA, TX, TX>);
	skel.setStride(1, 1);
	skel.setLabel("TRMV");
	skel(result, result, A, vx.vec(), vx.stride(), (TS)1, (TS)0, n, 2, uplo == Uplo::Upper, diag == Diag::Unit,
		(trans == Op::NoTrans) ? 0 : (trans == Op::Trans ? 1 : 2));
	copy(n, result, 1, x, incx);
}

// One step of a substitution: the element `step` of x is solved for, from the elements before (if forward) or after it that have
// been solved for, and the others are as they were. The coefficients of the row are those of op(A) (A is triangular, and the
// triangle that is used is the one of op(A)).
template<typename TA, typename TX, typename TS = scalar_type<TA, TX>>
TX trsv_step_uf(skepu::Index1D index, TX old, const skepu::Vec<TX> prev, const skepu::Mat<TA> A, size_type n, size_type step, int forward, int unit, int op)
{
	if (index.i != step)
		return old;
	TS sum = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF j = 0; j < n; ++j)
	{
		if (forward ? (j < step) : (j > step))
		{
			TA a = (op == 0) ? A(step, j) : A(j, step);
			if (op == 2)
				a = skepu::complex::conj(a);
			sum += a * prev(j);
		}
	}
	TA diagonal = unit ? (TA)1 : A(step, step);
	if (op == 2)
		diagonal = skepu::complex::conj(diagonal);
	return (old - sum) / diagonal;
}

// Solves op(A) * x = b for x (x holds b, and then the solution), for a triangular A. It is a substitution, one step for each element
// of x: each step is a Map that makes the new x from the old (the elements that are solved for do not change).
template<typename TA, typename TX>
void trsv(
	Uplo                             uplo,
	blas::Op                         trans,
	Diag                             diag,
	size_type                        n,
	Matrix<TA> BLAS_CONST&           A,
	size_type                        lda,
	Vector<TX> &                     x,
	stride_type                      incx
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.trsv");
	check_uplo("trsv", uplo);
	if (n == 0)
		return;
	check_matrix("trsv", "A", A, n, n, lda);
	check_vector("trsv", "x", x.size(), n, incx);

	const int op = (trans == Op::NoTrans) ? 0 : (trans == Op::Trans ? 1 : 2);
	// op(A) is lower triangular: the first element is solved for first
	const bool forward = (uplo == Uplo::Lower) == (trans == Op::NoTrans);

	Vector<TX> w0(n), w1(n);
	Vector<TX> *cur = &w0, *next = &w1;
	copy(n, x, incx, w0, 1);
	auto skel = Map<1>(trsv_step_uf<TA, TX>);
	skel.setLabel("TRSV step");
	for (size_type s = 0; s < n; ++s)
	{
		skel(*next, *cur, *cur, A, n, forward ? s : n - 1 - s, forward, diag == Diag::Unit, op);
		std::swap(cur, next);
	}
	copy(n, *cur, 1, x, incx);
}


// ----------------------------------------------------------
//   SYR, HER, SYR2, HER2
// ----------------------------------------------------------

// A := A + alpha * x * x^T (kind 0, symmetric) or alpha * x * x^H (kind 1, Hermitian), and with two vectors
//   A := A + alpha * (x * y^T + y * x^T)  or  alpha * x * y^H + conj(alpha) * y * x^H,
// in the triangle of A only (and a real diagonal for Hermitian)
template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
TA rank_update_uf(skepu::Index2D index, TA a, const skepu::Vec<TX> x, const skepu::Vec<TY> y, stride_type incx, stride_type incy, TS alpha, size_type n, int kind, int upper, int two)
{
	const size_type i = index.row, j = index.col;
	if (i >= n || j >= n || (upper ? (i > j) : (i < j)))
		return a;
	TS v;
	if (!two)
	{
		v = alpha * x(i * incx) * ((kind == 1) ? skepu::complex::conj(x(j * incx)) : x(j * incx));
	}
	else if (kind == 0)
	{
		v = alpha * (x(i * incx) * y(j * incy) + y(i * incy) * x(j * incx));
	}
	else
	{
		v = alpha * x(i * incx) * skepu::complex::conj(y(j * incy)) + skepu::complex::conj(alpha) * y(i * incy) * skepu::complex::conj(x(j * incx));
	}
	TA res = a + v;
	if (kind == 1 && i == j)
		res = skepu::complex::real(res);
	return res;
}

template<typename TX, typename TY, typename TA, typename TS>
void rank_update(const char *routine, int kind, bool two, Uplo uplo, size_type n, TS alpha, Vector<TX> &x, stride_type incx, Vector<TY> &y, stride_type incy, Matrix<TA> &A, size_type lda)
{
	check_uplo(routine, uplo);
	const TS zero = 0;
	if (n == 0 || alpha == zero)
		return;
	check_matrix(routine, "A", A, n, n, lda);
	vec_view<TX> vx(routine, "x", x, n, incx, true, false);
	vec_view<TY> vy(routine, "y", y, n, incy, true, false);
	auto skel = Map<1>(rank_update_uf<TX, TY, TA>);
	skel.setLabel(routine);
	skel(A, A, vx.vec(), vy.vec(), vx.stride(), vy.stride(), alpha, n, kind, uplo == Uplo::Upper, two);
}

// A := A + alpha * x * x^T for a symmetric A, stored in a triangle
template<typename TX, typename TA, typename TS = scalar_type<TA, TX>>
void syr(Uplo uplo, size_type n, TS alpha, Vector<TX> BLAS_CONST& x, stride_type incx, Matrix<TA> &A, size_type lda SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.syr");
	rank_update("syr", 0, false, uplo, n, alpha, x, incx, x, incx, A, lda);
}

// A := A + alpha * x * x^H for a Hermitian A, stored in a triangle (alpha is real)
template<typename TX, typename TA, typename TS = real_type<TA, TX>>
void her(Uplo uplo, size_type n, TS alpha, Vector<TX> BLAS_CONST& x, stride_type incx, Matrix<TA> &A, size_type lda SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.her");
	rank_update("her", 1, false, uplo, n, alpha, x, incx, x, incx, A, lda);
}

// A := A + alpha * (x * y^T + y * x^T) for a symmetric A, stored in a triangle
template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
void syr2(Uplo uplo, size_type n, TS alpha, Vector<TX> BLAS_CONST& x, stride_type incx, Vector<TY> BLAS_CONST& y, stride_type incy, Matrix<TA> &A, size_type lda SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.syr2");
	rank_update("syr2", 0, true, uplo, n, alpha, x, incx, y, incy, A, lda);
}

// A := A + alpha * x * y^H + conj(alpha) * y * x^H for a Hermitian A, stored in a triangle
template<typename TX, typename TY, typename TA, typename TS = scalar_type<TA, TX, TY>>
void her2(Uplo uplo, size_type n, TS alpha, Vector<TX> BLAS_CONST& x, stride_type incx, Vector<TY> BLAS_CONST& y, stride_type incy, Matrix<TA> &A, size_type lda SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.her2");
	rank_update("her2", 1, true, uplo, n, alpha, x, incx, y, incy, A, lda);
}




// ================================================================================
// ================================================================================
// ==========================       LEVEL 3 BLAS
// ================================================================================
// ================================================================================


// ----------------------------------------------------------
//   GEMM
// ----------------------------------------------------------

// C(i, j) := alpha * sum_l op(A)(i, l) * op(B)(l, j) + beta * C(i, j), in the leading m x n part of C
// op is 0 (no transpose), 1 (transpose) or 2 (conjugate transpose); the elements of A and B
// are read through random access proxies since a row of a transposed matrix is not the
// row (or column) that goes with the output element. The terms are not read if alpha is zero (or k is), and C is not read if
// beta is zero, as in the reference BLAS.
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC gemm_uf(skepu::Index2D index, TC c, const skepu::Mat<TA> A, const skepu::Mat<TB> B, size_type k, int opa, int opb, TS alpha, TS beta, size_type m, size_type n)
{
	if (index.row >= m || index.col >= n)
		return c;
	const TS zero = 0;
	TS out = 0;
	if (!(alpha == zero))
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
		out = alpha * res;
	}
	if (!(beta == zero))
		out += beta * c;
	return out;
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
	SKEPU_CALL_SITE_PARAM
)
{
	SKEPU_TRACE_API("blas.gemm");

	// quick return
	if (m == 0 || n == 0)
		return;

	check_matrix("gemm", "A", A, (transA == Op::NoTrans) ? m : k, (transA == Op::NoTrans) ? k : m, lda);
	check_matrix("gemm", "B", B, (transB == Op::NoTrans) ? k : n, (transB == Op::NoTrans) ? n : k, ldb);
	check_matrix("gemm", "C", C, m, n, ldc);

	auto skel = skepu::Map<1>(gemm_uf<TA, TB, TC>);
	skel.setLabel("GEMM");
	const int opa = (transA == Op::NoTrans) ? 0 : (transA == Op::Trans ? 1 : 2);
	const int opb = (transB == Op::NoTrans) ? 0 : (transB == Op::Trans ? 1 : 2);
	skel(C, C, A, B, k, opa, opb, alpha, beta, m, n);
}

// ----------------------------------------------------------
//   SYMM, HEMM
// ----------------------------------------------------------

// C := alpha * M * B + beta * C (side left, M is m x m) or alpha * B * M + beta * C (side right, M is n x n), where M is
// symmetric (kind 0) or Hermitian (kind 1) and stored in a triangle of A
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC symm_uf(skepu::Index2D index, TC c, const skepu::Mat<TA> A, const skepu::Mat<TB> B, size_type k, TS alpha, TS beta, size_type m, size_type n, int right, int kind, int upper)
{
	if (index.row >= m || index.col >= n)
		return c;
	TS sum = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < k; ++l)
	{
		if (right)
			sum += B(index.row, l) * triangle_elem<TA>(A, l, index.col, kind, upper, 0);
		else
			sum += triangle_elem<TA>(A, index.row, l, kind, upper, 0) * B(l, index.col);
	}
	const TS zero = 0;
	return (beta == zero) ? alpha * sum : alpha * sum + beta * c;
}

template<typename TA, typename TB, typename TC, typename TS>
void symmetric_multiply(const char *routine, int kind, Side side, Uplo uplo, size_type m, size_type n, TS alpha, Matrix<TA> &A, size_type lda, Matrix<TB> &B, size_type ldb, TS beta, Matrix<TC> &C, size_type ldc)
{
	check_uplo(routine, uplo);
	if (m == 0 || n == 0)
		return;
	const size_type k = (side == Side::Left) ? m : n;
	check_matrix(routine, "A", A, k, k, lda);
	check_matrix(routine, "B", B, m, n, ldb);
	check_matrix(routine, "C", C, m, n, ldc);
	auto skel = Map<1>(symm_uf<TA, TB, TC>);
	skel.setLabel(routine);
	skel(C, C, A, B, k, alpha, beta, m, n, side == Side::Right, kind, uplo == Uplo::Upper);
}

// C := alpha * A * B + beta * C (left) or alpha * B * A + beta * C (right) for a symmetric A, stored in a triangle
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
void symm(Side side, Uplo uplo, size_type m, size_type n, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, Matrix<TB> BLAS_CONST& B, size_type ldb, TS beta, Matrix<TC> &C, size_type ldc SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.symm");
	symmetric_multiply("symm", 0, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

// The same for a Hermitian A
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
void hemm(Side side, Uplo uplo, size_type m, size_type n, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, Matrix<TB> BLAS_CONST& B, size_type ldb, TS beta, Matrix<TC> &C, size_type ldc SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.hemm");
	symmetric_multiply("hemm", 1, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}


// ----------------------------------------------------------
//   SYRK, HERK, SYR2K, HER2K
// ----------------------------------------------------------

// The triangle of C (n x n) := alpha * op(A) * op(B)^T + (alpha * op(B) * op(A)^T for two matrices) + beta * C, where op is the
// transpose if trans is nonzero, and for kind 1 (Hermitian) the second factor is conjugated, so that the products are
// op(A) * op(B)^H and conj(alpha) * op(B) * op(A)^H, and the diagonal is real
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
TC rank_k_uf(skepu::Index2D index, TC c, const skepu::Mat<TA> A, const skepu::Mat<TB> B, size_type k, TS alpha, TS beta, size_type n, int trans, int kind, int upper, int two)
{
	const size_type i = index.row, j = index.col;
	if (i >= n || j >= n || (upper ? (i > j) : (i < j)))
		return c;
	TS s1 = 0, s2 = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < k; ++l)
	{
		const TA ai = trans ? A(l, i) : A(i, l);
		const TB bj = trans ? B(l, j) : B(j, l);
		s1 += ai * ((kind == 1) ? skepu::complex::conj(bj) : bj);
		if (two)
		{
			const TB bi = trans ? B(l, i) : B(i, l);
			const TA aj = trans ? A(l, j) : A(j, l);
			s2 += bi * ((kind == 1) ? skepu::complex::conj(aj) : aj);
		}
	}
	const TS zero = 0;
	TS res = alpha * s1;
	if (two)
		res = res + ((kind == 1) ? skepu::complex::conj(alpha) : alpha) * s2;
	if (!(beta == zero))
		res = res + beta * c;
	if (kind == 1 && i == j)
		res = skepu::complex::real(res);
	return res;
}

template<typename TA, typename TB, typename TC, typename TS>
void rank_k(const char *routine, int kind, bool two, Uplo uplo, Op trans, size_type n, size_type k, TS alpha, Matrix<TA> &A, size_type lda, Matrix<TB> &B, size_type ldb, TS beta, Matrix<TC> &C, size_type ldc)
{
	check_uplo(routine, uplo);
	if (kind == 1 && trans == Op::Trans)
		SKEPU_ERROR("blas::" << routine << ": trans must be NoTrans or ConjTrans");
	if (n == 0)
		return;
	const bool t = (trans != Op::NoTrans); // (for a symmetric update ConjTrans is the same as Trans for real matrices)
	check_matrix(routine, "A", A, t ? k : n, t ? n : k, lda);
	if (two)
		check_matrix(routine, "B", B, t ? k : n, t ? n : k, ldb);
	check_matrix(routine, "C", C, n, n, ldc);
	auto skel = Map<1>(rank_k_uf<TA, TB, TC>);
	skel.setLabel(routine);
	skel(C, C, A, B, k, alpha, beta, n, t, kind, uplo == Uplo::Upper, two);
}

// The triangle of C := alpha * A * A^T + beta * C (trans NoTrans, A is n x k) or alpha * A^T * A + beta * C (A is k x n)
template<typename TA, typename TC, typename TS = scalar_type<TA, TC>>
void syrk(Uplo uplo, Op trans, size_type n, size_type k, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, TS beta, Matrix<TC> &C, size_type ldc SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.syrk");
	rank_k("syrk", 0, false, uplo, trans, n, k, alpha, A, lda, A, lda, beta, C, ldc);
}

// The same with A * A^H or A^H * A (trans is NoTrans or ConjTrans), for a Hermitian C (alpha and beta are real)
template<typename TA, typename TC, typename TS = real_type<TA, TC>>
void herk(Uplo uplo, Op trans, size_type n, size_type k, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, TS beta, Matrix<TC> &C, size_type ldc SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.herk");
	rank_k("herk", 1, false, uplo, trans, n, k, alpha, A, lda, A, lda, beta, C, ldc);
}

// The triangle of C := alpha * A * B^T + alpha * B * A^T + beta * C (or with A^T * B + B^T * A)
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
void syr2k(Uplo uplo, Op trans, size_type n, size_type k, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, Matrix<TB> BLAS_CONST& B, size_type ldb, TS beta, Matrix<TC> &C, size_type ldc SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.syr2k");
	rank_k("syr2k", 0, true, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// The triangle of C := alpha * A * B^H + conj(alpha) * B * A^H + beta * C (or with A^H * B + B^H * A), for a Hermitian C (beta is real)
template<typename TA, typename TB, typename TC, typename TS = scalar_type<TA, TB, TC>>
void her2k(Uplo uplo, Op trans, size_type n, size_type k, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, Matrix<TB> BLAS_CONST& B, size_type ldb, real_type<TS> beta, Matrix<TC> &C, size_type ldc SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.her2k");
	rank_k("her2k", 1, true, uplo, trans, n, k, alpha, A, lda, B, ldb, (TS)beta, C, ldc);
}


// ----------------------------------------------------------
//   TRMM
// ----------------------------------------------------------

// R(i, j) := alpha * sum_l op(T)(i, l) * B(l, j) (left) or alpha * sum_l B(i, l) * op(T)(l, j) (right), for a triangular T
template<typename TA, typename TB, typename TS = scalar_type<TA, TB>>
TB trmm_uf(skepu::Index2D index, const skepu::Mat<TB> B, const skepu::Mat<TA> A, size_type k, TS alpha, int right, int upper, int unit, int op)
{
	TS sum = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < k; ++l)
	{
		// op(T)(r, c)
		const size_type r = right ? l : index.row, c = right ? index.col : l;
		TA t = (op == 0) ? triangle_elem<TA>(A, r, c, 2, upper, unit) : triangle_elem<TA>(A, c, r, 2, upper, unit);
		if (op == 2)
			t = skepu::complex::conj(t);
		sum += right ? B(index.row, l) * t : t * B(l, index.col);
	}
	return alpha * sum;
}

// The leading m x n part of a matrix is replaced by the leading m x n part of another
template<typename T>
T replace_part_uf(skepu::Index2D index, T old, const skepu::Mat<T> part, size_type m, size_type n)
{
	return (index.row < m && index.col < n) ? part(index.row, index.col) : old;
}

// B := alpha * op(A) * B (left) or alpha * B * op(A) (right) for a triangular A
template<typename TA, typename TB, typename TS = scalar_type<TA, TB>>
void trmm(Side side, Uplo uplo, blas::Op trans, Diag diag, size_type m, size_type n, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, Matrix<TB> &B, size_type ldb SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.trmm");
	check_uplo("trmm", uplo);
	if (m == 0 || n == 0)
		return;
	const size_type k = (side == Side::Left) ? m : n;
	check_matrix("trmm", "A", A, k, k, lda);
	check_matrix("trmm", "B", B, m, n, ldb);

	// (the product goes to a new matrix, since each element of it needs a whole row or column of B)
	Matrix<TB> result(m, n);
	auto skel = Map<0>(trmm_uf<TA, TB>);
	skel.setLabel("TRMM");
	skel(result, B, A, k, alpha, side == Side::Right, uplo == Uplo::Upper, diag == Diag::Unit,
		(trans == Op::NoTrans) ? 0 : (trans == Op::Trans ? 1 : 2));

	auto put = Map<1>(replace_part_uf<TB>);
	put.setLabel("TRMM result");
	put(B, B, result, m, n);
}


// ----------------------------------------------------------
//   TRSM
// ----------------------------------------------------------

// The start of a triangular solve: R := alpha * B (the leading m x n part)
template<typename TB, typename TS>
TB trsm_scale_uf(skepu::Index2D index, const skepu::Mat<TB> B, TS alpha)
{
	const TS zero = 0;
	return (alpha == zero) ? (TB)0 : (TB)(alpha * B(index.row, index.col));
}

// One step of a substitution for op(A) * X = R (left) or X * op(A) = R (right): the row (left) or column (right) `step` of X is
// solved for, from the rows or columns before (forward) or after it that have been solved for, and the rest is as it was.
template<typename TA, typename TB, typename TS = scalar_type<TA, TB>>
TB trsm_step_uf(skepu::Index2D index, TB old, const skepu::Mat<TB> prev, const skepu::Mat<TA> A, size_type count, size_type step, int right, int forward, int unit, int op)
{
	if (right ? (index.col != step) : (index.row != step))
		return old;
	TS sum = 0;
	for (SKEPU_BLAS_SIZE_TYPE_UF l = 0; l < count; ++l)
	{
		if (forward ? (l < step) : (l > step))
		{
			// op(A)(step, l) for the left side, op(A)(l, step) for the right
			const size_type r = right ? l : step, c = right ? step : l;
			TA a = (op == 0) ? A(r, c) : A(c, r);
			if (op == 2)
				a = skepu::complex::conj(a);
			sum += right ? prev(index.row, l) * a : a * prev(l, index.col);
		}
	}
	TA diagonal = unit ? (TA)1 : A(step, step);
	if (op == 2)
		diagonal = skepu::complex::conj(diagonal);
	return (old - sum) / diagonal;
}

// Solves op(A) * X = alpha * B (left) or X * op(A) = alpha * B (right) for X (B is replaced by it), for a triangular A. It is a
// substitution, one step for each row (left) or column (right) of X: each step is a Map that makes the new X from the old.
template<typename TA, typename TB, typename TS = scalar_type<TA, TB>>
void trsm(Side side, Uplo uplo, blas::Op trans, Diag diag, size_type m, size_type n, TS alpha, Matrix<TA> BLAS_CONST& A, size_type lda, Matrix<TB> &B, size_type ldb SKEPU_CALL_SITE_PARAM)
{
	SKEPU_TRACE_API("blas.trsm");
	check_uplo("trsm", uplo);
	if (m == 0 || n == 0)
		return;
	const size_type k = (side == Side::Left) ? m : n;
	check_matrix("trsm", "A", A, k, k, lda);
	check_matrix("trsm", "B", B, m, n, ldb);

	const bool right = (side == Side::Right);
	const int op = (trans == Op::NoTrans) ? 0 : (trans == Op::Trans ? 1 : 2);
	// For the left side the first row is solved for first if op(A) is lower triangular, for the right side if it is upper
	const bool op_lower = (uplo == Uplo::Lower) == (trans == Op::NoTrans);
	const bool forward = right ? !op_lower : op_lower;

	Matrix<TB> w0(m, n), w1(m, n);
	Matrix<TB> *cur = &w0, *next = &w1;
	auto scale = Map<0>(trsm_scale_uf<TB, TS>);
	scale.setLabel("TRSM scale");
	scale(w0, B, alpha);

	auto skel = Map<1>(trsm_step_uf<TA, TB>);
	skel.setLabel("TRSM step");
	for (size_type s = 0; s < k; ++s)
	{
		skel(*next, *cur, *cur, A, k, forward ? s : k - 1 - s, right, forward, diag == Diag::Unit, op);
		std::swap(cur, next);
	}

	auto put = Map<1>(replace_part_uf<TB>);
	put.setLabel("TRSM result");
	put(B, B, *cur, m, n);
}

#undef SKEPU_BLAS_STRIDE_TYPE_UF
#undef SKEPU_BLAS_SIZE_TYPE_UF
}}

static skepu::PrecompilerMarker endOfBlasHPP;
#endif // SKEPU_PRECOMPILED
