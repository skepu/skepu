#pragma once
// Helpers for the checked BLAS tests: host reference computations and comparisons.

#include <cmath>
#include <complex>
#include <vector>
#include <skepu>

namespace blas_check
{

inline bool close(double got, double want, double tol = 1e-4)
{
	return std::fabs(got - want) <= tol * std::max(1.0, std::fabs(want));
}

// Deterministic, non-trivial test values in a small range, no zeros.
inline float val(size_t i, size_t salt = 0)
{
	return 1.0f + ((i * 7 + salt * 13) % 11) * 0.25f - ((i + salt) % 3) * 1.5f;
}

template<typename V>
std::vector<float> to_host(V &v)
{
	v.updateHost();
	std::vector<float> h(v.size());
	for (size_t i = 0; i < h.size(); ++i)
		h[i] = v(i);
	return h;
}

inline void fill(skepu::Vector<float> &v, size_t salt)
{
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = val(i, salt);
}

inline void fill(skepu::Matrix<float> &m, size_t salt)
{
	for (size_t i = 0; i < m.size_i(); ++i)
		for (size_t j = 0; j < m.size_j(); ++j)
			m[i * m.size_j() + j] = val(i * m.size_j() + j, salt);
}

inline std::vector<float> to_host(skepu::Matrix<float> &m)
{
	m.updateHost();
	std::vector<float> h(m.size());
	for (size_t i = 0; i < m.size_i(); ++i)
		for (size_t j = 0; j < m.size_j(); ++j)
			h[i * m.size_j() + j] = m(i, j);
	return h;
}

// Host references for the matrices that are stored in a triangle: Z is double or std::complex<double>
namespace ref
{

inline double conj(double z) { return z; }
inline std::complex<double> conj(std::complex<double> z) { return std::conj(z); }
inline double real(double z) { return z; }
inline double real(std::complex<double> z) { return z.real(); }

template<typename Z, typename T>
std::vector<Z> convert(const std::vector<T> &v)
{
	std::vector<Z> r(v.size());
	for (size_t i = 0; i < v.size(); ++i)
		r[i] = Z(v[i]);
	return r;
}

// the n x n matrix that a triangle of A (row length ld) stands for:
// kind 0 symmetric, 1 Hermitian, 2 triangular (a unit diagonal if unit)
template<typename Z>
std::vector<Z> triangle(const std::vector<Z> &A, size_t ld, size_t n, int kind, bool upper, bool unit)
{
	std::vector<Z> M(n * n, Z(0));
	for (size_t r = 0; r < n; ++r)
		for (size_t c = 0; c < n; ++c)
		{
			const bool inside = upper ? (r <= c) : (r >= c);
			Z v = Z(0);
			if (kind == 2) { if (inside) v = (unit && r == c) ? Z(1) : A[r * ld + c]; }
			else if (inside) v = A[r * ld + c];
			else { v = A[c * ld + r]; if (kind == 1) v = conj(v); }
			if (kind == 1 && r == c) v = Z(real(v));
			M[r * n + c] = v;
		}
	return M;
}

// op(M) of an r x c matrix (op 0, 1 or 2: none, transpose, conjugate transpose)
template<typename Z>
std::vector<Z> opm(const std::vector<Z> &M, size_t r, size_t c, int op)
{
	if (op == 0)
		return M;
	std::vector<Z> R(r * c);
	for (size_t i = 0; i < r; ++i)
		for (size_t j = 0; j < c; ++j)
			R[j * r + i] = (op == 2) ? conj(M[i * c + j]) : M[i * c + j];
	return R;
}

// A (m x k) times B (k x n)
template<typename Z>
std::vector<Z> mul(const std::vector<Z> &A, size_t m, size_t k, const std::vector<Z> &B, size_t n)
{
	std::vector<Z> R(m * n, Z(0));
	for (size_t i = 0; i < m; ++i)
		for (size_t l = 0; l < k; ++l)
			for (size_t j = 0; j < n; ++j)
				R[i * n + j] += A[i * k + l] * B[l * n + j];
	return R;
}

}

}
