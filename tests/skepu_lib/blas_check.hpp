#pragma once
// Helpers for the checked BLAS tests: host reference computations and comparisons.

#include <cmath>
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

}
