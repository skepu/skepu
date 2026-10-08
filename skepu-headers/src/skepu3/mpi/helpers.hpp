#pragma once

// A container of the same shape as another (with the same element type, or another), as in the node-level library

#include "skepu3/mpi/vector.hpp"
#include "skepu3/mpi/matrix.hpp"
#include "skepu3/mpi/tensor.hpp"

namespace skepu
{
	template<typename T>
	inline Vector<T> container_like(skepu::Vector<T> &c, std::string label = "")
	{
		return Vector<T>(c.size(), label);
	}

	template<typename T>
	inline Matrix<T> container_like(skepu::Matrix<T> &c, std::string label = "")
	{
		return Matrix<T>(c.size_i(), c.size_j(), label);
	}

	template<typename T>
	inline Tensor3<T> container_like(skepu::Tensor3<T> &c, std::string label = "")
	{
		return Tensor3<T>(c.size_i(), c.size_j(), c.size_k(), label);
	}

	template<typename T>
	inline Tensor4<T> container_like(skepu::Tensor4<T> &c, std::string label = "")
	{
		return Tensor4<T>(c.size_i(), c.size_j(), c.size_k(), c.size_l(), label);
	}

	template<typename T, typename V>
	inline Vector<T> container_like(skepu::Vector<V> &c, std::string label = "")
	{
		return Vector<T>(c.size(), label);
	}

	template<typename T, typename V>
	inline Matrix<T> container_like(skepu::Matrix<V> &c, std::string label = "")
	{
		return Matrix<T>(c.size_i(), c.size_j(), label);
	}

	template<typename T, typename V>
	inline Tensor3<T> container_like(skepu::Tensor3<V> &c, std::string label = "")
	{
		return Tensor3<T>(c.size_i(), c.size_j(), c.size_k(), label);
	}

	template<typename T, typename V>
	inline Tensor4<T> container_like(skepu::Tensor4<V> &c, std::string label = "")
	{
		return Tensor4<T>(c.size_i(), c.size_j(), c.size_k(), c.size_l(), label);
	}
}
