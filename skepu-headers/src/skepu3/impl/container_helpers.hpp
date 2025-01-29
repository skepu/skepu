#pragma once

namespace skepu
{	
	
  template<typename T>
  inline Vector<T> container_like(skepu::Vector<T> &c)
  {
    Vector<T> ret(c.size_i());
    return std::move(ret);
  }

  template<typename T>
  inline Matrix<T> container_like(skepu::Matrix<T> &c)
  {
    Matrix<T> ret(c.size_i(), c.size_j());
    return std::move(ret);
  }

  template<typename T>
  inline Tensor3<T> container_like(skepu::Tensor3<T> &c)
  {
    Tensor3<T> ret(c.size_i(), c.size_j(), c.size_k());
    return std::move(ret);
  }

  template<typename T>
  inline Tensor4<T> container_like(skepu::Tensor4<T> &c)
  {
    Tensor4<T> ret(c.size_i(), c.size_j(), c.size_k(), c.size_l());
    return std::move(ret);
  }
  
  
  
  template<typename T, typename V>
  inline Vector<T> container_like(skepu::Vector<V> &c)
  {
    Vector<T> ret(c.size_i());
    return std::move(ret);
  }
  
  template<typename T, typename V>
  inline Matrix<T> container_like(skepu::Matrix<V> &c)
  {
    Matrix<T> ret(c.size_i(), c.size_j());
    return std::move(ret);
  }
  
  template<typename T, typename V>
  inline Tensor3<T> container_like(skepu::Tensor3<V> &c)
  {
    Tensor3<T> ret(c.size_i(), c.size_j(), c.size_k());
    return std::move(ret);
  }

  template<typename T, typename V>
  inline Tensor4<T> container_like(skepu::Tensor4<V> &c)
  {
    Tensor4<T> ret(c.size_i(), c.size_j(), c.size_k(), c.size_l());
    return std::move(ret);
  }
  
}