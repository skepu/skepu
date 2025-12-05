#pragma once

namespace skepu
{	
	
  template<typename T>
  inline Vector<T> container_like(skepu::Vector<T> &c, std::string label = "")
  {
    Vector<T> ret(c.size_i(), label);
    return std::move(ret);
  }

  template<typename T>
  inline Matrix<T> container_like(skepu::Matrix<T> &c, std::string label = "")
  {
    Matrix<T> ret(c.size_i(), c.size_j(), label);
    return std::move(ret);
  }

  template<typename T>
  inline Tensor3<T> container_like(skepu::Tensor3<T> &c, std::string label = "")
  {
    Tensor3<T> ret(c.size_i(), c.size_j(), c.size_k(), label);
    return std::move(ret);
  }

  template<typename T>
  inline Tensor4<T> container_like(skepu::Tensor4<T> &c, std::string label = "")
  {
    Tensor4<T> ret(c.size_i(), c.size_j(), c.size_k(), c.size_l(), label);
    return std::move(ret);
  }
  
  
  
  template<typename T, typename V>
  inline Vector<T> container_like(skepu::Vector<V> &c, std::string label = "")
  {
    Vector<T> ret(c.size_i(), label);
    return std::move(ret);
  }
  
  template<typename T, typename V>
  inline Matrix<T> container_like(skepu::Matrix<V> &c, std::string label = "")
  {
    Matrix<T> ret(c.size_i(), c.size_j(), label);
    return std::move(ret);
  }
  
  template<typename T, typename V>
  inline Tensor3<T> container_like(skepu::Tensor3<V> &c, std::string label = "")
  {
    Tensor3<T> ret(c.size_i(), c.size_j(), c.size_k(), label);
    return std::move(ret);
  }

  template<typename T, typename V>
  inline Tensor4<T> container_like(skepu::Tensor4<V> &c, std::string label = "")
  {
    Tensor4<T> ret(c.size_i(), c.size_j(), c.size_k(), c.size_l(), label);
    return std::move(ret);
  }
  
}