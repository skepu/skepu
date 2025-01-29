/*! \file vector.h
*  \brief Contains a class declaration for the Vector container.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <map>
#include <iomanip>

#include "backend/malloc_allocator.h"

#ifdef SKEPU_PRECOMPILED

#include "backend/environment.h"
#include "backend/device_mem_pointer_cl.h"
#include "backend/device_mem_pointer_cu.h"

#endif // SKEPU_PRECOMPILED


namespace skepu
{
	// TENSOR 3
	
	template<typename T> class Region3D;
	template<typename T> class Region4D;
	template<typename T> class Pool3D;
	template<typename T> class Pool4D;
	
	template<typename T>
	class Tensor3;
	
	// Proxy vector for user functions
	template<typename T>
	struct Ten3
	{
		using ContainerType = Tensor3<T>;
		
		Ten3(T *dataptr, size_t si, size_t sj, size_t sk)
		: data{dataptr},
		size_i{si}, size_j{sj}, size_k{sk},
		stride_1{sj * sk}, stride_2{sk}
		{}
		
		Ten3(): data{nullptr}, size_i{0}, size_j{0}, size_k{0}, stride_1{0}, stride_2{0} {}
		
		T &operator[](size_t index)       { return this->data[index]; }
		T  operator[](size_t index) const { return this->data[index]; }
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T &operator()(size_t i, size_t j, size_t k)       { return this->data[i * this->stride_1 + j * this->stride_2 + k]; }
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T  operator()(size_t i, size_t j, size_t k) const { return this->data[i * this->stride_1 + j * this->stride_2 + k]; }
		
		T *data;
		size_t size_i, size_j, size_k;
		size_t stride_1, stride_2;
	};
	
	
	template <typename T>
	class Tensor3Iterator;
	
	template<typename T>
	class Tensor3 : public Vector<T>
	{
	public:
		using size_type = typename Vector<T>::size_type;
		using proxy_type = Ten3<T>;
		
#ifdef SKEPU_CUDA
		using device_pointer_type_cu = typename Vector<T>::device_pointer_type_cu;
#endif
		
		typedef Tensor3Iterator<T> iterator;
		typedef Tensor3Iterator<const T> const_iterator;
		
		friend class Tensor3Iterator<T>;
		
		explicit Tensor3()
		: m_size_i(0), m_size_j(0), m_size_k(0),
		m_stride_1(0), m_stride_2(0),
		Vector<T>{}
		{}
			
		explicit Tensor3(size_type si, size_type sj, size_type sk, std::string label = "", const T& val = T())
		: m_size_i(si), m_size_j(sj), m_size_k(sk),
		m_stride_1(sj * sk), m_stride_2(sk),
		Vector<T>(si * sj * sk, label, val)
		{}
			
		// Construct from pointer
		Tensor3(T * const ptr, size_type si, size_type sj, size_type sk, bool deallocEnabled = true)
		: m_size_i(si), m_size_j(sj), m_size_k(sk),
		m_stride_1(sj * sk), m_stride_2(sk),
		Vector<T>(ptr, si * sj * sk, deallocEnabled)
		{}
		
		// Copy constructor	
		Tensor3(const Tensor3& c)
		: m_size_i(c.m_size_i), m_size_j(c.m_size_j), m_size_k(c.m_size_k),
		m_stride_1(c.m_stride_1), m_stride_2(c.m_stride_2),
		Vector<T>(c)
		{}
		
		// Move constructor	
		Tensor3(Tensor3&& c)
		: m_size_i(0), m_size_j(0), m_size_k(0),
		m_stride_1(0), m_stride_2(0),
		Vector<T>()
		{
			this->swap(c);
		}
			
		void swap(Tensor3<T>& from)
		{
			static_cast<Vector<T>&>(*this).swap(static_cast<Vector<T>&>(from));
			std::swap(m_size_i, from.m_size_i);
			std::swap(m_size_j, from.m_size_j);
			std::swap(m_size_k, from.m_size_k);
			std::swap(m_stride_1, from.m_stride_1);
			std::swap(m_stride_2, from.m_stride_2);
		}
	
		void init(size_type si, size_type sj, size_type sk)
		{
			Vector<T>::init(si * sj * sk);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->m_stride_1 = sj * sk;
			this->m_stride_2 = sk;
		}
		
		void init(size_type si, size_type sj, size_type sk, const T& val)
		{
			Vector<T>::init(si * sj * sk, val);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->m_stride_1 = sj * sk;
			this->m_stride_2 = sk;
		}
		
		iterator begin()
		{
			return iterator(*this, &Vector<T>::m_data[0]);
		}
		
		iterator end()
		{
			return iterator(*this, &Vector<T>::m_data[this->m_size]);
		}
		
		// These do nothing special for now
		iterator stridedBegin(size_t, int) { return this->begin(); };
		const_iterator begin(size_t, int) const
		{
			return const_iterator(*this, &Vector<T>::m_data[0]);
		}
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{
			return { this->m_data, this->m_size_i, this->m_size_j, this->m_size_k };
		}
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		size_type size() const
		{
			return Vector<T>::m_size;
		}
		
		size_type size_i() const
		{
			return m_size_i;
		}
		
		size_type size_j() const
		{
			return m_size_j;
		}
		
		size_type size_k() const
		{
			return m_size_k;
		}
		
		size_type size_l() const
		{
			return 0;
		}
		
		// All dimensions
		std::tuple<size_type, size_type, size_type, size_type> size_info() const
		{
			return {this->m_size_i, this->m_size_j, this->m_size_k, 1};
		}
		
		T& operator()(size_type i, size_type j, size_type k)
		{
			return Vector<T>::m_data[i * this->m_stride_1 + j * this->m_stride_2 + k];
		}
		
		const T& operator()(size_type i, size_type j, size_type k) const
		{
			return Vector<T>::m_data[i * this->m_stride_1 + j * this->m_stride_2 + k];
		}
		
		const Tensor3<T>& getParent() const { return *this; }
		Tensor3<T>& getParent() { return *this; }
		
#ifdef SKEPU_CUDA
		template<typename Ignore>
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::Default, Ignore)
		{
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->size(), deviceID, accessMode);
			proxy_type proxy;
			proxy.data = devptr->getDeviceDataPointer();
			proxy.size_i = this->m_size_i;
			proxy.size_j = this->m_size_j;
			proxy.size_k = this->m_size_k;
			return {devptr, proxy};
		}
		
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode)
		{
			return this->cudaProxy(deviceID, accessMode, ProxyTag::Default{}, 0);
		}
#endif // SKEPU_CUDA
		
		friend std::ostream& operator<<(std::ostream &os, Tensor3<T>& ten)
		{
			os << "Tensor3: (" << ten.size_i()
				<< " x " << ten.size_j()
				<< " x " << ten.size_k() << ")\n";
			
			for (int j = 0; j < ten.size_j(); ++j)
			{
				for (int i = 0; i < ten.size_i(); ++i)
				{
					for (int k = 0; k < ten.size_k(); ++k)
					{
						auto val = ten(i,j,k);
						std::cout << std::setw(5) << val << " ";
					}
					os << " | ";
				}
				os << "\n";
			}
			return os << "\n";
		}
		
		friend class Region3D<T>;
		friend class Pool3D<T>;
		
	private:
		
		size_type m_size_i, m_size_j, m_size_k;
		size_type m_stride_1, m_stride_2;
		
	};
	
	
	template <typename T>
	class Tensor3Iterator
	{
	public:
		typedef Tensor3Iterator<T> iterator;
		typedef Tensor3Iterator<const T> const_iterator;
	   typedef typename std::conditional<std::is_const<T>::value,
						const Tensor3<typename std::remove_const<T>::type>, Tensor3<T>>::type parent_type;
		
		using proxy_type = typename parent_type::proxy_type;
	
	public: //-- Constructors & Destructor --//
		
		Tensor3Iterator(parent_type& vec, T *std_iterator);
		
	public: //-- Types --//
		
#ifdef SKEPU_CUDA
		typedef typename parent_type::device_pointer_type_cu device_pointer_type_cu;
#endif
		
#ifdef SKEPU_OPENCL
		typedef typename parent_type::device_pointer_type_cl device_pointer_type_cl;
#endif
		
	public: //-- Extras --//
		
		Index3D getIndex() const;
		
		parent_type& getParent() const;
		iterator& begin(); // returns itself
		iterator stridedBegin(size_t, int) { return this->begin(); };
		size_t size(); // returns number of elements "left" in parent container from this index
		
		T* getAddress();
		T* data();
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{ return {this->m_std_iterator, this->size()}; }
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		// Does not care about device data, use with care...sometimes pass negative indices...
		T& operator()(const ssize_t index = 0);
		const T& operator()(const ssize_t index) const;
		
	public: //-- Operators --//
		
		T& operator[](const ssize_t index);
		const T& operator[](const ssize_t index) const;
		
		bool operator==(const iterator& i);
		bool operator!=(const iterator& i);
		bool operator<(const iterator& i);
		bool operator>(const iterator& i);
		bool operator<=(const iterator& i);
		bool operator>=(const iterator& i);
		
		const iterator& operator++();
		iterator operator++(int);
		const iterator& operator--();
		iterator operator--(int);
		
		const iterator& operator+=(const ssize_t i);
		const iterator& operator-=(const ssize_t i);
		
		iterator operator-(const ssize_t i) const;
		iterator operator+(const ssize_t i) const;
		
		typename parent_type::difference_type operator-(const iterator& i) const;
		
		T& operator *();
		const T& operator* () const;
		
		const T& operator-> () const;
		T& operator-> ();
		
	protected: //-- Data --//
		
		parent_type& m_parent;
		T *m_std_iterator;
	};
	
	
	
	
	
	
	// TENSOR 4
	
	template<typename T>
	class Tensor4;
	
	// Proxy vector for user functions
	template<typename T>
	struct Ten4
	{
		using ContainerType = Tensor4<T>;
		
		Ten4(T *dataptr, size_t si, size_t sj, size_t sk, size_t sl)
		: data{dataptr},
		size_i{si}, size_j{sj}, size_k{sk}, size_l{sl},
		stride_1{sj * sk * sl}, stride_2{sk * sl}, stride_3{sl}
		{}
		
		Ten4(): data{nullptr}, size_i{0}, size_j{0}, size_k{0}, size_l{0},
		stride_1{0}, stride_2{0}, stride_3{0}
		{}
		
		T &operator[](size_t index)       { return this->data[index]; }
		T  operator[](size_t index) const { return this->data[index]; }
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T &operator()(size_t i, size_t j, size_t k, size_t l)
		{ return this->data[i * this->stride_1 + j * this->stride_2 + k * this->stride_3 + l]; }
		
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T  operator()(size_t i, size_t j, size_t k, size_t l) const
		{ return this->data[i * this->stride_1 + j * this->stride_2 + k * this->stride_3 + l]; }
		
		T *data;
		size_t size_i, size_j, size_k, size_l;
		size_t stride_1, stride_2, stride_3;
	};
	
	
	template <typename T>
	class Tensor4Iterator;
	
	template<typename T>
	class Tensor4 : public Vector<T>
	{
	public:
		using size_type = typename Vector<T>::size_type;
		using proxy_type = Ten4<T>;
		
#ifdef SKEPU_CUDA
		using device_pointer_type_cu = typename Vector<T>::device_pointer_type_cu;
#endif
		
		typedef Tensor4Iterator<T> iterator;
		typedef Tensor4Iterator<const T> const_iterator;
		
		friend class Tensor4Iterator<T>;
		
		explicit Tensor4()
		: m_size_i(0), m_size_j(0), m_size_k(0), m_size_l(0),
		m_stride_1(0), m_stride_2(0), m_stride_3(0),
		Vector<T>()
		{}
		
		explicit Tensor4(size_type si, size_type sj, size_type sk, size_type sl, std::string label = "", const T& val = T())
		: m_size_i(si), m_size_j(sj), m_size_k(sk), m_size_l(sl),
		m_stride_1(sj * sk * sl), m_stride_2(sk * sl), m_stride_3(sl),
		Vector<T>(si * sj * sk * sl, label, val)
		{}
			
		// Construct from pointer
		Tensor4(T * const ptr, size_type si, size_type sj, size_type sk, size_type sl, bool deallocEnabled = true)
		: m_size_i(si), m_size_j(sj), m_size_k(sk), m_size_l(sl),
		m_stride_1(sj * sk * sl), m_stride_2(sk * sl), m_stride_3(sl),
		Vector<T>(ptr, si * sj * sk * sl, deallocEnabled)
		{}
		
		// Copy constructor	
		Tensor4(const Tensor4& c)
		: m_size_i(c.m_size_i), m_size_j(c.m_size_j), m_size_k(c.m_size_k), m_size_l(c.m_size_l),
		m_stride_1(c.m_stride_1), m_stride_2(c.m_stride_2), m_stride_3(c.m_stride_3),
		Vector<T>(c)
		{}
		
		// Move constructor	
		Tensor4(Tensor4&& c)
		: m_size_i(0), m_size_j(0), m_size_k(0), m_size_l(0),
		m_stride_1(0), m_stride_2(0), m_stride_3(0),
		Vector<T>()
		{
			this->swap(c);
		}

		Tensor4& operator=(Tensor4&& other)
		{
			this->swap(other);
			return *this;
		}
		
		void swap(Tensor4<T>& from)
		{
			static_cast<Vector<T>&>(*this).swap(static_cast<Vector<T>&>(from));
			std::swap(m_size_i, from.m_size_i);
			std::swap(m_size_j, from.m_size_j);
			std::swap(m_size_k, from.m_size_k);
			std::swap(m_size_l, from.m_size_l);
			std::swap(m_stride_1, from.m_stride_1);
			std::swap(m_stride_2, from.m_stride_2);
			std::swap(m_stride_3, from.m_stride_3);
		}
		
		void init(size_type si, size_type sj, size_type sk, size_type sl)
		{
			Vector<T>::init(si * sj * sk * sl);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->m_size_l = sl;
			this->m_stride_1 = sj * sk * sl;
			this->m_stride_2 = sk * sl;
			this->m_stride_3 = sl;
		}
		
		void init(size_type si, size_type sj, size_type sk, size_type sl, const T& val)
		{
			Vector<T>::init(si * sj * sk * sl, val);
			this->m_size_i = si;
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->m_size_l = sl;
			this->m_stride_1 = sj * sk * sl;
			this->m_stride_2 = sk * sl;
			this->m_stride_3 = sl;
		}
		
		iterator begin()
		{
			return iterator(*this, &Vector<T>::m_data[0]);
		}
		
		iterator end()
		{
			return iterator(*this, &Vector<T>::m_data[this->m_size]);
		}
		
		// These do nothing special for now
		iterator stridedBegin(size_t, int) { return this->begin(); };
		const_iterator begin(size_t, int) const
		{
			return const_iterator(*this, &Vector<T>::m_data[0]);
		};
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{
			return { this->m_data, this->m_size_i, this->m_size_j, this->m_size_k, this->m_size_l };
		}
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		size_type size() const
		{
			return Vector<T>::m_size;
		}
		
		size_type size_i() const
		{
			return m_size_i;
		}
		
		size_type size_j() const
		{
			return m_size_j;
		}
		
		size_type size_k() const
		{
			return m_size_k;
		}
		
		size_type size_l() const
		{
			return m_size_l;
		}
		
		// All dimensions
		std::tuple<size_type, size_type, size_type, size_type> size_info() const
		{
			return {this->m_size_i, this->m_size_j, this->m_size_k, this->m_size_l};
		}
		
		T& operator()(size_type i, size_type j, size_type k, size_type l)
		{
			return Vector<T>::m_data[i * this->m_stride_1 + j * this->m_stride_2 + k * this->m_stride_3 + l];
		}
		
		const T& operator()(size_type i, size_type j, size_type k, size_type l) const
		{
			return Vector<T>::m_data[i * this->m_stride_1 + j * this->m_stride_2 + k * this->m_stride_3 + l];
		}
		
		const Tensor4<T>& getParent() const { return *this; }
		Tensor4<T>& getParent() { return *this; }
		
#ifdef SKEPU_CUDA
		template<typename Ignore>
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::Default, Ignore)
		{
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->size(), deviceID, accessMode);
			proxy_type proxy;
			proxy.data = devptr->getDeviceDataPointer();
			proxy.size_i = this->m_size_i;
			proxy.size_j = this->m_size_j;
			proxy.size_k = this->m_size_k;
			proxy.size_l = this->m_size_l;
			return {devptr, proxy};
		}
		
		std::pair<device_pointer_type_cu, proxy_type>
		cudaProxy(size_t deviceID, AccessMode accessMode)
		{
			return this->cudaProxy(deviceID, accessMode, ProxyTag::Default{}, 0);
		}
#endif // SKEPU_CUDA

		friend std::ostream& operator<<(std::ostream &os, Tensor4<T>& ten4)
		{
			ten4.updateHost();
			os << "Tensor4: (" << ten4.size_i() << " x " << ten4.size_j() << " x " << ten4.size_k() << " x " << ten4.size_l() << ")\n";
			
			for (size_t i = 0; i < ten4.size_i(); ++i)
			{
				for (size_t j = 0; j < ten4.size_j(); ++j)
				{
					for (size_t k = 0; k < ten4.size_k(); ++k)
					{
						for (size_t l = 0; l < ten4.size_l(); ++l)
						{
							std::cout << std::setw(12) << std::fixed << ten4(i,j,k,l) << " ";
						}
						os << "| ";
					}
					os << "\n";
				}
				os << "---------------------------------\n";
			}
			return os << "\n";
		}
		
		friend class Region4D<T>;
		friend class Pool4D<T>;
		
	private:
		
		size_type m_size_i, m_size_j, m_size_k, m_size_l;
		size_type m_stride_1, m_stride_2, m_stride_3;
		
	};
	
	
	
	template <typename T>
	class Tensor4Iterator
	{
	public:
		typedef Tensor4Iterator<T> iterator;
		typedef Tensor4Iterator<const T> const_iterator;
	   typedef typename std::conditional<std::is_const<T>::value,
						const Tensor4<typename std::remove_const<T>::type>, Tensor4<T>>::type parent_type;
		
		using proxy_type = typename parent_type::proxy_type;
	
	public: //-- Constructors & Destructor --//
		
		Tensor4Iterator(parent_type& vec, T *std_iterator);
		
	public: //-- Types --//
		
#ifdef SKEPU_CUDA
		typedef typename parent_type::device_pointer_type_cu device_pointer_type_cu;
#endif
		
#ifdef SKEPU_OPENCL
		typedef typename parent_type::device_pointer_type_cl device_pointer_type_cl;
#endif
		
	public: //-- Extras --//
		
		Index4D getIndex() const;
		
		parent_type& getParent() const;
		iterator stridedBegin(size_t, int) { return *this; };
		iterator& begin(); // returns itself
		size_t size(); // returns number of elements "left" in parent container from this index
		
		T* getAddress();
		T* data();
		
		template<typename Ignore>
		proxy_type hostProxy(ProxyTag::Default, Ignore)
		{ return {this->m_std_iterator, this->size()}; }
		
		proxy_type hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }
		
		// Does not care about device data, use with care...sometimes pass negative indices...
		T& operator()(const ssize_t index = 0);
		const T& operator()(const ssize_t index) const;
		
	public: //-- Operators --//
		
		T& operator[](const ssize_t index);
		const T& operator[](const ssize_t index) const;
		
		bool operator==(const iterator& i);
		bool operator!=(const iterator& i);
		bool operator<(const iterator& i);
		bool operator>(const iterator& i);
		bool operator<=(const iterator& i);
		bool operator>=(const iterator& i);
		
		const iterator& operator++();
		iterator operator++(int);
		const iterator& operator--();
		iterator operator--(int);
		
		const iterator& operator+=(const ssize_t i);
		const iterator& operator-=(const ssize_t i);
		
		iterator operator-(const ssize_t i) const;
		iterator operator+(const ssize_t i) const;
		
		typename parent_type::difference_type operator-(const iterator& i) const;
		
		T& operator *();
		const T& operator* () const;
		
		const T& operator-> () const;
		T& operator-> ();
		
	protected: //-- Data --//
		
		parent_type& m_parent;
		T *m_std_iterator;
	};
	
	template<typename T>
	std::ostream& operator<<(std::ostream &os, Ten4<T>& ten4)
	{
		os << "Ten4: (" << ten4.size_i << " x " << ten4.size_j << " x " << ten4.size_k << " x " << ten4.size_l << ")\n";
		
		for (size_t i = 0; i < ten4.size_i; ++i)
		{
			for (size_t j = 0; j < ten4.size_j; ++j)
			{
				for (size_t k = 0; k < ten4.size_k; ++k)
				{
					for (size_t l = 0; l < ten4.size_l; ++l)
					{
						std::cout << std::setw(12) << std::fixed << ten4(i,j,k,l) << " ";
					}
					os << "| ";
				}
				os << "\n";
			}
			os << "---------------------------------\n";
		}
		return os << "\n";
	}
}

#include "backend/impl/tensor/tensor_iterator.inl"
