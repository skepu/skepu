/*! \file vector.h
*  \brief Contains a class declaration for the Vector container.
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <map>

#include "backend/malloc_allocator.h"

#ifdef SKEPU_PRECOMPILED

#include "backend/environment.h"
#include "backend/device_mem_pointer_cl.h"
#include "backend/device_mem_pointer_cu.h"

#endif // SKEPU_PRECOMPILED


namespace skepu
{
	enum class FlushMode {
		Default, Dealloc
	};

	template<typename T>
	class Vector;

	// Proxy vector for user functions
	template<typename T>
	struct Vec
	{
		using ContainerType = Vector<T>;

		Vec(T *dataptr, size_t sizearg): data{dataptr}, size{sizearg} {}
		Vec(): data{nullptr}, size{0} {} // empty proxy constructor

#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T &operator[](size_t index)       { return this->data[index]; }

#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T  operator[](size_t index) const { return this->data[index]; }

#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T &operator()(size_t index)       { return this->data[index]; }

#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T  operator()(size_t index) const { return this->data[index]; }

		T *data;
		size_t size;
	};

	template <typename T>
	class VectorIterator;

	template <typename T>
	class StridedVectorIterator;

	/*!
	*  \class Vector
	*
	*  \brief A vector container class, implemented as a wrapper for std::vector.
	*
	*  A \p skepu::Vector is a container of vector/array type and is implemented as a wrapper for \p std::vector.
	*  Its interface and behaviour is largely compatible with \p std::vector but with some additions and variations.
	*  Instead of the regular element, it sometimes returns a proxy element so it can distinguish between reads
	*  and writes. It also keeps track of which parts of it are currently allocated and uploaded to the GPU.
	*  If a computation is done, changing the vector in the GPU memory, it is not directly transferred back to the
	*  host memory. Instead, the vector waits until an element is accessed before any copying is done.
	*
	*  It also implements support for allocating and de-allocating page-locked memory using cudaMallocHost and cudaFreeHost.
	*  This could help is running asynchronous operations especially when using multiple CUDA devices.
	*  It can be enabled by defining USE_PINNED_MEMORY flag in the skeleton program.
	*
	*  Please refer to C++ STL vector documentation for more information about CPU side implementation.
	*/
	template <typename T>
	class Vector
	{

	public:

		typedef typename std::vector<T>::size_type size_type;
		typedef T value_type;
		typedef ptrdiff_t difference_type;
		typedef T* pointer;
		typedef T& reference;
		typedef T const & const_reference;
		typedef Vec<T> proxy_type;

		typedef VectorIterator<T> iterator;
		typedef VectorIterator<const T> const_iterator;

		typedef StridedVectorIterator<T> strided_iterator;
		typedef StridedVectorIterator<const T> const_strided_iterator;

		//-- For Testing --//

		friend std::ostream& operator<< (std::ostream& output, Vector<T>& vec)
		{
			vec.flush();
			output << vec.size() << ".... ";
			for (typename Vector<T>::size_type i = 0; i < vec.size(); ++i)
			{
				output<<vec(i) <<" ";
			}
			return output;
		}

	public: //-- For Testing --//

		void setLabel(std::string label)
		{
			if (!this->m_label)
				this->m_label = new std::string(label);
			else
				*this->m_label = label;
		}

		std::string getLabel() const
		{
			if (this->m_label)
				return *this->m_label;

			std::stringstream ss;
			ss << skepu::debug.getObjectName(this->getAddress());
			return ss.str();
		}

		UniqueIdentifier::ID getObjectID() const
		{
			return this->m_object_id;
		}

		void randomize(int min = 0, int max = RAND_MAX);
		void save(const std::string& filename, const std::string& delimiter=" ");
		void load(const std::string& filename, size_type numElements = 0);

	public: //-- Typedefs --//

#ifdef SKEPU_CUDA
		typedef backend::DeviceMemPointer_CU<T>* device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
		typedef backend::DeviceMemPointer_CL<T>* device_pointer_type_cl;
		typedef backend::DeviceMemPointer_CL<const T>* device_const_pointer_type_cl;
#endif

	public: //-- Constructors & Destructor --//

		Vector();
		Vector(const Vector& vec);
		Vector(Vector&& vec);
		Vector(std::initializer_list<T> l);
		explicit Vector(size_type num, const T& val = T());
		explicit Vector(size_type num, std::string label, const T& val = T());
		Vector(T * const ptr, size_type size, bool deallocEnabled = true);

		~Vector();

		void init(size_type size);
		void init(size_type size, const T& val);

	public: //-- Member classes --//

		class proxy_elem;

	public: //-- Operators --//

#ifdef SKEPU_ENABLE_DEPRECATED_OPERATOR
		const T& operator[](const size_type index) const;
#ifdef SKEPU_PRECOMPILED
		proxy_elem operator[](const size_type index);
#else
		T& operator[](const size_type index);
#endif // SKEPU_PRECOMPILED
#endif // SKEPU_ENABLE_DEPRECATED_OPERATOR


		template<typename Ignore>
		Vec<T> hostProxy(ProxyTag::Default, Ignore)
		{
			return {this->m_data, this->m_size};
		}

		Vec<T> hostProxy() { return this->hostProxy(ProxyTag::Default{}, 0); }

		Vector<T>& operator=(const Vector<T>&);

		bool operator==(const Vector<T>&);
		bool operator!=(const Vector<T>&);

	public: //-- STL vector regular interface --//

		// Iterators
		iterator begin();
		iterator end();
		strided_iterator stridedBegin(size_t n, int dir);

		const_iterator begin() const;
		const_iterator end() const;
		const_strided_iterator stridedBegin(size_t n, int dir) const;

		// Capacity
		size_type size() const       { return this->m_size; }
		size_type total_cols() const { return this->m_size; }

		size_type size_i() const { return this->m_size; }
		size_type size_j() const { return 0; }
		size_type size_k() const { return 0; }
		size_type size_l() const { return 0; }

		// All dimensions
		std::tuple<size_type, size_type, size_type, size_type> size_info() const
		{
			return {this->m_size, 1, 1, 1};
		}

		const Vector<T>& getParent() const { return *this; }
		Vector<T>& getParent() { return *this; }

		// Element access
		const T& at(size_type loc) const;

#ifdef SKEPU_PRECOMPILED
		proxy_elem at(size_type loc);
#else
		T& at(size_type loc);
#endif // SKEPU_PRECOMPILED

        const skepu::Scalar<T> elem(size_type loc);

		void swap(Vector<T>& from);

		T *getAddress() { return this->m_data; }
		const T *getAddress() const { return this->m_data; }

		T *data() { return this->m_data; }
		const T *data() const { return this->m_data; }

	public: //-- Additions to interface --//

#ifdef SKEPU_OPENCL
		device_const_pointer_type_cl updateDevice_CL(const T* start, size_type numElements, backend::Device_CL* device, bool copy) const;
		device_pointer_type_cl updateDevice_CL(T* start, size_type numElements, backend::Device_CL* device, bool copy);
		void flush_CL(FlushMode mode);
		bool isVectorOnDevice_CL(backend::Device_CL* device, bool multi=false) const;
#endif

#ifdef SKEPU_CUDA
		void copyDataToAnInvalidDeviceCopy(backend::DeviceMemPointer_CU<T> *copy, size_t deviceID, size_t streamID = 0) const;
		device_pointer_type_cu updateDevice_CU(T* start, size_type numElements, size_t deviceID, AccessMode accessMode, bool markOnlyLocalCopiesInvalid = false, size_t streamID = 0) const;
		void flush_CU(FlushMode mode);
		bool isVectorOnDevice_CU(size_t deviceID) const;
		bool isModified_CU(size_t deviceID) const;

		template<typename Ignore>
		std::pair<device_pointer_type_cu, Vec<T>> cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::Default, Ignore)
		{
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->m_size, deviceID, accessMode);
			return {devptr, {devptr->getDeviceDataPointer(), this->m_size}};
		}

		std::pair<device_pointer_type_cu, Vec<T>> cudaProxy(size_t deviceID, AccessMode accessMode)
		{
			return this->cudaProxy(deviceID, accessMode, ProxyTag::Default{}, 0);
		}
#endif

		void flush(FlushMode mode = FlushMode::Default);

		// Does not care about device data, use with care
		T& operator()(const size_type index) { return m_data[index]; }
		const T& operator()(const size_type index) const { return m_data[index]; }

		// To be able to explicitly force updates without flushing entire vector.
		// Could be used with operator () above to avoid unneccesary function calls
		// due to implicit synch.
		void updateHost(bool = true) const;
		void invalidateDeviceData(bool = true) const;
		void updateHostAndInvalidateDevice();
		void releaseDeviceAllocations();
		void updateHostAndReleaseDeviceAllocations();

		void setValidFlag(bool val)
		{
			m_valid = val;
		}

	protected: //-- Data --//
		T *m_data = nullptr;
		mutable bool m_valid; /*! to keep track of whether the main copy is valid or not */
		size_type m_size = 0;
		bool m_deallocEnabled = true;
		mutable bool m_noValidDeviceCopy;
		std::string *m_label = nullptr;
		UniqueIdentifier::ID m_object_id;

#ifdef SKEPU_OPENCL
		mutable std::map<std::pair<cl_device_id, const T* >, device_pointer_type_cl > m_deviceMemPointers_CL;
		mutable std::map<std::pair<cl_device_id, const T* >, device_const_pointer_type_cl > m_deviceConstMemPointers_CL;
#endif

#ifdef SKEPU_CUDA
//      std::map<std::pair< int, std::pair< T*, size_type > >, device_pointer_type_cu > m_deviceMemPointers_CU;
		mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_CU[MAX_GPU_DEVICES];

   /*! This is a temporary list that keeps track of copies that are changed on device but are not synced with host memory... */
		mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_Modified_CU[MAX_GPU_DEVICES];
#endif

//-- Private helpers --//

#ifdef SKEPU_OPENCL
		void updateHost_CL() const;
		void invalidateDeviceData_CL() const;
		void releaseDeviceAllocations_CL() const;
#endif

#ifdef SKEPU_CUDA
		void updateHost_CU(int deviceID = -1) const;
		void invalidateDeviceData_CU(int deviceID = -1) const;
		void releaseDeviceAllocations_CU(int deviceID = -1) const;
#endif

	};

	/*!
	*  \class Vector::iterator
	*  \author Johan Enmyren, Usman Dastgeer
	*  \version 0.7
	*
	*  \brief An vector iterator class.
	*
	*  An iterator class for \p skepu::Vector. behaves like the vector iterator for \p std::vector
	*  but similar to \p skepu::Vector it sometimes returns a \p proxy_elem instead of the actual
	*  element. Also makes sure the vector is properly synchronized with device before returning
	*  any elements.
	*/
	template <typename T>
	class VectorIterator
	{
	public:
		typedef VectorIterator<T> iterator;
		typedef VectorIterator<const T> const_iterator;
	   typedef typename std::conditional<std::is_const<T>::value,
						const Vector<typename std::remove_const<T>::type>, Vector<T>>::type parent_type;

		typedef Vec<T> proxy_type;

	public: //-- Constructors & Destructor --//

		VectorIterator(parent_type& vec, T *std_iterator);

	public: //-- Types --//

#ifdef SKEPU_CUDA
		typedef typename parent_type::device_pointer_type_cu device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
		typedef typename parent_type::device_pointer_type_cl device_pointer_type_cl;
#endif

	public: //-- Extras --//

		Index1D getIndex() const;

		parent_type& getParent() const;
		iterator& begin(); // returns itself
		typename Vector<T>::strided_iterator stridedBegin(size_t n, int dir);
		size_t size(); // returns number of elements "left" in parent container from this index

		T* getAddress() const;
		T* data();

		template<typename Ignore>
		Vec<T> hostProxy(ProxyTag::Default, Ignore)
		{ return {this->m_std_iterator, this->size()}; }

		Vec<T> hostProxy() { return this->hostProxy(ProxyTag::Default{}); }

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

	template <typename T>
	class StridedVectorIterator: public VectorIterator<T>
	{
	public:
		StridedVectorIterator(typename VectorIterator<T>::parent_type& vec, T *std_iterator, int stride)
		: VectorIterator<T>(vec, std_iterator), m_stride(stride) {}

		const StridedVectorIterator<T>& operator++()
		{
			this->m_std_iterator += this->m_stride;
			return *this;
		}

		StridedVectorIterator<T> operator++(int) //Postfix
		{
			StridedVectorIterator<T> temp(*this);
			this->m_std_iterator += this->m_stride;
			return temp;
		}

		const StridedVectorIterator<T>& operator+=(const ssize_t i)
		{
			this->m_std_iterator += i * this->m_stride;
			return *this;
		}

		StridedVectorIterator<T> operator+(const ssize_t i) const
		{
			StridedVectorIterator<T> temp(*this);
			temp += i;
			return temp;
		};

		T& operator()(const ssize_t index = 0)
		{
			return this->m_std_iterator[index * this->m_stride];
		}

		const T& operator()(const ssize_t index) const
		{
			return this->m_std_iterator[index * this->m_stride];
		}


		// If stride > 0: Returns the number of elements remaining, starting from the iterator / abs(stride)
		// If stride < 0: Returns the number of elements preceeding, ending with the iterator / abs(stride)
		// Else return "infinity"
		size_t size() const
		{
			if (this->m_stride >= 0)
				return (this->m_parent.end() - *this) / this->m_stride;
			else if (this->m_stride < 0)
				return (*this - this->m_parent.begin() - this->m_stride) / (-1 * this->m_stride);
			else
				return std::numeric_limits<size_t>::max();
		}

	private:
		int m_stride;
	};

}

#include "backend/impl/vector/vector.inl"

#endif
