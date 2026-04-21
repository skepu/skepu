/*! \file matrix.h
 *  \brief Contains a class declaration for the Matrix container.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <map>


#include "backend/malloc_allocator.h"

#ifdef SKEPU_PRECOMPILED

#include "backend/environment.h"
#include "backend/device_mem_pointer_cu.h"

#endif // SKEPU_PRECOMPILED


namespace skepu
{
	template<typename T>
	class Matrix;

	// Proxy matrix for user functions
	template<typename T>
	struct Mat
	{
		using ContainerType = Matrix<T>;

		T *data;
		size_t rows;
		size_t cols;

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
		T &operator()(size_t i, size_t j)       { return this->data[i * this->cols + j]; }
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T  operator()(size_t i, size_t j) const { return this->data[i * this->cols + j]; }
	};

	// Proxy matrix for user functions
	template<typename T>
	struct MatRow
	{
		using ContainerType = Matrix<T>;

		T *data;
		size_t cols;

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
	};


	// Proxy matrix for user functions
	template<typename T>
	struct MatCol
	{
		using ContainerType = Matrix<T>;

		T *data;
		size_t rows, cols;

#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T &operator[](size_t index)       { return this->data[index * this->cols]; }
#ifdef SKEPU_CUDA
		__host__ __device__
#endif
		T  operator[](size_t index) const { return this->data[index * this->cols]; }

		#ifdef SKEPU_CUDA
				__host__ __device__
		#endif
				T &operator()(size_t index)       { return this->data[index * this->cols]; }
		#ifdef SKEPU_CUDA
				__host__ __device__
		#endif
				T  operator()(size_t index) const { return this->data[index * this->cols]; }
	};


	template <typename T>
	class MatrixIterator;

	/*!
	 *  \class Matrix
	 *
	 *  \brief A matrix container class (2D matrix), internally uses 1D container (std::vector) to store elements in a contiguous memory allocations.
	 *
	 *  A \p skepu::Matrix is a 2D container that internally stores in a 1D \p std::vector to store elements in a contiguous memory allocations.
	 *  Its interface and behaviour is largely compatible with \p skepu::Vector and \p std::vector but with some additions and variations.
	 *  Instead of the regular element, it sometimes returns a proxy element so it can distinguish between reads
	 *  and writes. It also keeps track of which parts of it are currently allocated and uploaded to the GPU.
	 *  If a computation is done, changing the matrix in the GPU memory, it is not directly transferred back to the
	 *  host memory. Instead, the Matrix waits until an element is accessed before any copying is done.
	 *
	 *  It also implements support for allocating and de-allocating page-locked memory using cudaMallocHost and cudaFreeHost.
	 *  This could help is running asynchronous operations especially when using multiple CUDA devices.
	 *  It can be enabled by defining USE_PINNED_MEMORY flag in the skeleton program.
	 */
	template<typename T>
	class Matrix
	{
		// typedefs
	public:

		typedef MatrixIterator<T> iterator;
		typedef MatrixIterator<const T> const_iterator;
		typedef Mat<T> proxy_type;

#ifdef SKEPU_CUDA
		typedef backend::DeviceMemPointer_CU<T>* device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
		typedef backend::DeviceMemPointer_CL<T>* device_pointer_type_cl;
		typedef backend::DeviceMemPointer_CL<const T>* device_const_pointer_type_cl;
#endif

#ifdef USE_PINNED_MEMORY
		typedef std::vector<T, malloc_allocator<T> > container_type;
		typedef typename std::vector<T, malloc_allocator<T> >::iterator vector_iterator;
		typedef typename std::vector<T, malloc_allocator<T> >::size_type size_type;
		typedef typename std::vector<T, malloc_allocator<T> >::value_type value_type;
		typedef typename std::vector<T, malloc_allocator<T> >::difference_type difference_type;
		typedef typename std::vector<T, malloc_allocator<T> >::pointer pointer;
		typedef typename std::vector<T, malloc_allocator<T> >::reference reference;
		typedef typename std::vector<T, malloc_allocator<T> >::const_reference const_reference;
#else
		typedef std::vector<T> container_type;
		typedef typename std::vector<T>::iterator vector_iterator;
		typedef typename std::vector<T>::size_type size_type;
		typedef typename std::vector<T>::value_type value_type;
		typedef typename std::vector<T>::difference_type difference_type;
		typedef typename std::vector<T>::pointer pointer;
		typedef typename std::vector<T>::reference reference;
		typedef typename std::vector<T>::const_reference const_reference;
#endif

	public: //-- For Testing --//

		void setLabel(std::string label)
		{
			this->m_label = label;
		}

		std::string getLabel() const
		{
			if (this->m_label != "")
				return this->m_label;

			std::stringstream ss;
			ss << skepu::debug.getObjectName(this->getAddress());
			return ss.str();
		}

		UniqueIdentifier::ID getObjectID() const
		{
			return this->m_object_id;
		}

		void setValidFlag(bool val);
		void randomize(int min = 0, int max = RAND_MAX);
		void randomizeReal(double min = 0, double max = 1);
		void save(const std::string& filename);
		void load(const std::string& filename, size_type rowWidth, size_type numRows = 0);

		friend std::ostream& operator<<(std::ostream &os, Matrix<T>& matrix)
		{
			matrix.updateHost();

			os << "Matrix: (" << matrix.total_rows() << " X " << matrix.total_cols() << ")\n";
			for(typename Matrix<T>::size_type i = 0; i < matrix.size(); i++)
			{
				os << matrix(i / matrix.total_cols(), i % matrix.total_cols()) << " ";
				if ((i+1) % matrix.total_cols() == 0)
					os << "\n";
			}
			return os;
		}

	// Constructors, destructors
	public:

		Matrix();
		Matrix(const Matrix<T>& copy);
		Matrix(Matrix<T>&& move);
		Matrix(size_type _rows, size_type _cols, const T& val = T());
		Matrix(size_type _rows, size_type _cols, std::string label, const T& val = T());
		Matrix(std::initializer_list<T> l);
		Matrix(T * const ptr, size_type _rows, size_type _cols, bool deallocEnabled = true);

		~Matrix();


		void operator=(std::initializer_list<T> l)
		{
			if (l.size() != this->size())
				SKEPU_ERROR("Matrix: Invalid initializer list size");

			size_t i = 0;
			for (const T& elem : l)
				this->m_data[i++] = elem;
		}

		void init(size_type _rows, size_type _cols);
		void init(size_type _rows, size_type _cols, const T& val);

		const Matrix<T>& getParent() const
		{
			return *this;
		}

		Matrix<T>& getParent()
		{
			return *this;
		}

		/*!
		 * Returns total size of Matrix.
		 * \return size of the Matrix.
		 */
		size_type size() const
		{
			return this->m_rows * this->m_cols;
		}

		/*!
		 * Returns total number of rows in the Matrix.
		 * \return rows in the Matrix.
		 */
		size_type total_rows() const
		{
			return this->m_rows;
		}

		/*!
		 * Returns total number of columns in the Matrix.
		 * \return columns in the Matrix.
		 */
		size_type total_cols() const
		{
			return this->m_cols;
		}

		size_type size_i() const { return this->m_rows; }
		size_type size_j() const { return this->m_cols; }
		size_type size_k() const { return 0; }
		size_type size_l() const { return 0; }

		// All dimensions
		std::tuple<size_type, size_type, size_type, size_type> size_info() const
		{
			return {this->m_rows, this->m_cols, 1, 1};
		}

		std::tuple<size_type, size_type, size_type, size_type> row_size_info() const
		{
			return {this->m_cols, 1, 1, 1};
		}

		std::tuple<size_type, size_type, size_type, size_type> col_size_info() const
		{
			return {this->m_rows, 1, 1, 1};
		}

		// highly dangerous, use with care.
		T *getAddress()
		{
			return this->m_data;
		}

		T *data()
		{
			return this->m_data;
		}

		const T *getAddress() const
		{
			return this->m_data;
		}

		const T *data() const
		{
			return this->m_data;
		}

		template<typename Ignore>
		Mat<T> hostProxy(ProxyTag::Default, Ignore)
		{
			Mat<T> proxy;
			proxy.data = this->m_data;
			proxy.rows = this->m_rows;
			proxy.cols = this->m_cols;
			return proxy;
		}

		Mat<T> hostProxy()
		{
			return this->hostProxy(ProxyTag::Default{}, 0);
		}

		// Matrix row

		MatRow<T> hostProxy(ProxyTag::MatRow, size_t r)
		{
			MatRow<T> proxy;
			proxy.data = this->m_data + r * this->m_cols;
			proxy.cols = this->m_cols;
			return proxy;
		}

		MatRow<T> hostProxy(ProxyTag::MatRow, Index1D row)
		{
			return this->hostProxy(ProxyTag::MatRow{}, row.i);
		}

		MatRow<T> hostProxy(ProxyTag::MatRow, Index2D coord)
		{
			return this->hostProxy(ProxyTag::MatRow{}, coord.row);
		}


		// Matrix column

		MatCol<T> hostProxy(ProxyTag::MatCol, size_t c)
		{
			MatCol<T> proxy;
			proxy.data = this->m_data + c;
			proxy.rows = this->m_rows;
			proxy.cols = this->m_cols;
			return proxy;
		}

		MatCol<T> hostProxy(ProxyTag::MatCol, Index1D col)
		{
			return this->hostProxy(ProxyTag::MatCol{}, col.i);
		}

		MatCol<T> hostProxy(ProxyTag::MatCol, Index2D coord)
		{
			return this->hostProxy(ProxyTag::MatCol{}, coord.col);
		}




		/*!
		 *  A small utility to change rows and columns numbers with each other. A Matrix (4x7) will become (7x4) after this function call without
		 *  changing the actual values. Not similar to transpose where you actually change the values.
		 */
		void change_layout()
		{
			size_type tmp = m_rows;
			m_rows=m_cols;
			m_cols = tmp;

			if (m_transpose_matrix && m_transpose_matrix->total_rows() == m_cols && m_transpose_matrix->total_cols() == m_rows && !m_dataChanged)
				m_transpose_matrix->change_layout();
		}

	private:

#ifdef SKEPU_CUDA
		mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_CU[MAX_GPU_DEVICES];

		/*! This is a temporary list that keeps track of copies that are changed on device but are not synced with host memory... */
		mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_Modified_CU[MAX_GPU_DEVICES];
#endif

#ifdef SKEPU_OPENCL
		mutable std::map<std::pair<cl_device_id, std::pair<const T*, size_type>>, device_pointer_type_cl > m_deviceMemPointers_CL;
		mutable std::map<std::pair<cl_device_id, std::pair<const T*, size_type>>, device_const_pointer_type_cl > m_deviceConstMemPointers_CL;
#endif

		size_type m_rows, m_cols;
		mutable bool m_dataChanged;
		mutable bool m_noValidDeviceCopy;
		bool m_deallocEnabled = true;
		std::string m_label = "";
		UniqueIdentifier::ID m_object_id;

#ifdef USE_PINNED_MEMORY
		mutable std::vector<T, malloc_allocator<T> > m_data;
#else
		T *m_data = nullptr;
#endif

		mutable bool m_valid; /*! to keep track of whether the main copy is valid or not */

		// for col_iterator,
		mutable Matrix<T> *m_transpose_matrix = nullptr;

		template<typename Type>
		void item_swap(Type &t1, Type &t2);


	// External classes
	public:

		class proxy_elem;

	public: //-- Operators --//

		Matrix<T>& operator=(const Matrix<T>& other);
		Matrix<T>& operator=(const T& elem);
		void set(const size_t & row, const size_t & col, const T & value);

		bool operator==(const Matrix<T>& c1);
		bool operator!=(const Matrix<T>& c1);

	public: //-- STL vector regular interface --//

		// Iterators
		iterator begin();
		const_iterator begin() const;
		iterator begin(size_t row);
		const_iterator begin(size_t row) const;

		// These do nothing special for now
		iterator stridedBegin(size_t, int) { return this->begin(); };
		const_iterator begin(size_t, int) const { return this->begin(); };

		iterator end();
		const_iterator end() const;
		iterator end(size_t row);
		const_iterator end(size_t row) const;

		void flush(FlushMode mode = FlushMode::Default);

		// Element access
#ifdef SKEPU_PRECOMPILED
		proxy_elem at(size_type row, size_type col);
#else
		T& at(size_type row, size_type col);
#endif // SKEPU_PRECOMPILED

		void swap(Matrix<T>& from);

	public: //-- Additions to interface --//

	#ifdef SKEPU_OPENCL
		device_pointer_type_cl updateDevice_CL(T* start, size_type rows, size_type cols, backend::Device_CL* device, bool copy);
		device_pointer_type_cl updateDevice_CL(T* start, size_type cols, backend::Device_CL* device, bool copy);

		device_const_pointer_type_cl updateDevice_CL(const T* start, size_type rows, size_type cols, backend::Device_CL* device, bool copy) const;
		device_const_pointer_type_cl updateDevice_CL(const T* start, size_type cols, backend::Device_CL* device, bool copy) const;
		void flush_CL(FlushMode mode);
	#endif

	#ifdef SKEPU_CUDA
		void copyDataToAnInvalidDeviceCopy(backend::DeviceMemPointer_CU<T> *copy, size_t deviceID, size_t streamID = 0) const;
		device_pointer_type_cu updateDevice_CU(T* start, size_type rows, size_type cols, size_t deviceID, size_t streamID, AccessMode accessMode, bool usePitch, bool markOnlyLocalCopiesInvalid=false) const;
		device_pointer_type_cu updateDevice_CU(T* start, size_type cols, size_t deviceID, AccessMode accessMode, bool markOnlyLocalCopiesInvalid=false, size_t streamID = 0) const;
		void flush_CU(FlushMode mode);

		bool isMatrixOnDevice_CU(size_t deviceID) const;
		bool isModified_CU(size_t deviceID) const;

		template<typename Ignore>
		std::pair<device_pointer_type_cu, Mat<T>>
		cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::Default, Ignore)
		{
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->m_rows * this->m_cols, deviceID, accessMode);
			Mat<T> proxy;
			proxy.data = devptr->getDeviceDataPointer();
			proxy.rows = this->m_rows;
			proxy.cols = this->m_cols;
			return {devptr, proxy};
		}

		std::pair<device_pointer_type_cu, Mat<T>>
		cudaProxy(size_t deviceID, AccessMode accessMode)
		{
			return this->cudaProxy(deviceID, accessMode, ProxyTag::Default{}, 0);
		}

		std::pair<device_pointer_type_cu, MatRow<T>>
		cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::MatRow, Index1D row)
		{
			// TODO: Optimize
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->m_rows * this->m_cols, deviceID, accessMode);
			MatRow<T> proxy;
			proxy.data = devptr->getDeviceDataPointer() + row.i * this->m_cols;
			proxy.cols = this->m_cols;
			return {devptr, proxy};
		}

		std::pair<device_pointer_type_cu, MatCol<T>>
		cudaProxy(size_t deviceID, AccessMode accessMode, ProxyTag::MatCol, Index1D col)
		{
			// TODO: Optimize
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->m_rows * this->m_cols, deviceID, accessMode);
			MatCol<T> proxy;
			proxy.data = devptr->getDeviceDataPointer() + col.i;
			proxy.rows = this->m_rows;
			return {devptr, proxy};
		}
	#endif

		// Don't Care about device data
		const T& operator()(const size_type row, const size_type col) const;

		// Don't Care about device data
		T& operator()(const size_type row, const size_type col);

#ifdef SKEPU_ENABLE_DEPRECATED_OPERATOR
		// Does not care about device data, use with care
		T& operator()(const size_type index);

		// Does not care about device data, use with care
		T& operator()(const Index2D index);

		// Care about device data
		const T& operator[](const size_type index) const;

		// Care about device data
		T& operator[](const size_type index);
#endif // SKEPU_ENABLE_DEPRECATED_OPERATOR

		void transpose_CPU();

#ifdef SKEPU_OPENMP
		void transpose_OMP();
#endif

#ifdef SKEPU_CUDA
		void transpose_CU(backend::Device_CU *device);
#endif

#ifdef SKEPU_OPENCL
		void transpose_CL(size_t deviceID);
		std::vector<std::pair<cl_kernel, backend::Device_CL*> > *m_transposeKernels_CL;
#endif

		// unary transpose operator
		Matrix<T>& operator~()
		{
			if (m_transpose_matrix && m_transpose_matrix->m_rows == m_cols && m_transpose_matrix->m_cols == m_rows && !m_dataChanged)
				return *m_transpose_matrix;

#if defined(SKEPU_CUDA)
			transpose_CU(backend::Environment<int>::getInstance()->m_devices_CU.at(backend::Environment<int>::getInstance()->m_best_cuda_device_id));
#elif  defined(SKEPU_OPENCL)
			transpose_CL(0);
#elif defined(SKEPU_OPENMP)
			transpose_OMP();
#else
			transpose_CPU();
#endif
			this->m_dataChanged = false;
			return *m_transpose_matrix;
		}

		// unary transpose operator
		Matrix<T>& transpose(const skepu::BackendSpec &spec)
		{
			if (this->m_transpose_matrix && this->m_transpose_matrix->m_rows == this->m_cols && this->m_transpose_matrix->m_cols == this->m_rows && !this->m_dataChanged)
				return *this->m_transpose_matrix;

			switch (spec.type())
			{
			case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
				this->transpose_CU(backend::Environment<int>::getInstance()->m_devices_CU[backend::Environment<int>::getInstance()->m_best_cuda_device_id]);
				break;
#endif
			case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
				this->transpose_CL(0);
				break;
#endif
			case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
				this->transpose_OMP();
				break;
#endif
			default:
				this->transpose_CPU();
			}

			this->m_dataChanged = false;
			return *this->m_transpose_matrix;
		}

		// To be able to explicitly force updates without flushing entire matrix.
		// Could be used with operator () above to avoid unneccesary function calls
		// due to implicit synch.

		void updateHost(bool = true) const;
		void invalidateDeviceData(bool = true) const;
		void updateHostAndInvalidateDevice();
		void releaseDeviceAllocations();
		void updateHostAndReleaseDeviceAllocations();


		const Matrix<T>& operator+=(const Matrix<T>& rhs);
		const Matrix<T>& operator+=(const T& rhs);

		const Matrix<T>& operator-=(const Matrix<T>& rhs);
		const Matrix<T>& operator-=(const T& rhs);

		const Matrix<T>& operator*=(const Matrix<T>& rhs);
		const Matrix<T>& operator*=(const T& rhs);

		const Matrix<T>& operator/=(const Matrix<T>& rhs);
		const Matrix<T>& operator/=(const T& rhs);

		const Matrix<T>& operator%=(const Matrix<T>& rhs);
		const Matrix<T>& operator%=(const T& rhs);

	private:

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

	}; // end class Matrix...


	/*!
	 *  \class Matrix::iterator
	 *
	 *  \brief An matrix iterator class that tranverses row-wise.
	 *
	 *  An iterator class for \p skepu::Matrix. It traverses a Matrix row-wise assuming Matrix is stored in row-major order
	 *  which is \p skepu::Matrix default style. It behaves like the 1D container iterators like iterator for \p std::vector
	 *  but similar to \p skepu::Matrix it sometimes returns a \p proxy_elem instead of the actual
	 *  element. Also makes sure the matrix is properly synchronized with device before returning
	 *  any elements.
	 */
	template <typename T>
	class MatrixIterator
	{
	public:
		typedef MatrixIterator<T> iterator;
		typedef MatrixIterator<const T> const_iterator;
		typedef typename std::conditional<std::is_const<T>::value,
					const Matrix<typename std::remove_const<T>::type>, Matrix<T>>::type parent_type;
		typedef Mat<T> proxy_type;

	#ifdef SKEPU_CUDA
		typedef typename parent_type::device_pointer_type_cu device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
		typedef typename parent_type::device_pointer_type_cl device_pointer_type_cl;
#endif

	public: //-- Constructors & Destructor --//

	/*	TODO: Fix pinned memory
#ifdef USE_PINNED_MEMORY
		typedef typename std::vector<typename std::remove_const<T>::type, malloc_allocator<typename std::remove_const<T>::type> >::iterator iterator_type;
#else
		typedef typename std::vector<typename std::remove_const<T>::type>::iterator iterator_type;
#endif*/

		MatrixIterator(parent_type *mat, T *std_iterator);

	public: //-- Extras --//

		Index2D getIndex() const;

		parent_type& getParent() const;
		iterator& begin(); // returns itself
		size_t size(); // returns number of elements "left" in parent container from this index

		T* getAddress() const;
		T* data();

		//Does care about device data, uses updateAndInvalidateDevice for read and write access
		T& operator()(const ssize_t rows, const ssize_t cols);

		//Does care about device data, uses updateDevice, for readonly access
		const T& operator()(const ssize_t rows, const ssize_t cols) const;

		//Does not care about device data, use with care
		T& operator()(const ssize_t index=0);

		Mat<T> hostProxy()
		{
			Mat<T> proxy;
			proxy.data = this->m_parent->m_data;
			proxy.rows = this->m_parent->m_rows;
			proxy.cols = this->m_parent->m_cols;
			return proxy;
		}

	public: //-- Operators --//

		T& operator[](const ssize_t index);

		const T& operator[](const ssize_t index) const;


	//	operator const_iterator() const;
	//	operator iterator_type() const;

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

		iterator& stride_row(const ssize_t stride=1);

		iterator operator-(const ssize_t i) const;
		iterator operator+(const ssize_t i) const;

		typename parent_type::difference_type operator-(const iterator& i) const;

		T& operator *();
		const T& operator* () const;

		const T& operator-> () const;
		T& operator-> ();

	private: //-- Data --//

		parent_type* m_parent;
		T *m_std_iterator;
	};

} // end namespace skepu


#include "backend/impl/matrix/matrix.inl"

#endif
