namespace skepu
{
	/*!
	 *  \brief Randomizes the vector.
	 *
	 *  Sets each element of the vector to a random number between \p min and \p max.
	 *  The numbers are generated as \p integers but are cast to the type of the vector.
	 *
	 *  \param min The smallest number an element can become.
	 *  \param max The largest number an element can become.
	 */
	template<typename T>
	void Vector<T>::randomize(int min, int max)
	{
		invalidateDeviceData();
		
		for (size_type i = 0; i < size(); i++)
		{
			m_data[i] = (T)( rand() % max + min );
		}
	}
	
	
	/*!
	 *  \brief Saves content of vector to a file.
	 *
	 *  Outputs the vector as text on one line with space between elements to the specified file.
	 *  Mainly for testing purposes.
	 *
	 *  \param filename Name of file to save to.
	 */
	template<typename T>
	void Vector<T>::save(const std::string& filename, const std::string& delimiter)
	{
		std::ofstream file(filename.c_str());
		
		if (file.is_open())
		{
			for (size_type i = 0; i < size(); ++i)
			{
				file << at(i) << delimiter;
			}
			file.close();
		}
		else
		{
			std::cout<<"Unable to open file\n";
		}
	}
	
	
	/*!
	 *  \brief Loads the vector from a file.
	 *
	 *  Reads a variable number of elements from a file. In the file, all elemets should be in ASCII
	 *  on one line with whitespace between each element. Mainly for testing purposes.
	 *
	 *  \param filename Name of file to save to.
	 *  \param numElements The number of elements to load. Default value 0 means all values.
	 */
	template<typename T>
	void Vector<T>::load(const std::string& filename, size_type numElements)
	{
		std::ifstream file(filename.c_str());

		if (file.is_open())
		{
			std::string line;
			getline (file,line);
			std::istringstream ss(line);
			T num;
			
			// Load all elements
			if (numElements == 0)
			{
				while (ss >> num)
					push_back(num);
			}
			// Load only numElements elements
			else
			{
				for (size_type i = 0; i < numElements; ++i)
				{
					ss >> num;
					push_back(num);
				}
			}
			
			file.close();
		}
		else
		{
			std::cout<<"Unable to open file\n";
		}
	}
	
///////////////////////////////////////////////
// Constructors START
///////////////////////////////////////////////

	/*!
	 *  Constructs an empty vector.
	 */
	template <typename T>
	inline Vector<T>::Vector()
	:	m_data(nullptr),
		m_size(0),
		m_deallocEnabled(false),
		m_valid(false),
		m_noValidDeviceCopy(true),
		m_object_id(UniqueIdentifier::generate())
	{
		DEBUG_TEXT_LEVEL1("Vector: Constructor to an empty state");
	}
	
	/*!
	 *  The copy occurs w.r.t. elements. As copy constructor creates a new storage.
	 *
	 *  Updates vector \p c before copying.
	 */
	template <typename T>
	inline Vector<T>::Vector(const Vector& c)
	:	m_size(c.m_size),
		m_deallocEnabled(true),
		m_valid(true),
		m_noValidDeviceCopy(true),
		m_object_id(UniqueIdentifier::generate())
	{
		DEBUG_TEXT_LEVEL1("Vector: Copy constructor with " << this->m_size << " elements");
		
		this->init(this->m_size);
		c.updateHost();
		std::copy(c.m_data, c.m_data + this->m_size, this->m_data);
	}
	
	
	template <typename T>
	inline Vector<T>::Vector(Vector&& move):
		m_data(nullptr),
		m_size(0),
		m_deallocEnabled(false),
		m_valid(false),
		m_noValidDeviceCopy(true),
		m_object_id(UniqueIdentifier::generate())
	{
		DEBUG_TEXT_LEVEL1("Vector: Move constructor with " << move.size() << " elements");
		this->swap(move);
	}
	
	
	template <typename T>
	inline Vector<T>::Vector(std::initializer_list<T> l)
	:	m_size(l.size()),
		m_deallocEnabled(true),
		m_valid(true),
		m_noValidDeviceCopy(true),
		m_object_id(UniqueIdentifier::generate())
	{
		DEBUG_TEXT_LEVEL1("Vector: Constructor from initializer list with " << this->m_size << " elements");
		this->init(this->m_size);
		
		int i = 0;
		for (const T& elem : l)
			this->m_data[i++] = elem;
	}
	
	
	/**!
	 * Used to construct vector on a raw data pointer passed to it as its payload data.
	 * Useful when creating the vector object with existing raw data pointer.
	 */
	template <typename T>
	inline Vector<T>::Vector(T * const ptr, size_type size, bool deallocEnabled)
	:	m_size (size),
		m_deallocEnabled(deallocEnabled),
		m_valid(true),
		m_noValidDeviceCopy(true),
		m_object_id(UniqueIdentifier::generate())
	{
		DEBUG_TEXT_LEVEL1("Vector: Constructor from existing pointer with " << this->m_size << " elements");
		
		if (this->m_size < 1)
			SKEPU_ERROR("The vector size must be positive.");
		
		if (!ptr)
		{
			SKEPU_ERROR("Error: The supplied pointer for initializing vector object is invalid");
			return;
		}
		
		this->m_data = ptr;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	inline Vector<T>::Vector(size_type num, const T& val)
	:	m_size(num),
		m_deallocEnabled(true),
		m_valid(true),
		m_noValidDeviceCopy(true),
		m_object_id(UniqueIdentifier::generate())
	{
		DEBUG_TEXT_LEVEL1("Vector: Constructor with " << num << " elements");
		this->init(num, val);
	}

	template <typename T>
	inline Vector<T>::Vector(size_type num, std::string label, const T& val)
	:	m_size(num),
		m_label(new std::string(label)),
		m_deallocEnabled(true),
		m_valid(true),
		m_noValidDeviceCopy(true),
		m_object_id(UniqueIdentifier::generate())
	{
		DEBUG_TEXT_LEVEL1("Vector: Constructor with label and " << num << " elements");
		this->init(num, val);
	}


	// Init
	
	template <typename T>
	void Vector<T>::init(size_type size)
	{
		DEBUG_TEXT_LEVEL1("Vector: Allocating with " << size << " elements");
		
		if (!this->m_data)
		{
			if (size < 1)
				SKEPU_ERROR("The container size must be positive.");
			this->m_size = size;
			backend::allocateHostMemory<T>(this->m_data, this->m_size);
	    this->m_deallocEnabled = true;

			SKEPU_TRACE_ALLOCATION(this->getObjectID(), this->getLabel(), __LINE__);
		}
		else SKEPU_ERROR("Container is already initialized");
	}
	
	template <typename T>
	void Vector<T>::init(size_type size, const T& val)
	{
		this->init(size);
		std::fill(this->m_data, this->m_data + m_size, val);
	}
	
///////////////////////////////////////////////
// Constructors END
///////////////////////////////////////////////

///////////////////////////////////////////////
// Destructor START
///////////////////////////////////////////////
	
	/*!
	 *  Releases all allocations made on device.
	 */
	template <typename T>
	Vector<T>::~Vector()
	{
		releaseDeviceAllocations();

		if (this->m_data && this->m_deallocEnabled)
		{
			backend::deallocateHostMemory<T>(this->m_data);
			SKEPU_TRACE_DEALLOCATION(this->getObjectID(), this->getLabel(), __LINE__);
			DEBUG_TEXT_LEVEL1("Vector: Destroyed with " << this->m_size << " elements");
		}
		else
	  {
	    DEBUG_TEXT_LEVEL1("Vector: Note, did not deallocate data.");
	  }

		if (this->m_label != nullptr) delete this->m_label;
	}
	
///////////////////////////////////////////////
// Destructor END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Public Helpers START
///////////////////////////////////////////////

	/*!
	 *  Updates the vector from its device allocations.
	 */
	template <typename T>
	inline void Vector<T>::updateHost(bool enable) const
	{
		if (!enable)
			return;
	
#ifdef SKEPU_OPENCL
		updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   /*! the m_valid logic is only implemented for CUDA backend. The OpenCL still uses the old memory management mechanism */
		if(m_valid) // if already up to date then no need to check...
			return;
   
		updateHost_CU();
#endif

		m_valid = true;
	}

	/*!
	 *  Invalidates (mark copies data invalid) all device data that this vector has allocated.
	 */
	template <typename T>
	inline void Vector<T>::invalidateDeviceData(bool enable) const
	{
		if (!enable)
			return;
	
#ifdef SKEPU_OPENCL
		invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
		if(m_noValidDeviceCopy)
			assert(m_valid);
   
		if(!m_noValidDeviceCopy)
		{
			invalidateDeviceData_CU();
			m_noValidDeviceCopy = true;
			m_valid = true;
		}
#endif
	}

	/*!
	 *  First updates the vector from its device allocations. Then invalidates (mark copies data invalid) the data allocated on devices.
	 */
	template <typename T>
	inline void Vector<T>::updateHostAndInvalidateDevice()
	{
		updateHost();
		invalidateDeviceData();
	}

	/*!
	 *  Removes the data copies allocated on devices.
	 */
	template <typename T>
	inline void Vector<T>::releaseDeviceAllocations()
	{
#ifdef SKEPU_OPENCL
		releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
		m_valid = true;
   
		releaseDeviceAllocations_CU();
#endif
	}

	/*!
	 *  First updates the vector from its device allocations. Then removes the data copies allocated on devices.
	 */
	template <typename T>
	inline void Vector<T>::updateHostAndReleaseDeviceAllocations()
	{
		updateHost();
		releaseDeviceAllocations();
	}


///////////////////////////////////////////////
// Public Helpers END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Operators START
///////////////////////////////////////////////

#ifdef SKEPU_ENABLE_DEPRECATED_OPERATOR

	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	const T& Vector<T>::operator[](const size_type index) const
	{
		updateHost();
//    updateHostAndInvalidateDevice();

		return m_data[index];
	}

#ifdef SKEPU_PRECOMPILED
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 *
	 *  Returns a proxy_elem instead of an ordinary element. The proxy_elem usually
	 *  behaves like an ordinary, but there might be exceptions.
	 */
	template <typename T>
	typename Vector<T>::proxy_elem Vector<T>::operator[](const size_type index)
	{
		return proxy_elem(*this, index);
	}
	
#else
	
	template<typename T>
	T& Vector<T>::operator[](const size_type index)
	{
		return this->m_data[index];
	}
	
#endif // SKEPU_PRECOMPILED
#endif // SKEPU_ENABLE_DEPRECATED_OPERATOR
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	Vector<T>& Vector<T>::operator=(const Vector<T>& other)
	{
		if (*this == other)
			return *this;
		 
		updateHostAndReleaseDeviceAllocations();
		other.updateHost();
		
		if (m_size < other.m_size)
		{
			if (m_data)
				backend::deallocateHostMemory<T>(m_data);
			
			m_size = m_size = other.m_size;
			
			backend::allocateHostMemory<T>(m_data, m_size);
		}
		else
		{
			m_size = other.m_size;
		}
		
		std::copy(other.m_data, other.m_data + m_size, m_data);
		
		return *this;
	}

///////////////////////////////////////////////
// Operators END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Regular interface functions START
///////////////////////////////////////////////
	
	template <typename T>
	typename Vector<T>::iterator Vector<T>::begin()
	{
		return iterator(*this, &this->m_data[0]);
	}
	
	template <typename T>
	typename Vector<T>::iterator Vector<T>::end()
	{
		return iterator(*this, &this->m_data[this->m_size]);
	}
	
	// Returns begin() if s >= 0, else returns end() - 1
	template <typename T>
	typename Vector<T>::strided_iterator Vector<T>::stridedBegin(size_t n, int s)
	{
		return strided_iterator(*this, &this->m_data[(s < 0) ? (-n + 1) * s : 0], s);
	}
	
	template <typename T>
	typename Vector<T>::const_iterator Vector<T>::begin() const
	{
		return const_iterator(*this, &this->m_data[0]);
	}
	
	template <typename T>
	typename Vector<T>::const_iterator Vector<T>::end() const
	{
		return const_iterator(*this, &this->m_data[this->m_size]);
	}
	
	// Returns begin() if s >= 0, else returns end() - 1
	template <typename T>
	typename Vector<T>::const_strided_iterator Vector<T>::stridedBegin(size_t n, int s) const
	{
		return const_strided_iterator(*this, &this->m_data[(s < 0) ? (-n + 1) * s : 0], s);
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	const T& Vector<T>::at(size_type loc) const
	{
		updateHost();
		return this->m_data[loc];
	}
	
#ifdef SKEPU_PRECOMPILED
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 *
	 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
	 *  behaves like an ordinary, but there might be exceptions.
	 */
	template <typename T>
	typename Vector<T>::proxy_elem Vector<T>::at(size_type loc)
	{
		return proxy_elem(*this, loc);
	}
	
#else
	
	template <typename T>
	T& Vector<T>::at(size_type loc)
	{
		return this->m_data[loc];
	}
	
#endif // SKEPU_PRECOMPILED
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::swap(Vector<T>& from)
	{
		updateHostAndReleaseDeviceAllocations();
		from.updateHostAndReleaseDeviceAllocations();
		
		std::swap(this->m_data, from.m_data);
		std::swap(this->m_size, from.m_size);
		std::swap(this->m_deallocEnabled, from.m_deallocEnabled);
	  std::swap(this->m_valid, from.m_valid);
	  std::swap(this->m_noValidDeviceCopy, from.m_noValidDeviceCopy);
		std::swap(this->m_label, from.m_label);
		std::swap(this->m_object_id, from.m_object_id);

#ifdef SKEPU_OPENCL
  	std::swap(this->m_deviceMemPointers_CL, from.m_deviceMemPointers_CL);
    std::swap(this->m_deviceConstMemPointers_CL, from.m_deviceConstMemPointers_CL);
#endif

#ifdef SKEPU_CUDA
	  std::swap(this->m_deviceMemPointers_CU, from.m_deviceMemPointers_CU);
	  std::swap(this->m_deviceMemPointers_Modified_CU, from.m_deviceMemPointers_Modified_CU);
#endif
	}

///////////////////////////////////////////////
// Regular interface functions END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Additions to interface START
///////////////////////////////////////////////

	/*!
	 *  Flushes the vector, synchronizing it with the device.
	 *
	 *  Then release all device allocations if deallocDevice is `true`.
	 */
	template <typename T>
	void Vector<T>::flush(FlushMode mode)
	{
#ifdef SKEPU_OPENCL
		this->flush_CL(mode);
#endif

#ifdef SKEPU_CUDA
		this->flush_CU(mode);
#endif
	}
	

///////////////////////////////////////////////
// Additions to interface END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Comparison operators START
///////////////////////////////////////////////

	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator==(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		if (m_size != c1.m_size)
			return false;
		
		if (m_data == c1.m_data)
			return true;
		
		for (size_type i = 0; i < m_size; i++)
		{
			if (m_data[i] != c1.m_data[i])
				return false;
		}
		return true;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator!=(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		if (m_size != c1.m_size)
			return true;
		
		if (m_data == c1.m_data && m_size == c1.m_size)
			return false;
		
		for (size_type i = 0; i < m_size; i++)
		{
			if (m_data[i] != c1.m_data[i])
				return true;
		}
		return false;
	}
	

///////////////////////////////////////////////
// Comparison operators END
///////////////////////////////////////////////

}


#include "vector_iterator.inl"

#ifdef SKEPU_PRECOMPILED

#include "vector_proxy.inl"
#include "vector_cl.inl"
#include "vector_cu.inl"

#endif // SKEPU_PRECOMPILED
