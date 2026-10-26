#ifdef SKEPU_OPENCL

namespace skepu
{
	/*!
	 *  \brief Update device with vector content.
	 *
	 *  Update device with a vector range. If vector does not have an allocation on the device for
	 *  the current range, create a new allocation and if specified, also copy vector data to device.
	 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so vector can keep track of where
	 *  and what it has stored on devices.
	 *
	 *  \param start Pointer to first element in range to be updated with device.
	 *  \param numElements Number of elemets in range.
	 *  \param device Pointer to the device that should be synched with.
	 *  \param copy Boolean value that tells whether to only allocate or also copy vector data to device. True copies, False only allocates.
	 */
	template <typename T>
	typename Vector<T>::device_pointer_type_cl Vector<T>::updateDevice_CL(T* start, size_type numElements, backend::Device_CL* device, bool copy)
	{
#ifdef SKEPU_CUDA
		syncCUDAbeforeOpenCL(true); // (the use may write: OpenCL says it only in the copy flag)
#endif
		T* root = &m_data[0];
		std::pair<cl_device_id, const T*> key(device->getDeviceID(), root);
		auto result = m_deviceMemPointers_CL.find(key);

		if (result == m_deviceMemPointers_CL.end()) //insert new, alloc mem and copy
		{
			auto temp = new backend::DeviceMemPointer_CL<T>{start, numElements, device, this->getLabel()};
#ifdef SKEPU_TRACING
			temp->m_container_id = this->getObjectID();
#endif
			if (copy)
			{
				// Make sure uptodate
				updateHost_CL();
				// Copy
				temp->copyHostToDevice();
			}
			result = m_deviceMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key,temp));
		}
		else if (result->second->m_effectiveHostDataPointer != start || result->second->m_numElements != numElements)
		{
			// A different part of the vector is requested: the existing allocation may hold the only up-to-date
			// copy of the data (written by a skeleton), so bring it back to the host before releasing it.
			updateHost_CL();
			m_deviceMemPointers_CL.erase(result->first);
			delete result->second;
			auto temp = new backend::DeviceMemPointer_CL<T>{start, numElements, device, this->getLabel()};
#ifdef SKEPU_TRACING
			temp->m_container_id = this->getObjectID();
#endif
			if (copy)
			{
				// Make sure uptodate
				updateHost_CL();
				// Copy
				temp->copyHostToDevice();
			}
			result = m_deviceMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key,temp));
		}
		// already exists, update from host if needed
		else if (copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
		{
#ifdef SKEPU_TRACING
			result->second->m_label = this->getLabel();
			result->second->m_container_id = this->getObjectID();
#endif
			// Make sure uptodate
			updateHost_CL(); // FIXT IT: Only check for this copy and not for all copies.
			// Copy
			result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
		}
		return result->second;
	}


	template <typename T>
	typename Vector<T>::device_const_pointer_type_cl Vector<T>::updateDevice_CL(const T* start, size_type numElements, backend::Device_CL* device, bool copy) const
	{
#ifdef SKEPU_CUDA
		syncCUDAbeforeOpenCL(false);
#endif
		const T* root = &m_data[0];
		std::pair< cl_device_id, const T* > key(device->getDeviceID(), root);
		auto result = m_deviceConstMemPointers_CL.find(key);

		if (result == m_deviceConstMemPointers_CL.end()) //insert new, alloc mem and copy
		{
			auto temp = new device_const_pointer_type_cl(start, numElements, device, this->getLabel());
#ifdef SKEPU_TRACING
			temp->m_label = this->getLabel();
			temp->m_container_id = this->getObjectID();
#endif
			if (copy)
			{
				temp->copyHostToDevice();
			}

			result = m_deviceConstMemPointers_CL.insert(m_deviceConstMemPointers_CL.begin(), std::make_pair(key,temp));
		}
		// Already exists, update from host if needed
		else if (copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
		{
#ifdef SKEPU_TRACING
			result->second->m_label = this->getLabel();
			result->second->m_container_id = this->getObjectID();
#endif
			result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
		}

		return result->second;
	}


	template <typename T>
	bool Vector<T>::isVectorOnDevice_CL(backend::Device_CL* device, bool multi) const
	{
		if(!multi)
		{
			typename std::map<std::pair< cl_device_id, std::pair< T*, size_type > >, device_pointer_type_cl >::iterator result;
			std::pair< cl_device_id, std::pair< T*, size_type > > key(device->getDeviceID(), std::pair< T*, size_type >(&m_data[0], size()));
			result = m_deviceMemPointers_CL.find(key);
			return !(result == m_deviceMemPointers_CL.end());
		}
		return false; // handle multi case here...
	}


	/*!
	 *  \brief Flushes the vector.
	 *
	 *  First it updates the vector from all its device allocations, then it releases all allocations.
	 */
	template <typename T>
	void Vector<T>::flush_CL(FlushMode mode)
	{
		DEBUG_TEXT_LEVEL3("Vector flush OpenCL\n")
		this->updateHost_CL();
		if (mode == FlushMode::Dealloc)
			this->releaseDeviceAllocations_CL();
	}


	/*!
	 *  \brief Updates the host from devices.
	 *
	 *  Updates the vector from all its device allocations.
	 */
	template <typename T>
	inline void Vector<T>::updateHost_CL() const
	{
		for (auto &memptr : this->m_deviceMemPointers_CL)
		{
#ifdef SKEPU_TRACING
			memptr.second->m_label = this->getLabel();
			memptr.second->m_container_id = this->getObjectID();
#endif
			memptr.second->copyDeviceToHost();
		}
	}


	/*!
	 *  \brief Invalidates the device data.
	 *
	 *  Invalidates the device data by releasing all allocations. This way the vector is updated
	 *  and then data must be copied back to devices if used again.
	 */
	template <typename T>
	inline void Vector<T>::invalidateDeviceData_CL() const
	{
		// deallocs all device mem for vector for now
		if (!this->m_deviceMemPointers_CL.empty() || !this->m_deviceConstMemPointers_CL.empty())
		{
			DEBUG_TEXT_LEVEL3("Invalidate device data OpenCL (" << this->size() << " elements)");
			releaseDeviceAllocations_CL();
		}
		// Could maybe be made better by only setting a flag that data is not valid
	}


	/*!
	 *  \brief Releases device allocations.
	 *
	 *  Releases all device allocations for this vector. The memory pointers are removed.
	 */
	template <typename T>
	inline void Vector<T>::releaseDeviceAllocations_CL() const
	{
		for (auto &memptr : this->m_deviceMemPointers_CL)
			delete memptr.second;

		m_deviceMemPointers_CL.clear();

		// (the copies of the const overload are not changed by OpenCL, but the data they hold is old after a write of another backend or of the host)
		for (auto &memptr : this->m_deviceConstMemPointers_CL)
			delete memptr.second;

		m_deviceConstMemPointers_CL.clear();
	}

#ifdef SKEPU_CUDA
	// CUDA and OpenCL in the same program (see the declarations in the class). The backends do not know of each other's copies, so
	// without this the one that uses the vector second would read the stale host memory, or an old copy of its own.
	template <typename T>
	void Vector<T>::syncCUDAbeforeOpenCL(bool mayWrite) const
	{
		if (m_noValidDeviceCopy)
			return; // CUDA has no copy
		DEBUG_TEXT_LEVEL1("BACKEND SWITCH CUDA -> OpenCL, label: " << this->getLabel() << ": the vector goes through the host memory"
			<< (m_valid ? ", CUDA has no changes" : ", CUDA has changes that are transferred to the host") << (mayWrite ? ", the CUDA copies are invalidated" : ""));
		if (!m_valid)
		{
			updateHost_CU(); // (the transfers are traced by the CUDA device pointers)
			m_valid = true;
		}
		if (mayWrite)
		{
			invalidateDeviceData_CU();
			m_noValidDeviceCopy = true;
		}
	}

	template <typename T>
	void Vector<T>::syncOpenCLbeforeCUDA(bool mayWrite) const
	{
		if (m_deviceMemPointers_CL.empty() && m_deviceConstMemPointers_CL.empty())
			return; // OpenCL has no copy
		DEBUG_TEXT_LEVEL1("BACKEND SWITCH OpenCL -> CUDA, label: " << this->getLabel() << ": the vector goes through the host memory"
			<< (mayWrite ? ", the OpenCL copies are released" : ""));
		updateHost_CL(); // (only copies that OpenCL has changed are transferred)
		if (mayWrite)
			releaseDeviceAllocations_CL();
	}
#endif
}

#endif
