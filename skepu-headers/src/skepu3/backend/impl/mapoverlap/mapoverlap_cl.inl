/*! \file mapoverlap_cl.inl
*  \brief Contains the definitions of OpenCL specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu
{
	namespace backend
	{
		/*!
		 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
		 *  Argument startIdx tell from where to perform a partial MapOverlap. Set startIdx=0 for MapOverlap of entire input.
		 *  The function uses only \em one device which is decided by a parameter. Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL(size_t deviceID, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			// Setup parameters
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			size_t numElem = arg.size() - startIdx;
			size_t overlap = (size_t)this->m_overlap[0];
			size_t n = numElem + std::min(startIdx, overlap);
			size_t out_offset = std::min(startIdx, overlap);

			size_t out_numelements;
			if (this->m_edge != skepu::Edge::None)
				out_numelements = numElem;
			else
				out_numelements = numElem - overlap * 2;
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap);
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHostAndInvalidateDevice();
				for (size_t i = 0; i < overlap; ++i)
				{
					wrap[i] = arg.end()(i - overlap);
					wrap[overlap+i] = arg(i);
				}
			}
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrap_mem_p(&wrap[0], wrap.size(), device, "Internal (overlap-wrap)");
#ifdef SKEPU_TRACING
			wrap_mem_p.m_container_id = UniqueIdentifier::generate();
#endif
			wrap_mem_p.copyHostToDevice();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			const size_t numThreads = std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
			const size_t numBlocks = std::max<size_t>(1, std::min(n / numThreads + (n % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks()));
			const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
			
			// Copy elements to device and allocate output memory.
			auto in_mem_p = arg.updateDevice_CL(arg.getAddress() + startIdx - out_offset, n, device, true);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...)
					.getAddress(), get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
			auto outMemP    = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + startIdx, out_numelements, device, false)...);
			
			size_t threads = std::min<size_t>(numElem, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(numElem, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			CLKernel::mapOverlapVector(
				deviceID, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				in_mem_p,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				&wrap_mem_p,
				n, overlap, out_offset, out_numelements, _poly, _pad,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 1D", this, {numElem},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}

		/*!
		 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
		 *  Argument startIdx tell from where to perform a partial MapOverlap. Set startIdx=0 for MapOverlap of entire input.
		 *  The function uses a variable number of devices, dividing the range of elemets equally among the participating devices each mapping
		 *  its part. Using \em OpenCL as backend.
		 */
		template <typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapNumDevices_CL(size_t numDevices, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			// Divide the elements amongst the devices
			const size_t size = arg.size() - startIdx;
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			const size_t overlap = (size_t)this->m_overlap[0];
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				for (size_t i = 0; i < overlap; ++i)
				{
					wrap[i] = arg.end()(i - overlap);
					wrap[overlap + i] = arg(i);
				}
			}
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices - 1) ? rest : 0);
				size_t out_offset, n;
				
				// Copy wrap vector to device.
				DeviceMemPointer_CL<T> wrap_mem_p(&wrap[0], wrap.size(), device);
#ifdef SKEPU_TRACING
				wrap_mem_p.m_label = "Internal (overlap-wrap)";
				wrap_mem_p.m_container_id = UniqueIdentifier::generate();
#endif
				wrap_mem_p.copyHostToDevice();
				
				// Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
				if (i == 0)                 { out_offset = std::min(overlap, startIdx); n = numElem + overlap * 2; }
				else if (i == numDevices-1) { out_offset = overlap; n = numElem + overlap; }
				else                        { out_offset = overlap; n = numElem + overlap * 2; }
				
				size_t out_numelements = numElem;
				auto in_mem_p = arg.updateDevice_CL(arg.getAddress() + startIdx + i * numElemPerSlice - out_offset, n, device, true);
				
				// Setup parameters
				const size_t numThreads = std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
				const size_t numBlocks = std::max<size_t>(1, std::min(n / numThreads + (n % numThreads == 0 ? 0:1), this->m_selected_spec->GPUBlocks()));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				// Allocate memory for output.
				auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + startIdx + i * numElemPerSlice, numElem, device, false)...);
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...)
					.getAddress(), get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
					
				size_t threads = std::min<size_t>(numElem, numBlocks * numThreads); // handle multiple devices
				auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(numElem, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
				
				CLKernel::mapOverlapVector(
					i, numThreads, numBlocks * numThreads,
					std::get<OI>(outMemP)...,
					randomMemP,
					in_mem_p,
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
					&wrap_mem_p,
					n, overlap, out_offset, out_numelements, _poly, _pad,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 1D", this, {size},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
		 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
		 *  on multiple devices, calling mapOverlapNumDevices_CL.
		 *  Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_OpenCL(size_t startIdx, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap Vector: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingle_CL(0, startIdx, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapNumDevices_CL(numDevices, startIdx, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
		
		/*!
		 * For Matrix overlap, we need to check whether overlap configuration is runnable considering total size of shared memory available on that system.
		 * This method is a helper funtion doing that. It is called by another helper \p getThreadNumber_CL() method.
		 *
		 * \param numThreads Number of threads in a thread block.
		 * \param deviceID The device ID.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<typename T>
		bool MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::sharedMemAvailable_CL(size_t &numThreads, size_t deviceID)
		{
			size_t overlap = (size_t)this->m_overlap[0];
			size_t maxShMem = this->m_environment->m_devices_CL.at(deviceID)->getSharedMemPerBlock() / sizeof(T) - SHMEM_SAFITY_BUFFER; // little buffer for other usage
			size_t orgThreads = numThreads;
			
			numThreads = (numThreads + 2 * overlap < maxShMem) ? numThreads : maxShMem - 2 * overlap;
			
			if (orgThreads == numThreads) // return true when nothing changed because of overlap constraint
				return true;
			
			if (numThreads < 8) // not original numThreads then atleast 8 threads should be there
				SKEPU_ERROR("Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted");
				
			return false;
		}
		
		
		/*!
		 * Helper method used for calculating optimal thread count. For row- or column-wise overlap,
		 * it determines a thread block size with perfect division of problem size.
		 *
		 * \param width The problem size.
		 * \param numThreads Number of threads in a thread block.
		 * \param deviceID The device ID.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<typename T>
		int MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::getThreadNumber_CL(size_t width, size_t numThreads, size_t deviceID)
		{
			// first check whether shared memory would be ok for this numThreads. Changes numThreads accordingly
			if (!sharedMemAvailable_CL<T>(numThreads, deviceID) && numThreads < 1)
				SKEPU_ERROR("Too low overlap size to continue.");
			
			if (width % numThreads == 0)
				return width / numThreads;
			
			for (size_t i = numThreads - 1; i >= 1; i--) // decreament numThreads and see which one is a perfect divisor
			{
				if (width % i == 0)
					return width / i;
			}
			return -1;
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Performs MapOverlap on the first rows of the matrix, specified by the numrows argument.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_Row(size_t deviceID, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t n = arg.total_cols()*numrows;
			const size_t overlap = (size_t)this->m_overlap[0];
			const size_t out_offset = 0;
			const size_t out_numelements = n;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t rowWidth = arg.total_cols(); // same as numcols
			size_t colWidth = arg.total_rows(); // same as numcols
			size_t trdsize = rowWidth;
			size_t blocksPerRow = 1;
			
			if (rowWidth > this->m_selected_spec->GPUThreads())
			{
				int tmp = getThreadNumber_CL<T>(rowWidth, maxThreads, deviceID);
				if (tmp == -1 || tmp * numrows > maxBlocks)
					SKEPU_ERROR("Row width is larger than maximum thread size: " << rowWidth << " " << maxThreads);
				
				blocksPerRow = tmp;
				trdsize = rowWidth / blocksPerRow;
				
				if (trdsize < overlap)
					SKEPU_ERROR("Cannot execute overlap with current overlap width");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap * numrows);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t row = 0; row < numrows; row++)
				{
					auto inputEndTemp = inputBeginTemp + rowWidth;
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + row * overlap * 2] = inputEndTemp(i - overlap);// inputEndMinusOverlap(i);
						wrap[i + row * overlap * 2 + overlap] = inputBeginTemp(i);
					}
					inputBeginTemp += rowWidth;
				}
			}
			/*    else
         wrap.resize(1); // not used so minimize overhead;*/
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
#ifdef SKEPU_TRACING
			wrapMemP.m_label = "Internal (overlap-wrap)";
			wrapMemP.m_container_id = UniqueIdentifier::generate();
#endif
			wrapMemP.copyHostToDevice();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			const size_t numThreads = trdsize; // std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
			const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerRow * numrows, this->m_selected_spec->GPUBlocks()));
			const size_t sharedMemSize = sizeof(T) * (numThreads+2*overlap);
			
			// Copy elements to device and allocate output memory.
			auto inMemP  = arg.updateDevice_CL(arg.getAddress(), numrows, rowWidth, device, true);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress(), numrows, rowWidth, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
				
			size_t threads = std::min<size_t>(n, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(n, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			CLKernel::mapOverlapMatrixRowWise(
				deviceID, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				inMemP,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().row_size_info(),
				&wrapMemP, n,
				overlap, out_offset, out_numelements, _poly, _pad, blocksPerRow, rowWidth,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap RowWise", this, {colWidth, rowWidth},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Performs MapOverlap on the first rows of the matrix, specified by the numrows argument.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_RowMulti(size_t numDevices, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			const size_t overlap = (size_t)this->m_overlap[0];
			size_t rowWidth = arg.total_cols();
			size_t colWidth = arg.total_rows();
			const size_t totalElems = rowWidth*numrows;
			size_t trdsize = rowWidth;
			size_t blocksPerRow = 1;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			
			if (rowWidth > maxThreads)
			{
				int tmp = getThreadNumber_CL<T>(rowWidth, maxThreads, 0);
				if (tmp == -1 || tmp * numrows > maxBlocks)
					SKEPU_ERROR("ERROR! Row width is larger than maximum thread size: " << rowWidth << " " << maxThreads);
				
				blocksPerRow = tmp;
				trdsize = rowWidth / blocksPerRow;
				
				if (trdsize < overlap)
					SKEPU_ERROR("ERROR! Cannot execute overlap with current overlap width.");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			const size_t numRowsPerSlice = numrows / numDevices;
			const size_t numElemPerSlice = numRowsPerSlice * rowWidth;
			const size_t restRows = numrows % numDevices;
			
			//Need to get new values from other devices so that the overlap between devices is up to date.
			//Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
		//	inputBegin.getParent().updateHostAndInvalidateDevice();
			
			std::vector<T> wrap(overlap * 2 * numrows);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t row = 0; row < numrows; row++)
				{
					auto inputEndTemp = inputBeginTemp+rowWidth;
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + row * overlap * 2] = inputEndTemp(i - overlap);// inputEndMinusOverlap(i);
						wrap[overlap + i + row * overlap * 2] = inputBeginTemp(i);
					}
					inputBeginTemp += rowWidth;
				}
			}
			
			DeviceMemPointer_CL<T> *in_mem_p[MAX_GPU_DEVICES], *wrap_mem_p[MAX_GPU_DEVICES];
			decltype(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress(), 0, 0, 0, false)...)) outMemP[MAX_GPU_DEVICES];
			
			// First create OpenCL memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				
				// Copy wrap vector to device.
				wrap_mem_p[i] = new DeviceMemPointer_CL<T>(&wrap[i * numRowsPerSlice * overlap * 2], &wrap[i * numRowsPerSlice * overlap * 2], numRowsPerSlice * overlap * 2, device);
#ifdef SKEPU_TRACING
				wrap_mem_p[i]->m_label = "Internal (overlap-wrap)";
				wrap_mem_p[i]->m_container_id = UniqueIdentifier::generate();
#endif
				
				// Copy elements to device and allocate output memory.
				in_mem_p[i]  = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numRows, rowWidth, device, false);
				outMemP[i] = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numElemPerSlice, numRows, rowWidth, device, false)...);
			}
			
			size_t out_offset = 0;
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			// we will divide the computation row-wise... copy data and do operation
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t numElems = numRows * rowWidth;
				
				// Copy wrap vector only if it is a CYCLIC overlap policy.
				if (this->m_edge == Edge::Cyclic)
					wrap_mem_p[i]->copyHostToDevice();
				
				const size_t numThreads = trdsize; //std::min(maxThreads, rowWidth);
				const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerRow * numRows, maxBlocks));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				// Copy elements to device and allocate output memory.
				in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numRows, rowWidth, device, true);
				
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
					get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
					
				size_t threads = std::min<size_t>(numElems, numBlocks * numThreads); // handle multiple devices
				auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(numElems, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
				
				CLKernel::mapOverlapMatrixRowWise(
					i, numThreads, numBlocks * numThreads,
					std::get<OI>(outMemP[i])...,
					randomMemP,
					in_mem_p[i],
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().row_size_info(),
					wrap_mem_p[i],
					numElems, overlap, out_offset, numElems, _poly, _pad,
					blocksPerRow, rowWidth,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP[i])->changeDeviceData(), 0)...);
			}
			
			// to properly de-allocate the memory
			for (size_t i = 0; i < numDevices; ++i)
				pack_expand((delete std::get<OI>(outMemP[i]), 0)...);
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap RowWise", this, {colWidth, rowWidth},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_OpenCL(size_t numrows, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingle_CL_Row(0, numrows, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapSingle_CL_RowMulti(numDevices, numrows, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_Col(size_t deviceID, size_t _numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t n = arg.size();
			const size_t overlap = (size_t)this->m_overlap[0];
			const size_t out_offset = 0;
			const size_t out_numelements = n;
			const size_t colWidth = arg.total_rows();
			const size_t numcols = arg.total_cols();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t trdsize = colWidth;
			size_t blocksPerCol = 1;
			
			if (colWidth > maxThreads)
			{
				int tmp = getThreadNumber_CL<T>(colWidth, maxThreads, deviceID);
				if (tmp == -1 || blocksPerCol * numcols > maxBlocks)
					SKEPU_ERROR("Col width is larger than maximum thread size: " << colWidth << " " << maxThreads);
				
				blocksPerCol = tmp;
				trdsize = colWidth / blocksPerCol;
				if (trdsize < overlap)
					SKEPU_ERROR("Thread size should be larger than overlap width");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(overlap * 2 * (n / colWidth));
			
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t col=0; col< numcols; col++)
				{
					auto inputEndTemp = inputBeginTemp + numcols * (colWidth - 1);
					auto inputEndMinusOverlap = (inputEndTemp - (overlap - 1) * numcols);
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + col * overlap * 2] = inputEndMinusOverlap(i * numcols);
						wrap[overlap + i + col * overlap * 2] = inputBeginTemp(i * numcols);
					}
					inputBeginTemp++;
				}
			}
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
#ifdef SKEPU_TRACING
			wrapMemP.m_label = "Internal (overlap-wrap)";
			wrapMemP.m_container_id = UniqueIdentifier::generate();
#endif
			wrapMemP.copyHostToDevice();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			const size_t numThreads = trdsize; //std::min(maxThreads, rowWidth);
			const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerCol * numcols, maxBlocks));
			const size_t sharedMemSize = sizeof(T) * (numThreads+2*overlap);
			
			// Copy elements to device and allocate output memory.
			auto inMemP  = arg.updateDevice_CL(arg.getAddress(), colWidth, numcols, device, true);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress(), colWidth, numcols, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
					
			size_t threads = std::min<size_t>(n, numBlocks * numThreads);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(n, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			CLKernel::mapOverlapMatrixColWise(
				deviceID, numThreads, numBlocks * numThreads,
				std::get<OI>(outMemP)...,
				randomMemP,
				inMemP,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().col_size_info(),
				&wrapMemP,
				n, overlap, out_offset, out_numelements, _poly, _pad,
				blocksPerCol, numcols, colWidth,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap ColWise", this, {colWidth, numcols},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		*  Performs the column-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		*  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_ColMulti(size_t numDevices, size_t _numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			const size_t totalElems = arg.size();
			const size_t overlap = (size_t)this->m_overlap[0];
			const size_t colWidth = arg.total_rows();
			const size_t numCols = arg.total_cols();
			const size_t numRowsPerSlice = colWidth / numDevices;
			const size_t numElemPerSlice = numRowsPerSlice * numCols;
			const size_t restRows = colWidth % numDevices;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			
			std::vector<T> wrapStartDev(overlap * numCols), wrapEndDev(overlap * numCols);
			
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t col = 0; col < numCols; col++)
				{
					auto inputEndTemp = inputBeginTemp + numCols * (colWidth - 1);
					auto inputEndMinusOverlap = inputEndTemp - (overlap - 1) * numCols;
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrapStartDev[i + col * overlap] = inputEndMinusOverlap(i * numCols);
						wrapEndDev[i + col * overlap] = inputBeginTemp(i * numCols);
					}
					inputBeginTemp++;
				}
			}
			
			DeviceMemPointer_CL<T> *in_mem_p[MAX_GPU_DEVICES];
			decltype(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress(), 0, 0, 0, false)...)) outMemP[MAX_GPU_DEVICES];
			DeviceMemPointer_CL<T> *wrapStartDev_mem_p[MAX_GPU_DEVICES], *wrapEndDev_mem_p[MAX_GPU_DEVICES];
			
			size_t trdsize[MAX_GPU_DEVICES];
			size_t blocksPerCol[MAX_GPU_DEVICES];
			
			// First create OpenCL memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				
				trdsize[i] = numRows;
				blocksPerCol[i] = 1;
				
				if (numRows > maxThreads)
				{
					int tmp = getThreadNumber_CL<T>(numRows, maxThreads, i);
					if (tmp == -1 || tmp * numCols > maxBlocks)
						SKEPU_ERROR("Col width is larger than maximum thread size: " << colWidth << " " << maxThreads);
					
					blocksPerCol[i] = tmp;
					trdsize[i] = numRows / blocksPerCol[i];
					
					if (trdsize[i] < overlap)
						SKEPU_ERROR("Cannot execute overlap with current overlap width.");
				}
				
				size_t overlapElems = overlap*numCols;
				
				if (i == 0)                 in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice,                numRows + overlap,     numCols, device, false);
				else if (i == numDevices-1) in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + overlap,     numCols, device, false);
				else                        in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + overlap * 2, numCols, device, false);
				
				outMemP[i] = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numElemPerSlice, numRows, numCols, device, false)...);
				
				// wrap vector, don't copy just allocates space on OpenCL device.
				wrapStartDev_mem_p[i] = new DeviceMemPointer_CL<T>(&wrapStartDev[0], &wrapStartDev[0], wrapStartDev.size(), device);
				wrapEndDev_mem_p[i] = new DeviceMemPointer_CL<T>(&wrapEndDev[0], &wrapEndDev[0], wrapEndDev.size(), device);
			}
			
			int _deviceType[MAX_GPU_DEVICES];
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			
			// we will divide the computation row-wise... copy data and do operation
			for (size_t i = 0; i < numDevices; i++)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t numElems = numRows * numCols;
				size_t in_offset, overlapElems = overlap * numCols;
				
				// Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
				if (i == 0)
				{
					_deviceType[i] = -1;
					
					if (this->m_edge == Edge::Cyclic)
						wrapStartDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?
					
					in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numRows + overlap, numCols, device, true);
					in_offset = 0;
				}
				else if (i == numDevices-1)
				{
					_deviceType[i] = 1;
					
					// copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for multiple devices?
					if (this->m_edge == Edge::Cyclic)
						wrapEndDev_mem_p[i]->copyHostToDevice(); 
					
					in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + overlap, numCols, device, true);
					in_offset = overlapElems;
				}
				else
				{
					_deviceType[i] = 0;
					in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + 2 * overlap, numCols, device, true);
					in_offset = overlapElems;
				}
				
				const size_t numThreads = trdsize[i]; //std::min(maxThreads, rowWidth);
				const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerCol[i] * numCols, maxBlocks));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				DeviceMemPointer_CL<T> *useWrapMemP = (this->m_edge == Edge::Cyclic && i == numDevices - 1)
					? wrapEndDev_mem_p[i] : wrapStartDev_mem_p[i]; 
				
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
					get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
					
				size_t threads = std::min<size_t>(numElems, numBlocks * numThreads); // handle multiple devices
				auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(numElems, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
				
				CLKernel::mapOverlapMatrixColWiseMulti(
					i, numThreads, numBlocks * numThreads,
					std::get<OI>(outMemP[i])...,
					randomMemP,
					in_mem_p[i],
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().col_size_info(),
					useWrapMemP,
					numElems, overlap, in_offset,
					numElems, _poly, _deviceType[i], _pad,
					blocksPerCol[i], numCols, numRows,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP[i])->changeDeviceData(), 0)...);
			}
			
			// Properly de-allocate the memory
			for (size_t i = 0; i < numDevices; ++i)
			{
				delete wrapStartDev_mem_p[i];
				delete wrapEndDev_mem_p[i];
			}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap ColWise", this, {colWidth, numCols},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
		 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
		 *  on multiple devices, calling mapOverlapNumDevices_CL.
		 *  Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_OpenCL(size_t numcols, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->mapOverlapSingle_CL_Col(0, numcols, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				return this->mapOverlapSingle_CL_ColMulti(numDevices, numcols, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
		
		
		/*!
		 *  Performs the 2D MapOverlap using a single OpenCL GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapSingleThread_CL(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t in_rows  = arg.total_rows();
			const size_t in_cols  = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int edge = static_cast<int>(this->m_edge);
			const T pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			std::vector<T> wrap(1);
			
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress(), in_rows,  in_cols,  device, true)...);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getParent().getAddress(), out_rows, out_cols, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
				
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
#ifdef SKEPU_TRACING
			wrapMemP.m_label = "Internal (overlap-wrap)";
			wrapMemP.m_container_id = UniqueIdentifier::generate();
#endif
			wrapMemP.copyHostToDevice();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			size_t numThreads[2], numBlocks[2];
			numThreads[0] = std::min<size_t>(out_cols, 16);
			numThreads[1] = std::min(out_rows, maxThreads / 16);
			numBlocks[0] = (size_t)((out_cols + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
			numBlocks[1] = (size_t)((out_rows + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
			
			const size_t sharedCols = numThreads[0] + this->m_overlap_x * 2;
			const size_t sharedRows = numThreads[1] + this->m_overlap_y * 2;
			const size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);
			
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 2D: device = " << deviceID << ", numThreads = "
				<< "[" << numThreads[0] << " x " << numThreads[1] << "]" 
				<< ", numBlocks = " << "[" << numBlocks[0] << " x " << numBlocks[1] << "], shmem = " << sharedMemSize); 
			
			size_t threads = std::min<size_t>(out_rows * out_cols, numBlocks[0] * numBlocks[1] * numThreads[0] * numThreads[1]);
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(out_rows * out_cols, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			CLKernel::mapOverlap2D(
				deviceID, numThreads, numBlocks,
				std::get<OI>(outMemP)...,
				randomMemP,
				std::get<EI-outArity>(elwiseMemP)...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				out_rows, out_cols,
				this->m_overlap_y, this->m_overlap_x,
				in_rows, in_cols, sharedRows, sharedCols,
				edge, pad, &wrapMemP,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 2D", this, {in_rows, in_cols},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the 2D MapOverlap using multiple OpenCL GPUs.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapMultipleThread_CL(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			const size_t in_rows  = arg.total_rows();
			const size_t in_cols  = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			const size_t numRowsPerSlice = out_rows / numDevices;
			const size_t restRows = out_rows % numDevices;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int edge = static_cast<int>(this->m_edge);
			const T pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			std::vector<T> wrap(1);
			
			// Need to get new values from other devices so that the overlap between devices is up to date.
			// Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
			arg.updateHostAndInvalidateDevice();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t outRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t inRows = outRows + 2 * this->m_overlap_y; // no matter which device, number of input rows is same.
				
				auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * in_cols, inRows, in_cols, device, true)...);
				auto outMemP    = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * out_cols, outRows, out_cols, device, false)...);
			
				// Copy wrap vector to device.
				DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
#ifdef SKEPU_TRACING
				wrapMemP.m_label = "Internal (overlap-wrap)";
				wrapMemP.m_container_id = UniqueIdentifier::generate();
#endif
				wrapMemP.copyHostToDevice();
				
				size_t numBlocks[2], numThreads[2];
				numThreads[0] = std::min<size_t>(out_cols, 16);
				numThreads[1] = std::min(out_rows, maxThreads / 16);
				numBlocks[0] = (size_t)((out_cols + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
				numBlocks[1] = (size_t)((outRows + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
				
				const size_t sharedRows = numThreads[1] + this->m_overlap_y * 2;
				const size_t sharedCols = numThreads[0] + this->m_overlap_x * 2;
				const size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);
				const size_t stride = numThreads[0] + this->m_overlap_x * 2;
			
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
					get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
					
				size_t threads = std::min<size_t>(outRows * out_cols, numBlocks[0] * numBlocks[1] * numThreads[0] * numThreads[1]);
				auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(outRows * out_cols, threads);
				auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
				
				CLKernel::mapOverlap2D(
					i, numThreads, numBlocks,
					std::get<OI>(outMemP)...,
					randomMemP,
					std::get<EI-outArity>(elwiseMemP)...,
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
					outRows, out_cols,
					this->m_overlap_y, this->m_overlap_x,
					in_rows, in_cols, sharedRows, sharedCols,
					edge, pad, &wrapMemP,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
				SKEPU_TRACE_CALL(trace_handle, "MapOverlap 2D", this, {in_rows, in_cols},
					{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
					{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
					{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
					tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
				);
			}
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenCL(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 2D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CL(0, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				return this->mapOverlapMultipleThread_CL(numDevices, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
		
		
		
		
		
		
		
		
		
		
		/*!
		 *  Performs the 3D MapOverlap using a single OpenCL GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapSingleThread_CL(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t in_size_i  = arg.size_i();
			const size_t in_size_j  = arg.size_j();
			const size_t in_size_k  = arg.size_k();
			const size_t out_size_i = res.size_i();
			const size_t out_size_j = res.size_j();
			const size_t out_size_k = res.size_k();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int edge = static_cast<int>(this->m_edge);
			const T pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			std::vector<T> wrap(1);
			
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress(), in_size_i * in_size_j * in_size_k, device, true)...);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getParent().getAddress(), out_size_i * out_size_j * out_size_k, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
#ifdef SKEPU_TRACING
			wrapMemP.m_label = "Internal (overlap-wrap)";
			wrapMemP.m_container_id = UniqueIdentifier::generate();
#endif
			wrapMemP.copyHostToDevice();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			size_t numThreads[3], numBlocks[3];
			size_t sizeLength = (size_t)std::cbrt(maxThreads);
			numThreads[0] = std::min<size_t>(out_size_k, sizeLength);
			numThreads[1] = std::min<size_t>(out_size_j, std::min<size_t>(maxThreads / numThreads[0], sizeLength));
			numThreads[2] = std::min(out_size_i, maxThreads / (numThreads[0] * numThreads[1]));
			numBlocks[0] = (size_t)((out_size_k + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
			numBlocks[1] = (size_t)((out_size_j + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
			numBlocks[2] = (size_t)((out_size_i + numThreads[2] - 1) / numThreads[2]) * numThreads[2];
			
			const size_t sharedK = numThreads[0] + this->m_overlap_k * 2;
			const size_t sharedJ = numThreads[1] + this->m_overlap_j * 2;
			const size_t sharedI = numThreads[2] + this->m_overlap_i * 2;
			const size_t sharedMemSize =  sharedI * sharedJ * sharedK * sizeof(T);
			
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 3D: device = " << deviceID << ", numThreads = "
				<< "[" << numThreads[0] << " x " << numThreads[1] << " x " << numThreads[2] << "]" 
				<< ", numBlocks = " << "[" << numBlocks[0] << " x " << numBlocks[1] << " x " << numBlocks[2] << "], shmem = " << sharedMemSize);
					
			size_t threads = std::min<size_t>(out_size_i * out_size_j * out_size_k, numBlocks[0] * numBlocks[1] * numBlocks[2] * numThreads[0] * numThreads[1] * numThreads[2]); // handle division factor
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(out_size_i * out_size_j * out_size_k, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			CLKernel::mapOverlap3D(
				deviceID, numThreads, numBlocks,
				std::get<OI>(outMemP)...,
				randomMemP,
				std::get<EI-outArity>(elwiseMemP)...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				out_size_i, out_size_j, out_size_k, 
				this->m_overlap_i, this->m_overlap_j, this->m_overlap_k,
				in_size_i, in_size_j, in_size_k, 
				sharedI, sharedJ, sharedK,
				edge, pad, &wrapMemP,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 3D", this, {in_size_i, in_size_j, in_size_k},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the 3D MapOverlap using multiple OpenCL GPUs.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapMultipleThread_CL(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			const size_t in_rows   = arg.size_i();
			const size_t in_cols   = arg.size_j();
			const size_t in_peaks  = arg.size_k();
			const size_t out_rows  = res.size_i();
			const size_t out_cols  = res.size_j();
			const size_t out_peaks = res.size_k();
			const size_t numRowsPerSlice = out_rows / numDevices;
			const size_t restRows = out_rows % numDevices;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Need to get new values from other devices so that the overlap between devices is up to date.
			// Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
			arg.updateHostAndInvalidateDevice();
			/*
			SKEPU_TRACE_START_EVENT(trace_handle);
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t outRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t inRows = outRows + 2 * this->m_overlap_y; // no matter which device, number of input rows is same.
				
				auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * in_cols, inRows, in_cols, device, true)...);
				auto outMemP    = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * out_cols, outRows, out_cols, device, false)...);
				
				size_t numBlocks[2], numThreads[2];
				numThreads[0] = std::min<size_t>(out_cols, 16);
				numThreads[1] = std::min(out_rows, maxThreads / 16);
				numBlocks[0] = (size_t)((out_cols + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
				numBlocks[1] = (size_t)((outRows + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
				
				const size_t sharedRows = numThreads[1] + this->m_overlap_y * 2;
				const size_t sharedCols = numThreads[0] + this->m_overlap_x * 2;
				const size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);
				const size_t stride = numThreads[0] + this->m_overlap_x * 2;
			
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
					get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
				
				CLKernel::mapOverlap3D(
					i, numThreads, numBlocks,
					std::get<OI>(outMemP)...,
					std::get<EI-outArity>(elwiseMemP)...,
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
					outRows, out_cols,
					this->m_overlap_y, this->m_overlap_x,
					in_cols, sharedRows, sharedCols,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 3D", this, {size_i, size_j, size_k},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
			*/
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenCL(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 3D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CL(0, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				return this->mapOverlapMultipleThread_CL(numDevices, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
		
		
		
		
		
		
		
		
		
		/*!
		 *  Performs the 4D MapOverlap using a single OpenCL GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapSingleThread_CL(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t in_size_i  = arg.size_i();
			const size_t in_size_j  = arg.size_j();
			const size_t in_size_k  = arg.size_k();
			const size_t in_size_l  = arg.size_l();
			const size_t out_size_i = res.size_i();
			const size_t out_size_j = res.size_j();
			const size_t out_size_k = res.size_k();
			const size_t out_size_l = res.size_l();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			const std::string name = isPool ? "MapPool" : "MapOverlap";
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int edge = static_cast<int>(this->m_edge);
			const T pad = (this->m_edge == Edge::Pad) ? this->m_pad : T{};
			std::vector<T> wrap(1);
			
			auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress(), in_size_i * in_size_j * in_size_k * in_size_l, device, true)...);
			auto outMemP = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getParent().getAddress(), out_size_i * out_size_j * out_size_k * out_size_l, device, false)...);
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
#ifdef SKEPU_TRACING
			wrapMemP.m_label = "Internal (overlap-wrap)";
			wrapMemP.m_container_id = UniqueIdentifier::generate();
#endif
			wrapMemP.copyHostToDevice();
			
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			size_t numThreads[3], numBlocks[3];
			size_t sizeLength = (size_t)std::sqrt(std::sqrt(maxThreads));
			size_t numThreads_0a = std::min<size_t>(out_size_l, sizeLength);
			
			// Workaround to get full access to the innermost dimension,
			// i.e., do not split innermost dimension between blocks 
			bool atomicInnerMostDimension = true;
			if (atomicInnerMostDimension) numThreads_0a = out_size_l;
				
			numThreads[0] = std::min<size_t>(out_size_k, std::min<size_t>(maxThreads / numThreads_0a, sizeLength)) * numThreads_0a;
			numThreads[1] = std::min<size_t>(out_size_j, std::min<size_t>(maxThreads / numThreads[0], sizeLength));
			numThreads[2] = std::min(out_size_i, maxThreads / (numThreads[0] * numThreads[1]));
			
			
			size_t numBlocks_0a = (size_t)((out_size_l + numThreads_0a - 1) / numThreads_0a) * numThreads_0a;
			numBlocks[0] = (size_t)((out_size_k * out_size_l + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
			numBlocks[1] = (size_t)((out_size_j + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
			numBlocks[2] = (size_t)((out_size_i + numThreads[2] - 1) / numThreads[2]) * numThreads[2];
			
		/*	const size_t sharedL = numThreads_0a + this->m_overlap_l * 2;
			const size_t sharedK = (numThreads[0] / numThreads_0a) + this->m_overlap_k * 2;
			const size_t sharedJ = numThreads[1] + this->m_overlap_j * 2;
			const size_t sharedI = numThreads[2] + this->m_overlap_i * 2;*/
			 size_t sharedL = std::min<size_t>(in_size_l, numThreads_0a) + this->m_overlap_l * 2;
			 size_t sharedK = std::min<size_t>(in_size_k, numThreads[0] / numThreads_0a) + this->m_overlap_k * 2;
			 size_t sharedJ = std::min<size_t>(in_size_j, numThreads[1]) + this->m_overlap_j * 2;
			 size_t sharedI = std::min<size_t>(in_size_i, numThreads[2]) + this->m_overlap_i * 2;
			 size_t sharedMemSize =  sharedI * sharedJ * sharedK * sharedL * sizeof(T);
			
			if (isPool)
			{
				sharedL = std::min<size_t>(out_size_l, numThreads_0a) * this->m_overlap_l;
				sharedK = std::min<size_t>(out_size_k, numThreads[0] / numThreads_0a) * this->m_overlap_k;
				sharedJ = std::min<size_t>(out_size_j, numThreads[1]) * this->m_overlap_j;
				sharedI = std::min<size_t>(out_size_i, numThreads[2]) * this->m_overlap_i;
				sharedMemSize =  sharedI * sharedJ * sharedK * sharedL * sizeof(T);
			}
			
			DEBUG_TEXT_LEVEL1("OpenCL " << name << " 4D: device = " << deviceID << ", numThreads = "
				<< "[(" << numThreads_0a << ") " << numThreads[0] << " x " << numThreads[1] << " x " << numThreads[2] << "]" 
				<< ", numBlocks = " << "[(" << numBlocks_0a << ") " << numBlocks[0] << " x " << numBlocks[1] << " x " << numBlocks[2] << "], shmem = " << sharedMemSize);
				
			size_t threads = std::min<size_t>(out_size_i * out_size_j * out_size_k * out_size_l, numBlocks[0] * numBlocks[1] * numBlocks[2] * numThreads[0] * numThreads[1] * numThreads[2]); // handle division factor
			auto random = this->template prepareRandom_CL<MapOverlapFunc::randomCount>(out_size_i * out_size_j * out_size_k * out_size_l, threads);
			auto randomMemP = random.updateDevice_CL(random.getAddress(), threads, device, true);
			
			CLKernel::mapOverlap4D(
				deviceID, numThreads, numBlocks,
				std::get<OI>(outMemP)...,
				randomMemP,
				std::get<EI-outArity>(elwiseMemP)...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				out_size_i, out_size_j, out_size_k, out_size_l, 
				this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l,
				in_size_i, in_size_j, in_size_k, in_size_l, 
				sharedI, sharedJ, sharedK, sharedL,
				numThreads_0a, numBlocks_0a,
				(int)(isPool ? 1 : 0), edge, pad, &wrapMemP,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 4D", this, {in_size_i, in_size_j, in_size_k, in_size_l},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the 4D MapOverlap using multiple OpenCL GPUs.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapMultipleThread_CL(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			const size_t in_rows   = arg.size_i();
			const size_t in_cols   = arg.size_j();
			const size_t in_peaks  = arg.size_k();
			const size_t out_rows  = res.size_i();
			const size_t out_cols  = res.size_j();
			const size_t out_peaks = res.size_k();
			const size_t numRowsPerSlice = out_rows / numDevices;
			const size_t restRows = out_rows % numDevices;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Need to get new values from other devices so that the overlap between devices is up to date.
			// Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
			arg.updateHostAndInvalidateDevice();
			/*
			SKEPU_TRACE_START_EVENT(trace_handle);
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t outRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t inRows = outRows + 2 * this->m_overlap_y; // no matter which device, number of input rows is same.
				
				auto elwiseMemP = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<EI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * in_cols, inRows, in_cols, device, true)...);
				auto outMemP    = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * out_cols, outRows, out_cols, device, false)...);
				
				size_t numBlocks[2], numThreads[2];
				numThreads[0] = std::min<size_t>(out_cols, 16);
				numThreads[1] = std::min(out_rows, maxThreads / 16);
				numBlocks[0] = (size_t)((out_cols + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
				numBlocks[1] = (size_t)((outRows + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
				
				const size_t sharedRows = numThreads[1] + this->m_overlap_y * 2;
				const size_t sharedCols = numThreads[0] + this->m_overlap_x * 2;
				const size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);
				const size_t stride = numThreads[0] + this->m_overlap_x * 2;
			
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent().updateDevice_CL(get<AI>(std::forward<CallArgs>(args)...).getAddress(),
					get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity]))...);
				
				CLKernel::mapOverlap3D(
					i, numThreads, numBlocks,
					std::get<OI>(outMemP)...,
					std::get<EI-outArity>(elwiseMemP)...,
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-arity-outArity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
					outRows, out_cols,
					this->m_overlap_y, this->m_overlap_x,
					in_cols, sharedRows, sharedCols,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity-outArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 4D", this, {size_i, size_j, size_k, size_l},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);*/
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenCL(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 4D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CL(0, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				return this->mapOverlapMultipleThread_CL(numDevices, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
	}
}

#endif
