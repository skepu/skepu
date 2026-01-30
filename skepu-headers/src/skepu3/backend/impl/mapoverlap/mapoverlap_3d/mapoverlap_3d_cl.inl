/*! \file mapoverlap_3d_cl.inl
*  \brief Contains the definitions of OpenCL specific member functions for the MapOverlap3D skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu
{
	namespace backend
	{

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
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
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
				get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity]))...);
			
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
			
			const size_t sharedK = numThreads[0] + this->m_overlap[2] * 2;
			const size_t sharedJ = numThreads[1] + this->m_overlap[1] * 2;
			const size_t sharedI = numThreads[2] + this->m_overlap[0] * 2;
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
				std::get<EI-OutArity>(elwiseMemP)...,
				std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-InArity-OutArity>(anyMemP))...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
				out_size_i, out_size_j, out_size_k, 
				this->m_overlap[0], this->m_overlap[1], this->m_overlap[2],
				in_size_i, in_size_j, in_size_k, 
				sharedI, sharedJ, sharedK,
				edge, pad, &wrapMemP,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-InArity-OutArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
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
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
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
					get<AI>(std::forward<CallArgs>(args)...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity]))...);
				
				CLKernel::mapOverlap3D(
					i, numThreads, numBlocks,
					std::get<OI>(outMemP)...,
					std::get<EI-OutArity>(elwiseMemP)...,
					std::make_tuple(&get<AI>(std::forward<CallArgs>(args)...).getParent(), std::get<AI-InArity-OutArity>(anyMemP))...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_info(),
					outRows, out_cols,
					this->m_overlap_y, this->m_overlap_x,
					in_cols, sharedRows, sharedCols,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-InArity-OutArity>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
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
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
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

    } // backend

} // skepu

#endif // SKEPU_OPENCL