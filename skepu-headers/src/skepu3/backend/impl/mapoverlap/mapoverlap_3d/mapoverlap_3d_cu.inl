/*! \file mapoverlap_3d_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapOverlap3D skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <skepu3/matrix.hpp>

namespace skepu
{
	namespace backend
	{

		/*!
		 *  Performs the 3D MapOverlap using a single CUDA GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapSingleThread_CU(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			const size_t in_size_i = arg.size_i();
			const size_t in_size_j = arg.size_j();
			const size_t in_size_k = arg.size_k();
			const size_t out_size_i = res.size_i();
			const size_t out_size_j = res.size_j();
			const size_t out_size_k = res.size_k();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			cudaSetDevice(deviceID);
			
			auto in_mem_p = arg.updateDevice_CU(arg.getAddress(), in_size_i * in_size_j * in_size_k, deviceID, AccessMode::Read, true);
			auto out_mem_p = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CU(get<OI>(std::forward<CallArgs>(args)...).getAddress(), out_size_i * out_size_j * out_size_k, deviceID, AccessMode::Write, true)...);
			
			dim3 numBlocks, numThreads;
			
			size_t sizeLength = (size_t)std::cbrt(maxThreads);
			numThreads.x = std::min<size_t>(out_size_k, sizeLength);
			numThreads.y = std::min<size_t>(out_size_j, std::min<size_t>(maxThreads / numThreads.x, sizeLength));
			numThreads.z = std::min(out_size_i, maxThreads / (numThreads.x * numThreads.y));
			
			numBlocks.x = (out_size_k + numThreads.x - 1) / numThreads.x;
			numBlocks.y = (out_size_j + numThreads.y - 1) / numThreads.y;
			numBlocks.z = (out_size_i + numThreads.z - 1) / numThreads.z;
			
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).cudaProxy(deviceID, MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])...);
		
			// PRNG support
			size_t prng_threads = std::min<size_t>(out_size_i * out_size_j * out_size_k, numBlocks.x * numBlocks.y * numBlocks.z * numThreads.x * numThreads.y * numThreads.z);
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(out_size_i * out_size_j * out_size_k, prng_threads);
			auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, deviceID, AccessMode::ReadWrite);
			
			size_t sharedMem =  (numThreads.x + this->m_overlap[2] * 2) * (numThreads.y + this->m_overlap[1] * 2) * (numThreads.z + this->m_overlap[0] * 2) * sizeof(T);
			
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap kernel: size = " << out_size_i * out_size_j * out_size_k << ", one device, numBlocks = [" << numBlocks.x << "x" << numBlocks.y << "x" << numBlocks.z << "], numThreads = [" << numThreads.x << "x" << numThreads.y << "x" << numThreads.z << "]");
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks, numThreads, sharedMem, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_kernel<<<numBlocks, numThreads, sharedMem>>>
#endif
			(
				std::get<OI>(out_mem_p)->getDeviceDataPointer()...,
				randomMemP->getDeviceDataPointer(),
				in_mem_p->getDeviceDataPointer(), 
				std::get<AI-InArity-OutArity>(anyMemP).second...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				in_size_i, in_size_j, in_size_k,
				out_size_i, out_size_j, out_size_k,
				this->m_overlap[0], this->m_overlap[1], this->m_overlap[2],
				numThreads.z + this->m_overlap[0] * 2,
				numThreads.y + this->m_overlap[1] * 2,
				numThreads.x + this->m_overlap[2] * 2,
				this->m_edge, this->m_pad
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<OI>(out_mem_p)->changeDeviceData(), 0)...);
		}
		
		
		/*!
		*  Performs the 3D MapOverlap using multiple CUDA GPUs.
		*  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapMultipleThread_CU(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			const size_t in_size_i = arg.size_i();
			const size_t in_size_j = arg.size_j();
			const size_t in_size_k = arg.size_k();
			const size_t out_size_i = res.size_i();
			const size_t out_size_j = res.size_j();
			const size_t out_size_k = res.size_k();
			
			const size_t numRowsPerSlice = out_size_i / numDevices;
			const size_t restRows = out_size_i % numDevices;
			
			// Need to get new values from other devices so that the overlap between devices is up to date.
			// Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
		//	input.updateHostAndInvalidateDevice();
			/*
			typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
			decltype(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CU(get<OI>(std::forward<CallArgs>(args)...).getAddress(), 0, 0, AccessMode::None, true)...)) out_mem_p[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				
				size_t outRows;
				if (i == numDevices-1)
					outRows = numRowsPerSlice+restRows;
				else
					outRows = numRowsPerSlice;
				
				size_t inRows = outRows + this->m_overlap_y * 2; // no matter which device, number of input rows is same.
				
				in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numRowsPerSlice * in_cols, inRows * in_cols, i, AccessMode::None, true);
				out_mem_p[i] = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CU(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * out_cols, outRows * out_cols, i, AccessMode::None, true)...);
			}
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				
				size_t outRows;
				if (i == numDevices-1)
					outRows = numRowsPerSlice+restRows;
				else
					outRows = numRowsPerSlice;
				
				size_t inRows = outRows + this->m_overlap_y * 2; // no matter which device, number of input rows is same.
				
				in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numRowsPerSlice * in_cols, inRows * in_cols, i, AccessMode::Read, true);
				out_mem_p[i] = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CU(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * out_cols, outRows * out_cols, i, AccessMode::Write, true, true)...);
				
				dim3 numBlocks, numThreads;
			
				size_t sizeLength = (size_t)std::cbrt(maxThreads);
				numThreads.x = std::min<size_t>(out_size_k, sizeLength);
				numThreads.y = std::min<size_t>(out_size_j, std::min<size_t>(maxThreads / numThreads.x, sizeLength));
				numThreads.z = std::min(out_size_i, maxThreads / (numThreads.x * numThreads.y));
				
				numBlocks.x = (out_size_k + numThreads.x - 1) / numThreads.x;
				numBlocks.y = (out_size_j + numThreads.y - 1) / numThreads.y;
				numBlocks.z = (out_size_i + numThreads.z - 1) / numThreads.z;
				
				size_t sharedMem =  (numThreads.x + this->m_overlap_x * 2) * (numThreads.y + this->m_overlap_y * 2) * sizeof(T);
				
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).cudaProxy(i, MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])...);
		
				// PRNG support
				size_t prng_threads = std::min<size_t>(out_size_i * out_size_j * out_size_k, numBlocks.x * numBlocks.y * numBlocks.z * numThreads.x * numThreads.y * numThreads.z);
				auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(out_size_i * out_size_j * out_size_k, prng_threads);
				auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, i, AccessMode::ReadWrite);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<< numBlocks,numThreads, sharedMem, this->m_environment->m_devices_CU[i]->m_streams[0]>>>
#else
				this->m_cuda_kernel<<< numBlocks,numThreads, sharedMem >>>
#endif
				(
					std::get<OI>(out_mem_p[i])->getDeviceDataPointer()...,
					randomMemP->getDeviceDataPointer(),
					in_mem_p[i]->getDeviceDataPointer(), 
					std::get<AI-arity-OutArity>(anyMemP).second...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					in_size_i, in_size_j, in_size_k,
					out_size_i, out_size_j, out_size_k,
					this->m_overlap_i, this->m_overlap_j, this->m_overlap_k,
					numThreads.z + this->m_overlap_i * 2,
					numThreads.y + this->m_overlap_j * 2,
					numThreads.x + this->m_overlap_k * 2,
					this->m_edge, this->m_pad
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<OI>(out_mem_p[i])->changeDeviceData(), 0)...);
			}*/
			SKEPU_ERROR("Multi-device CUDA MapOverlap 3D not implemented");
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CUDA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap 3D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CU(0, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapMultipleThread_CU(numDevices, p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}

    } // backend

} // skepu

#endif // SKEPU_CUDA