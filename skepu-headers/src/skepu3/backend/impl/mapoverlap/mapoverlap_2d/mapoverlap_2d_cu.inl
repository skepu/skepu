/*! \file mapoverlap_2d_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapOverlap2D skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <skepu3/matrix.hpp>

namespace skepu
{
	namespace backend
	{

        /*!
		 *  Performs the 2D MapOverlap using a single CUDA GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapSingleThread_CU(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			const size_t in_rows = arg.total_rows();
			const size_t in_cols = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			
			cudaSetDevice(deviceID);
			
			auto in_mem_p = arg.updateDevice_CU(arg.getAddress(), in_rows * in_cols, deviceID, AccessMode::Read, true);
			auto out_mem_p = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CU(get<OI>(std::forward<CallArgs>(args)...).getAddress(), out_rows * out_cols, deviceID, AccessMode::Write, true)...);
			
			dim3 numBlocks, numThreads;
			
			numThreads.x = (out_cols > 16) ? 16 : out_cols;
			numThreads.y = (out_rows > 32) ? 32 : out_rows;
			numThreads.z = 1;
			
			numBlocks.x = (out_cols + numThreads.x - 1) / numThreads.x;
			numBlocks.y = (out_rows + numThreads.y - 1) / numThreads.y;
			numBlocks.z = 1;
			
			auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).cudaProxy(deviceID, MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])...);
		
			// PRNG support
			size_t prng_threads = std::min<size_t>(out_rows * out_cols, numBlocks.x * numBlocks.y * numThreads.x * numThreads.y);
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(out_rows * out_cols, prng_threads);
			auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, deviceID, AccessMode::ReadWrite);
			
			size_t sharedMem =  (numThreads.x + this->m_overlap[1] * 2) * (numThreads.y + this->m_overlap[0] * 2) * sizeof(T);
			
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap kernel: size = " << out_rows * out_cols << ", one device, numBlocks = [" << numBlocks.x << "x" << numBlocks.y << "], numThreads = [" << numThreads.x << "x" << numThreads.y << "]");
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks,numThreads, sharedMem, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_kernel<<<numBlocks,numThreads, sharedMem>>>
#endif
			(
				std::get<OI>(out_mem_p)->getDeviceDataPointer()...,
				randomMemP->getDeviceDataPointer(),
				in_mem_p->getDeviceDataPointer(),
				std::get<AI-InArity-OutArity>(anyMemP).second...,
				get<CI>(std::forward<CallArgs>(args)...)...,
				in_rows, in_cols,
				out_rows, out_cols,
				this->m_overlap[0], this->m_overlap[1],
				in_cols, out_cols,
				numThreads.y + this->m_overlap[0] * 2,
				numThreads.x + this->m_overlap[1] * 2,
				this->m_edge, this->m_pad
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<OI>(out_mem_p)->changeDeviceData(), 0)...);
		}
		
		
		/*!
		*  Performs the 2D MapOverlap using multiple CUDA GPUs.
		*  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapMultipleThread_CU(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			const size_t in_rows = arg.total_rows();
			const size_t in_cols = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			
			const size_t numRowsPerSlice = out_rows / numDevices;
			const size_t restRows = out_rows % numDevices;
			
			// Need to get new values from other devices so that the overlap between devices is up to date.
			// Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
		//	input.updateHostAndInvalidateDevice();
			
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
				
				size_t inRows = outRows + this->m_overlap[1] * 2; // no matter which device, number of input rows is same.
				
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
				
				size_t inRows = outRows + this->m_overlap[1] * 2; // no matter which device, number of input rows is same.
				
				in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numRowsPerSlice * in_cols, inRows * in_cols, i, AccessMode::Read, true);
				out_mem_p[i] = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).updateDevice_CU(get<OI>(std::forward<CallArgs>(args)...).getAddress() + i * numRowsPerSlice * out_cols, outRows * out_cols, i, AccessMode::Write, true, true)...);
				
				dim3 numBlocks;
				dim3 numThreads;
				
				numThreads.x = (out_cols > 16) ? 16 : out_cols;
				numThreads.y = (outRows > 32) ? 32 : outRows;
				numThreads.z = 1;
				
				numBlocks.x = (out_cols + numThreads.x - 1) / numThreads.x;
				numBlocks.y = (outRows + numThreads.y - 1) / numThreads.y;
				numBlocks.z = 1;
				
				size_t sharedMem =  (numThreads.x + this->m_overlap[0] * 2) * (numThreads.y + this->m_overlap[1] * 2) * sizeof(T);
				
				auto anyMemP = std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).cudaProxy(i, MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])...);
		
				// PRNG support
				size_t prng_threads = std::min<size_t>(out_rows * out_cols, numBlocks.x * numBlocks.y * numThreads.x * numThreads.y);
				auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(out_rows * out_cols, prng_threads);
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
					std::get<AI-InArity-OutArity>(anyMemP).second...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					in_rows, in_cols,
					outRows, out_cols,
					this->m_overlap[1], this->m_overlap[0],
					in_cols, out_cols,
					numThreads.y + this->m_overlap[1] * 2,
					numThreads.x + this->m_overlap[0] * 2,
					this->m_edge, this->m_pad
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<OI>(out_mem_p[i])->changeDeviceData(), 0)...);
			}
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CUDA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			auto &res = get<0>(std::forward<CallArgs>(args)...);
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap 2D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
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