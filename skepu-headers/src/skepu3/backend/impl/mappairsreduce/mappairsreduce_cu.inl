/*! \file mappairsreduce_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapPairsReduce skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>

namespace skepu
{
	namespace backend
	{
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs> 
		void MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CLKernel>
		::mapPairsReduceSingleThreadMultiGPU_CU(size_t numDevices, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			if (this->m_mode == ReduceMode::ColWise)
				std::swap(Vsize, Hsize);
			
			const size_t numRowsPerSlice = Vsize / numDevices;
			const size_t rest = Vsize % numDevices;
			
			auto oArgs = std::forward_as_tuple(get<OI>(std::forward<CallArgs>(args)...)...);
			auto veArgs = std::forward_as_tuple(get<VEI>(std::forward<CallArgs>(args)...)...);
			auto heArgs = std::forward_as_tuple(get<HEI>(std::forward<CallArgs>(args)...)...);
			auto aArgs = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			auto scArgs = std::forward_as_tuple(get<CI>(std::forward<CallArgs>(args)...)...);
			static constexpr auto proxy_tags = typename MapPairsFunc::ProxyTags{};
			
			typename to_device_pointer_cu<decltype(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent()...))>::type    outMemP[MAX_GPU_DEVICES];
			typename to_device_pointer_cu<decltype(std::make_tuple(get<VEI>(std::forward<CallArgs>(args)...).getParent()...))>::type velwiseMemP[MAX_GPU_DEVICES];
			typename to_device_pointer_cu<decltype(std::make_tuple(get<HEI>(std::forward<CallArgs>(args)...).getParent()...))>::type helwiseMemP[MAX_GPU_DEVICES];
			typename to_proxy_cu<typename MapPairsFunc::ProxyTags, decltype(std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent()...))>::type anyMemP[MAX_GPU_DEVICES];
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t baseIndex = startIdx + i * numRowsPerSlice * Hsize;
				outMemP[i] = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU((std::get<OI>(oArgs).begin() + baseIndex).getAddress(), Vsize, i, AccessMode::None)...);
				velwiseMemP[i] = std::make_tuple(std::get<VEI-outArity>(veArgs).getParent().updateDevice_CU((std::get<VEI-outArity>(veArgs) + startIdx).getAddress(), Vsize, i, AccessMode::None)...);
				helwiseMemP[i] = std::make_tuple(std::get<HEI-Varity-outArity>(heArgs).getParent().updateDevice_CU((std::get<HEI-Varity-outArity>(heArgs) + startIdx).getAddress(), Hsize, i, AccessMode::None)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-Varity-Harity-outArity>(aArgs).cudaProxy(i, AccessMode::None, std::get<AI-Varity-Harity-outArity>(proxy_tags), Index1D{0})...);
			}
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), closestPow2(Hsize));
				const size_t numBlocks = std::min(Vsize, this->m_selected_spec->GPUBlocks());
				const size_t baseIndex = startIdx + i * numRowsPerSlice * Hsize;
				const size_t sharedMemSize = sizeof(Ret) * numThreads;
				std::cout << "Shared Mem size: " << sharedMemSize << "\n";
				
				DEBUG_TEXT_LEVEL1("CUDA MapPairsReduce kernel: device " << i << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				outMemP[i]    = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU((std::get<OI>(oArgs).begin() + baseIndex).getAddress(), Vsize, i, AccessMode::Write, true)...);
				velwiseMemP[i] = std::make_tuple(std::get<VEI-outArity>(veArgs).getParent().updateDevice_CU((std::get<VEI-outArity>(veArgs) + startIdx).getAddress(), Vsize, i, AccessMode::Read)...);
				helwiseMemP[i] = std::make_tuple(std::get<HEI-Varity-outArity>(heArgs).getParent().updateDevice_CU((std::get<HEI-Varity-outArity>(heArgs) + startIdx).getAddress(), Hsize, i, AccessMode::Read)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-Varity-Harity-outArity>(aArgs).getParent().cudaProxy(i, MapPairsFunc::anyAccessMode[AI-Varity-Harity-outArity], std::get<AI-Varity-Harity-outArity>(proxy_tags), Index1D{0})...);
				
				size_t threads = std::min<size_t>(1, numBlocks * numThreads); // FIX
				auto random = this->template prepareRandom<MapPairsFunc::randomCount>(1, threads);
				auto randomMemP = random.updateDevice_CU(random.getAddress(), threads, i, AccessMode::ReadWrite);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks, numThreads, sharedMemSize, this->m_environment->m_devices_CU.at(i)->m_streams[i]>>>
#else
				this->m_cuda_kernel<<<numBlocks, numThreads, sharedMemSize>>>
#endif // USE_PINNED_MEMORY
				(
					std::get<OI>(outMemP[i])->getDeviceDataPointer()...,
					randomMemP->getDeviceDataPointer(),
					std::get<VEI-outArity>(velwiseMemP[i])->getDeviceDataPointer()...,
					std::get<HEI-Varity-outArity>(helwiseMemP[i])->getDeviceDataPointer()...,
					std::get<AI-Varity-Harity-outArity>(anyMemP[i]).second...,
					get<CI>(std::forward<CallArgs>(args)...)...,
					Vsize, Hsize,
					baseIndex,
					(this->m_mode == ReduceMode::ColWise)
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-Varity-Harity-outArity>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity-outArity])), 0)...);
				pack_expand((std::get<OI>(outMemP[i])->changeDeviceData(), 0)...);
			}
		}
		
		
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CLKernel>
		::CUDA(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA MapPairsReduce: Vsize = " << Vsize << ", Hsize = " << Hsize << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
			// Same method is invoked no matter how many GPUs we use
			this->mapPairsReduceSingleThreadMultiGPU_CU(numDevices, startIdx, Vsize, Hsize, oi, vei, hei, ai, ci, std::forward<CallArgs>(args)...);
		}
		
	} // end namespace backend
} // end namespace skepu

#endif
