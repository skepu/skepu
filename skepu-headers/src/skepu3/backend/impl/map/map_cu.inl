/*! \file map_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Map skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>
#include <functional>

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapSingleThread_CU(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto oArgs = std::forward_as_tuple(get<OI>(std::forward<CallArgs>(args)...)...);
			auto eArgs = std::forward_as_tuple(get<EI>(std::forward<CallArgs>(args)...)...);
			auto aArgs = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			auto scArgs = std::forward_as_tuple(get<CI>(std::forward<CallArgs>(args)...)...);
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};
			
			// Setup parameters
			const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), size);
			const size_t numBlocks = std::max<size_t>(1, std::min( (size / numThreads + (size % numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
			
			DEBUG_TEXT_LEVEL1("CUDA Map: numBlocks = " << numBlocks << ", numThreads = " << numThreads);
			
			// Copies the elements to the device
			auto outMemP = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU(
				std::get<OI>(oArgs).getAddress() + startIdx,
				size * abs(this->m_strides[OI]), deviceID,
				(abs(this->m_strides[OI]) != 1) ? AccessMode::ReadWrite : AccessMode::Write /* upload if not unit stride */)...
			);
			auto elwiseMemP = std::make_tuple(std::get<EI-outArity>(eArgs).getParent().updateDevice_CU(std::get<EI-outArity>(eArgs).getAddress() + startIdx, size * abs(this->m_strides[EI]), deviceID, AccessMode::Read)...);
			auto anyMemP    = std::make_tuple(std::get<AI-arity-outArity>(aArgs).getParent().cudaProxy(deviceID, MapFunc::anyAccessMode[AI-arity-outArity], std::get<AI-arity-outArity>(proxy_tags), Index1D{0})...);
		
			// PRNG support
			size_t prng_threads = std::min<size_t>(size, numBlocks * numThreads);
			auto random = this->template prepareRandom<MapFunc::randomCount>(size, prng_threads);
			auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, deviceID, AccessMode::ReadWrite);
			
			// Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
			(
				std::get<OI>(outMemP)->getDeviceDataPointer()...,
				randomMemP->getDeviceDataPointer(),
				std::get<EI-outArity>(elwiseMemP)->getDeviceDataPointer()...,
				std::get<AI-arity-outArity>(anyMemP).second...,
				std::get<CI-arity-anyArity-outArity>(scArgs)...,
				get<0>(std::forward<CallArgs>(args)...).getParent().size_j(),
				get<0>(std::forward<CallArgs>(args)...).getParent().size_k(),
				get<0>(std::forward<CallArgs>(args)...).getParent().size_l(),
				size, startIdx, this->m_strides
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<OI>(outMemP)->changeDeviceData(), 0)...);
			pack_expand((std::get<AI-arity-outArity>(anyMemP).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			
#ifdef TUNER_MODE
			cudaDeviceSynchronize();
#endif // TUNER_MODE
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapMultiStream_CU(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			CHECK_CUDA_ERROR(cudaSetDevice(deviceID));
			size_t numKernels = std::min<size_t>(this->m_environment->m_devices_CU.at(deviceID)->getNoConcurrentKernels(), size);
			size_t numElemPerSlice = size / numKernels;
			size_t rest = size % numKernels;
			
			auto oArgs = std::forward_as_tuple(get<OI>(std::forward<CallArgs>(args)...)...);
			auto eArgs = std::forward_as_tuple(get<EI>(std::forward<CallArgs>(args)...)...);
			auto aArgs = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			auto scArgs = std::forward_as_tuple(get<CI>(std::forward<CallArgs>(args)...)...);
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};
			
			typename to_device_pointer_cu<decltype(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent()...))>::type    outMemP[numKernels];
			typename to_device_pointer_cu<decltype(std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent()...))>::type elwiseMemP[numKernels];
			typename to_proxy_cu<typename MapFunc::ProxyTags, decltype(std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent()...))>::type             anyMemP[numKernels];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				
				outMemP[i] = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU((std::get<OI>(oArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[OI]), deviceID, AccessMode::None, false, i)...);
				elwiseMemP[i] = std::make_tuple(std::get<EI-outArity>(eArgs).getParent().updateDevice_CU((std::get<EI-outArity>(eArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[EI]), deviceID, AccessMode::None, false, i)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity-outArity>(aArgs).cudaProxy(deviceID, AccessMode::None, false, i, std::get<AI-arity-outArity>(proxy_tags), Index1D{0})...);
			}
			
			// Breadth-first memory transfers and kernel executions
			// First input memory transfer
			for (size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				
				auto outMemP = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU(
					(std::get<OI>(oArgs) + baseIndex).getAddress(),
					numElem * abs(this->m_strides[OI]), deviceID,
					(abs(this->m_strides[OI]) != 1) ? AccessMode::ReadWrite : AccessMode::Write /* upload if not unit stride */, false, i
				)...);
				elwiseMemP[i] = std::make_tuple(std::get<EI-outArity>(eArgs).getParent().updateDevice_CU((std::get<EI-outArity>(eArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[EI]), deviceID, AccessMode::Read,  false, i)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity-outArity>(aArgs).getParent().cudaProxy(deviceID, MapFunc::anyAccessMode[AI-arity-outArity], false, i, std::get<AI-arity-outArity>(proxy_tags), Index1D{0})...);
			}
			
			// Kernel executions
			for(size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
				const size_t numBlocks = std::min(numElem / numThreads + (numElem % numThreads == 0 ? 0:1), this->m_selected_spec->GPUBlocks());
			
				// PRNG support
				size_t prng_threads = std::min<size_t>(size, numBlocks * numThreads);
				auto random = this->template prepareRandom<MapFunc::randomCount>(size, prng_threads);
				auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, deviceID, AccessMode::ReadWrite);
				
				DEBUG_TEXT_LEVEL1("CUDA Map: Kernel " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(deviceID)->m_streams[i]>>>
#else
				this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
				(
					std::get<OI>(outMemP[i])->getDeviceDataPointer()...,
					randomMemP->getDeviceDataPointer(),
					std::get<EI-outArity>(elwiseMemP[i])->getDeviceDataPointer()...,
					std::get<AI-arity-outArity>(anyMemP[i]).second...,
					std::get<CI-arity-anyArity-outArity>(scArgs)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_j(),
					get<0>(std::forward<CallArgs>(args)...).getParent().size_k(),
					get<0>(std::forward<CallArgs>(args)...).getParent().size_l(),
					numElem, baseIndex, this->m_strides
				);
				
				// Change device data
				pack_expand((std::get<OI>(outMemP[i])->changeDeviceData(), 0)...);
				pack_expand((std::get<AI-arity-outArity>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			}

#ifdef TUNER_MODE
			cudaDeviceSynchronize();
#endif // TUNER_MODE
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapMultiStreamMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
#ifdef USE_PINNED_MEMORY
			const size_t numElemPerDevice = size / useNumGPU;
			const size_t deviceRest = size % useNumGPU;
			size_t numKernels[MAX_GPU_DEVICES];
			size_t numElemPerStream[MAX_GPU_DEVICES];
			size_t streamRest[MAX_GPU_DEVICES];
			size_t maxKernels = 0;
			
			auto oArgs = std::forward_as_tuple(get<OI>(std::forward<CallArgs>(args)...)...);
			auto eArgs = std::forward_as_tuple(get<EI>(std::forward<CallArgs>(args)...)...);
			auto aArgs = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			auto scArgs = std::forward_as_tuple(get<CI>(std::forward<CallArgs>(args)...)...);
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};
			
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				numKernels[i] = std::min<size_t>(this->m_environment->m_devices_CU.at(i)->getNoConcurrentKernels(), numElemPerDevice);
				maxKernels = std::max(maxKernels, numKernels[i]);
				
				size_t temp = numElemPerDevice + ((i == useNumGPU-1) ? deviceRest : 0);
				numElemPerStream[i] = temp / numKernels[i];
				streamRest[i] = temp % numKernels[i];
			}
			
			typename to_device_pointer_cu<decltype(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent()...))>::type    outMemP[MAX_GPU_DEVICES][maxKernels];
			typename to_device_pointer_cu<decltype(std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent()...))>::type elwiseMemP[MAX_GPU_DEVICES][maxKernels];
			typename to_proxy_cu<typename MapFunc::ProxyTags, decltype(std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent()...))>::type             anyMemP[MAX_GPU_DEVICES][maxKernels];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					const size_t numElem = numElemPerStream[i] + ((j == numKernels[i]-1) ? streamRest[i] : 0);
					const size_t baseIndex = startIdx + i * numElemPerDevice + j * numElemPerStream[i];
					
					outMemP[i][j] = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU((std::get<OI>(oArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[OI]), i, AccessMode::None, false, j)...);
					elwiseMemP[i][j] = std::make_tuple(std::get<EI-outArity>(eArgs).getParent().updateDevice_CU((std::get<EI-outArity>(eArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[EI]), i, AccessMode::None, false, j)...);
					anyMemP[i][j]    = std::make_tuple(std::get<AI-arity-outArity>(aArgs).cudaProxy(i, AccessMode::None, false, j, std::get<AI-arity-outArity>(proxy_tags), Index1D{0})...);
				}
			}
			
			// First input memory transfer
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					const size_t numElem = numElemPerStream[i] + ((j == numKernels[i]-1) ? streamRest[i] : 0);
					const size_t baseIndex = startIdx + i * numElemPerDevice + j * numElemPerStream[i];
					
					outMemP[i][j] = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU(
						(std::get<OI>(oArgs) + baseIndex).getAddress(),
						numElem * abs(this->m_strides[OI]), i,
						(abs(this->m_strides[OI]) != 1) ? AccessMode::ReadWrite : AccessMode::Write /* upload if not unit stride */, false, j
					)...);
					elwiseMemP[i][j] = std::make_tuple(std::get<EI-outArity>(eArgs).getParent().updateDevice_CU((std::get<EI-outArity>(eArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[EI]), i, AccessMode::Read,  false, j)...);
					anyMemP[i][j]    = std::make_tuple(std::get<AI-arity-outArity>(aArgs).cudaProxy(i, MapFunc::anyAccessMode[AI-arity-outArity], false, j, std::get<AI-arity-outArity>(proxy_tags), Index1D{0})...);
				}
			}
			
			// Kernel executions
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					const size_t numElem = numElemPerStream[i] + ((j == numKernels[i]-1) ? streamRest[i] : 0);
					const size_t baseIndex = i * numElemPerDevice + j * numElemPerStream[i];
					const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
					const size_t numBlocks = std::max<size_t>(1, std::min( (numElem / numThreads + (numElem % numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
					
					// PRNG support
					size_t prng_threads = std::min<size_t>(size, numBlocks * numThreads);
					auto random = this->template prepareRandom<MapFunc::randomCount>(size, prng_threads);
					auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, deviceID, AccessMode::ReadWrite);
					
					DEBUG_TEXT_LEVEL1("CUDA Map: Device " << i << ", kernel = " << j << "numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
					
					this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(i)->m_streams[j]>>>(
						std::get<OI>(outMemP[i][j])->getDeviceDataPointer()...,
						randomMemP->getDeviceDataPointer(),
						std::get<EI-outArity>(elwiseMemP[i][j])->getDeviceDataPointer()...,
						std::get<AI-arity-outArity>(anyMemP[i][j]).second...,
						std::get<CI-arity-anyArity-outArity>(scArgs)...,
						get<0>(std::forward<CallArgs>(args)...).getParent().size_j(),
						get<0>(std::forward<CallArgs>(args)...).getParent().size_k(),
						get<0>(std::forward<CallArgs>(args)...).getParent().size_l(),
						numElem, baseIndex, this->m_strides
					);
					
					pack_expand((std::get<OI>(outMemP[i][j])->changeDeviceData(), 0)...);
					pack_expand((std::get<AI-arity-outArity>(anyMemP[i][j]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
				}
			}
#endif // USE_PINNED_MEMORY
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapSingleThreadMultiGPU_CU(size_t numDevices, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			auto oArgs = std::forward_as_tuple(get<OI>(std::forward<CallArgs>(args)...)...);
			auto eArgs = std::forward_as_tuple(get<EI>(std::forward<CallArgs>(args)...)...);
			auto aArgs = std::forward_as_tuple(get<AI>(std::forward<CallArgs>(args)...)...);
			auto scArgs = std::forward_as_tuple(get<CI>(std::forward<CallArgs>(args)...)...);
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};
			
			typename to_device_pointer_cu<decltype(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).getParent()...))>::type    outMemP[MAX_GPU_DEVICES];
			typename to_device_pointer_cu<decltype(std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).getParent()...))>::type elwiseMemP[MAX_GPU_DEVICES];
			typename to_proxy_cu<typename MapFunc::ProxyTags, decltype(std::make_tuple(get<AI>(std::forward<CallArgs>(args)...).getParent()...))>::type             anyMemP[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				
				outMemP[i] = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU((std::get<OI>(oArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[OI]), i, AccessMode::None)...);
				elwiseMemP[i] = std::make_tuple(std::get<EI-outArity>(eArgs).getParent().updateDevice_CU((std::get<EI-outArity>(eArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[EI]), i, AccessMode::None)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity-outArity>(aArgs).cudaProxy(i, AccessMode::None, std::get<AI-arity-outArity>(proxy_tags), Index1D{0})...);
			}
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = startIdx + i * numElemPerSlice;
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
				const size_t numBlocks = std::max<size_t>(1, std::min( (numElem / numThreads + (numElem % numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
				
				DEBUG_TEXT_LEVEL1("CUDA Map: device " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				outMemP[i] = std::make_tuple(std::get<OI>(oArgs).getParent().updateDevice_CU(
					(std::get<OI>(oArgs) + baseIndex).getAddress(),
					numElem * abs(this->m_strides[OI]), i,
					(abs(this->m_strides[OI]) != 1) ? AccessMode::ReadWrite : AccessMode::Write /* upload if not unit stride */, true
				)...);
				elwiseMemP[i] = std::make_tuple(std::get<EI-outArity>(eArgs).getParent().updateDevice_CU((std::get<EI-outArity>(eArgs) + baseIndex).getAddress(), numElem * abs(this->m_strides[EI]), i, AccessMode::Read)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity-outArity>(aArgs).cudaProxy(i, MapFunc::anyAccessMode[AI-arity-outArity], std::get<AI-arity-outArity>(proxy_tags), Index1D{0})...);
				
				// PRNG support
				size_t prng_threads = std::min<size_t>(size, numBlocks * numThreads);
				auto random = this->template prepareRandom<MapFunc::randomCount>(size, prng_threads);
				auto randomMemP = random.updateDevice_CU(random.getAddress(), prng_threads, i, AccessMode::ReadWrite);
				
				// Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(i)->m_streams[0]>>>
#else
				this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
				(
					std::get<OI>(outMemP[i])->getDeviceDataPointer()...,
					randomMemP->getDeviceDataPointer(),
					std::get<EI-outArity>(elwiseMemP[i])->getDeviceDataPointer()...,
					std::get<AI-arity-outArity>(anyMemP[i]).second...,
					std::get<CI-arity-anyArity-outArity>(scArgs)...,
					get<0>(std::forward<CallArgs>(args)...).getParent().size_j(),
					get<0>(std::forward<CallArgs>(args)...).getParent().size_k(),
					get<0>(std::forward<CallArgs>(args)...).getParent().size_l(),
					numElem, baseIndex, this->m_strides
				);
				
				// Change device data
				pack_expand((std::get<OI>(outMemP[i])->changeDeviceData(), 0)...);
				pack_expand((std::get<AI-arity-outArity>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			}
			
			CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
			
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().setValidFlag(false), 0)...);
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::CUDA(size_t startIdx, size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA Map: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
			{
#ifdef USE_PINNED_MEMORY
				
				// Checks whether or not the GPU supports MemoryTransfer/KernelExec overlapping, if not call mapSingleThread function
				if (this->m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->isOverlapSupported())
					return this->mapMultiStream_CU(this->m_environment->bestCUDADevID, startIdx, size, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
				
#endif // USE_PINNED_MEMORY
				
				return this->mapSingleThread_CU(this->m_environment->bestCUDADevID, startIdx, size, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			}
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
#ifdef USE_PINNED_MEMORY
			
			// if pinned memory is used but the device does not support overlap the function continues with the previous implementation.
			// if the multistream version is being used the function will exit at this point.
			if (this->m_environment->supportsCUDAOverlap())
				return this->mapMultiStreamMultiGPU_CU(numDevices, startIdx, size, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
			
#endif // USE_PINNED_MEMORY
			
			this->mapSingleThreadMultiGPU_CU(numDevices, startIdx, size, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}
	} // namespace backend
} // namespace skepu

#endif
