/*! \file map_hy.inl
 *  \brief Contains the definitions of Hybrid execution specific member functions for the Map skeleton.
 * 
 * The data is divided between the CPU and the GPU(s) as follows:
 * 
 *  i =  0           ...                 size
 *       ####CPU#####   ########GPU########
 * 
 * The GPU part might in turn split its part of the work between multiple GPUs.
 */

#ifdef SKEPU_HYBRID

#include <omp.h>
#include <iostream>
#include <functional>

namespace skepu
{
	namespace backend
	{
	
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::Hybrid(size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			static constexpr auto proxy_tags = typename MapFunc::ProxyTags{};

			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize * size;
			const size_t gpuSize = size - cpuSize;
			size_t nthr = this->m_selected_spec->CPUThreads();

			DEBUG_TEXT_LEVEL1("Hybrid Map: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if (gpuSize < 32) // Not smaller than a warp (=32 threads)
			{
				DEBUG_TEXT_LEVEL1("Hybrid Map: Too small GPU size, fall back to CPU-only.");
				this->OMP(size, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
				return;
			}
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			
			nthr = std::max<size_t>(2, std::min<size_t>(nthr, cpuSize+1));
			omp_set_num_threads(nthr);
			const size_t numCPUThreads = nthr - 1; // One thread is used for GPU
			
			auto random = this->template prepareRandom<MapFunc::randomCount>(cpuSize, numCPUThreads);
			
			size_t counter = 0;
			std::vector<size_t> start_idxs(numCPUThreads);
			for (size_t t = 0; t < numCPUThreads; ++t)
			{
				start_idxs[t] = counter;
				counter += cpuSize / numCPUThreads + ((t < cpuSize % numCPUThreads) ? 1 : 0);
			}
			
#pragma omp parallel
			{
				size_t myId = omp_get_thread_num();
				
				if (myId == 0) // Let first thread take care of GPU
				{
#ifdef SKEPU_HYBRID_USE_CUDA
					this->CUDA(cpuSize, gpuSize, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#else
					this->CL(cpuSize, gpuSize, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
#endif
				}
				else
				{
					myId--; // Reindex CPU threads 0...numCPUThreads
					
					size_t workSize = cpuSize / numCPUThreads + ((myId < cpuSize % numCPUThreads) ? 1 : 0);
					const size_t first = start_idxs[myId];
					const size_t last = first + workSize;
					
					for (size_t i = first; i < last; ++i)
					{
						auto index = (std::get<0>(std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).begin()...)) + i).getIndex();
						auto res = F::forward(MapFunc::OMP, index, random(myId),
							get<EI>(std::forward<CallArgs>(args)...)(i)..., 
							get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity-outArity>(proxy_tags), index)...,
							get<CI>(std::forward<CallArgs>(args)...)...
						);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
					}
				}
			}
		}
		
	} // namespace backend
} // namespace skepu

#endif // SKEPU_HYBRID
