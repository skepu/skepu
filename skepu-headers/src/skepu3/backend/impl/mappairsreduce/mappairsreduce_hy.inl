/*! \file mappairsreduce_hy.inl
*  \brief Contains the definitions of Hybrid specific member functions for the MapPairsReduce skeleton.
*/

#ifdef SKEPU_HYBRID

#include <omp.h>
#include <iostream>
#include <vector>

namespace skepu
{
	namespace backend
	{
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CLKernel>
		::Hybrid(size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			static constexpr auto proxy_tags = typename MapPairsFunc::ProxyTags{};

			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			size_t nthr = this->m_selected_spec->CPUThreads();
			
			DEBUG_TEXT_LEVEL1("Hybrid MapPairsReduce: hsize = " << Hsize << ", vsize = " << Vsize << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// Used with mode RowWise
			const size_t cpuVSize = cpuPartitionSize * Vsize;
			const size_t gpuVSize = Vsize - cpuVSize;
			
			// Used with mode ColWise
			const size_t cpuHSize = cpuPartitionSize * Hsize;
			const size_t gpuHSize = Hsize - cpuHSize;
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if ((this->m_mode == ReduceMode::RowWise && gpuVSize < 32)
			 || (this->m_mode == ReduceMode::ColWise && gpuHSize < 32)) // Not smaller than a warp (=32 threads)
			{
				DEBUG_TEXT_LEVEL1("Hybrid MapPairsReduce: Too small GPU size, fall back to CPU-only.");
				this->OMP(Vsize, Hsize, oi, vei, hei, ai, ci, std::forward<CallArgs>(args)...);
				return;
			}
			
			// Sync with device data
			pack_expand((get<HEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<VEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity-outArity])), 0)...);
			
			nthr = std::max<size_t>(2, std::min<size_t>(nthr, cpuVSize+1));
			omp_set_num_threads(nthr);
			const size_t numCPUThreads = nthr - 1; // One thread is used for GPU
			
			auto random = (this->m_mode == ReduceMode::RowWise)
				? this->template prepareRandom<MapPairsFunc::randomCount>(cpuVSize * Hsize, numCPUThreads, Hsize)
				: this->template prepareRandom<MapPairsFunc::randomCount>(cpuHSize * Vsize, numCPUThreads, Vsize);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			size_t cpuActualSize = (this->m_mode == ReduceMode::RowWise) ? cpuVSize : cpuHSize;
			size_t counter = 0;
			std::vector<size_t> start_idxs(numCPUThreads);
			for (size_t t = 0; t < numCPUThreads; ++t)
			{
				start_idxs[t] = counter;
				counter += cpuActualSize / numCPUThreads + ((t < cpuActualSize % numCPUThreads) ? 1 : 0);
			}
			
#pragma omp parallel
			{
				size_t myId = omp_get_thread_num();
				
				if (myId == 0) // Let first thread take care of GPU
				{
					if (this->m_mode == ReduceMode::RowWise) {
#ifdef SKEPU_HYBRID_USE_CUDA
						this->CUDA
#else
						this->CL
#endif
						(cpuVSize, gpuVSize, Hsize, oi, vei, hei, ai, ci,
							get<OI> (std::forward<CallArgs>(args)...)...,
							get<VEI>(std::forward<CallArgs>(args)...) + cpuVSize...,
							get<HEI>(std::forward<CallArgs>(args)...)...,
							get<AI> (std::forward<CallArgs>(args)...)...,
							get<CI> (std::forward<CallArgs>(args)...)...
						);
					} else { // ColWise
#ifdef SKEPU_HYBRID_USE_CUDA
						this->CUDA
#else
						this->CL
#endif
						(cpuHSize, Vsize, gpuHSize, oi, vei, hei, ai, ci,
							get<OI> (std::forward<CallArgs>(args)...)...,
							get<VEI>(std::forward<CallArgs>(args)...)...,
							get<HEI>(std::forward<CallArgs>(args)...) + cpuHSize...,
							get<AI> (std::forward<CallArgs>(args)...)...,
							get<CI> (std::forward<CallArgs>(args)...)...
						);
					}
				}
				else
				{
					myId--; // Reindex CPU threads 0...numCPUThreads
					
					size_t workSize = cpuActualSize / numCPUThreads + ((myId < cpuActualSize % numCPUThreads) ? 1 : 0);
					const size_t first = start_idxs[myId];
					const size_t last = first + workSize;
					
					if (this->m_mode == ReduceMode::RowWise)
						for (size_t i = first; i < last; ++i)
						{
							pack_expand((get<OI>(std::forward<CallArgs>(args)...)(i) = get_or_return<OI>(this->m_start), 0)...);
							for (size_t j = 0; j < Hsize; ++j)
							{
								auto index = Index2D { i, j };
								auto temp = F::forward(MapPairsFunc::OMP, Index2D{ i, j }, random(omp_get_thread_num()),
									get<VEI>(std::forward<CallArgs>(args)...)(i)...,
									get<HEI>(std::forward<CallArgs>(args)...)(j)...,
									get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
									get<CI>(std::forward<CallArgs>(args)...)...
								);
								pack_expand((get<OI>(std::forward<CallArgs>(args)...)(i) = ReduceFunc::CPU(get<OI>(std::forward<CallArgs>(args)...)(i), get_or_return<OI>(temp)), 0)...);
							}
						}
					else if (this->m_mode == ReduceMode::ColWise)
						for (size_t j = first; j < last; ++j) // TODO: optimize?
						{
							pack_expand((get<OI>(std::forward<CallArgs>(args)...)(j) = get_or_return<OI>(this->m_start), 0)...);
							for (size_t i = 0; i < Vsize; ++i)
							{
								auto index = Index2D { i, j };
								auto temp = F::forward(MapPairsFunc::OMP, Index2D{ i, j }, random(omp_get_thread_num()),
									get<VEI>(std::forward<CallArgs>(args)...)(i)...,
									get<HEI>(std::forward<CallArgs>(args)...)(j)...,
									get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
									get<CI>(std::forward<CallArgs>(args)...)...
								);
								pack_expand((get<OI>(std::forward<CallArgs>(args)...)(j) = ReduceFunc::CPU(get<OI>(std::forward<CallArgs>(args)...)(j), get_or_return<OI>(temp)), 0)...);
							}
						}
					
					
					
				}
			}
			SKEPU_TRACE_CALL(trace_handle, "MapPairsReduce", this, {Hsize, Vsize},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<VEI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...,
					get<HEI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
	} // namespace backend
} // namespace skepu

#endif
