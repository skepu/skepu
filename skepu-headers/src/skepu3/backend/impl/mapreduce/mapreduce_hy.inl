/*! \file mapreduce_hy.inl
*  \brief Contains the definitions of Hybrid execution specific member functions for the MapReduce skeleton.
*/

#ifdef SKEPU_HYBRID

#include <omp.h>
#include <iostream>
#include <vector>

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		scalar_wrapper_t<typename MapFunc::Ret> MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::Hybrid(size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize * size;
			const size_t gpuSize = size - cpuSize;
			size_t nthr = this->m_selected_spec->CPUThreads();

			DEBUG_TEXT_LEVEL1("Hybrid MapReduce: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");

			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if (gpuSize == 0)
			{
				DEBUG_TEXT_LEVEL1("Hybrid MapReduce: Too small GPU size, fall back to CPU-only.");
				return this->OMP(size, oi, ei, ai, ci, res, std::forward<CallArgs>(args)...);
			}
			else if (cpuSize < 2)
			{
				DEBUG_TEXT_LEVEL1("Hybrid MapReduce: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				return this->CUDA(0, size, oi, ei, ai, ci, res, std::forward<CallArgs>(args)...);
#else
				return this->CL(0, size, oi, ei, ai, ci, res, std::forward<CallArgs>(args)...);
#endif
			}

			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);

			nthr = std::max<size_t>(2, std::min<size_t>(nthr, cpuSize+1));
			omp_set_num_threads(nthr);
			const size_t numCPUThreads = nthr - 1; // One thread is used for GPU

			const size_t q = cpuSize / numCPUThreads;
			const size_t rest = cpuSize % numCPUThreads;
			auto random = this->template prepareRandom<MapFunc::randomCount>(cpuSize, numCPUThreads);

			std::vector<Ret> parsums(numCPUThreads);
			std::vector<size_t> start_idxs(numCPUThreads);
			size_t counter = 0;
			for (size_t t = 0; t < numCPUThreads; ++t)
			{
				start_idxs[t] = counter;
				counter += q + ((t < rest) ? 1 : 0);
			}

			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel
			{
				size_t myId = omp_get_thread_num();

				if (myId == 0) // Let first thread take care of GPU part.
				{
#ifdef SKEPU_HYBRID_USE_CUDA
					res = this->CUDA(cpuSize, gpuSize, oi, ei, ai, ci, res, std::forward<CallArgs>(args)...);
#else
					res = this->CL(cpuSize, gpuSize, oi, ei, ai, ci, res, std::forward<CallArgs>(args)...);
#endif
				}
				else
				{
					myId--; // Reindex CPU threads 0...numCPUThreads-1

					size_t workSize = q + ((myId < rest) ? 1 : 0);
					const size_t first = start_idxs[myId];
					const size_t last = first + workSize;
					
					TempIndexType index;
					if constexpr (sizeof...(EI) > 0)
						index = (get<0>(std::forward<CallArgs>(args)...) + first).getIndex();
					else
						index = make_index(defaultDim{}, first, this->default_size_j, this->default_size_k, this->default_size_l);

					Ret psum = F::forward(MapFunc::OMP,
						index, random(myId),
						get<EI>(std::forward<CallArgs>(args)...)(first)...,
						get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);

					for (size_t i = first+1; i < last; ++i)
					{
						if constexpr (sizeof...(EI) > 0)
							index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
						else
							index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
						
						Ret tempMap = F::forward(MapFunc::OMP,
							index, random(myId),
							get<EI>(std::forward<CallArgs>(args)...)(i)...,
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
							get<CI>(std::forward<CallArgs>(args)...)...
						);
						pack_expand((get_or_return<OI>(psum) = ReduceFunc::OMP(get_or_return<OI>(psum), get_or_return<OI>(tempMap)), 0)...);
					}
					parsums[myId] = psum;
				}
			}

			// Final Reduce sequentially
			for (Ret const& parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);

			return res;
		}

	} // namespace backend
} // namespace skepu

#endif
