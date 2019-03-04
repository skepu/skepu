/*! \file mapreduce_hy.inl
*  \brief Contains the definitions of Hybrid execution specific member functions for the MapReduce skeleton.
*/

#ifdef SKEPU_HYBRID

#include <omp.h>
#include <iostream>
#include <vector>

namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::Hybrid(size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize*size;
			const size_t gpuSize = size-cpuSize;
			
			DEBUG_TEXT_LEVEL1("Hybrid MapReduce: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuSize == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid MapReduce: Too small GPU size, fall back to CPU-only.");
				return this->OMP(size, ei, ai, ci, res, args...);
			}
			else if(cpuSize < 2) {
				DEBUG_TEXT_LEVEL1("Hybrid MapReduce: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				return this->CUDA(0, size, ei, ai, ci, res, args...);
#else
				return this->CL(0, size, ei, ai, ci, res, args...);
#endif
			}
			
			
			// Set up thread indexing
			const size_t minThreads = 2; // At least 2 threads, one for CPU one for GPU
			const size_t maxThreads = cpuSize/2 + 1; // Max cpuSize/2 threads for CPU part plus one thread for taking care of GPU
			omp_set_num_threads(std::max(minThreads, std::min(this->m_selected_spec->CPUThreads(), maxThreads)));
			
			const size_t numCPUThreads = omp_get_max_threads() - 1;
			const size_t q = cpuSize / numCPUThreads;
			const size_t rest = cpuSize % numCPUThreads;
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			std::vector<Ret> parsums(numCPUThreads);
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel
			{
				size_t myId = omp_get_thread_num();
				
				if(myId == 0) {
					// Let first thread take care of GPU part.
#ifdef SKEPU_HYBRID_USE_CUDA
					res = this->CUDA(cpuSize, gpuSize, ei, ai, ci, res, args...);
#else
					res = this->CL(cpuSize, gpuSize, ei, ai, ci, res, args...);
#endif
				}
				else {
					// CPU threads
					myId--; // Reindex CPU threads 0...numCPUThreads-1
					
					const size_t first = myId * q;
					const size_t last = (myId + 1) * q + (myId == numCPUThreads - 1 ? rest : 0);
					
					Ret psum = F::forward(MapFunc::OMP, (get<0, CallArgs...>(args...) + first).getIndex(), get<EI, CallArgs...>(args...)(first)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
					for (size_t i = first+1; i < last; ++i)
					{
						Temp tempMap = F::forward(MapFunc::OMP, (get<0, CallArgs...>(args...) + i).getIndex(), get<EI, CallArgs...>(args...)(i)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
						psum = ReduceFunc::OMP(psum, tempMap);
					}
					parsums[myId] = psum;
				}
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				res = ReduceFunc::OMP(res, parsum);
			
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::Hybrid(size_t size, pack_indices<> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize*size;
			const size_t gpuSize = size-cpuSize;
			
			DEBUG_TEXT_LEVEL1("Hybrid MapReduce Index1D: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuSize == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid MapReduce: Too small GPU size, fall back to CPU-only.");
				return this->OMP(size, ei, ai, ci, res, args...);
			}
			else if(cpuSize < 2) {
				DEBUG_TEXT_LEVEL1("Hybrid MapReduce: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				return this->CUDA(0, size, ei, ai, ci, res, args...);
#else
				return this->CL(0, size, ei, ai, ci, res, args...);
#endif
			}
			
			
			// Set up thread indexing
			const size_t minThreads = 2; // At least 2 threads, one for CPU one for GPU
			const size_t maxThreads = cpuSize/2 + 1; // Max cpuSize/2 threads for CPU part plus one thread for taking care of GPU
			omp_set_num_threads(std::max(minThreads, std::min(this->m_selected_spec->CPUThreads(), maxThreads)));
			
			const size_t numCPUThreads = omp_get_max_threads() - 1;
			const size_t q = cpuSize / numCPUThreads;
			const size_t rest = cpuSize % numCPUThreads;
			
			// Sync with device data
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			std::vector<Ret> parsums(numCPUThreads);
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel
			{
				const size_t myId = omp_get_thread_num();
				const size_t lastThreadId = omp_get_num_threads() - 1;
				
				if(myId == lastThreadId) {
					// Let last thread take care of GPU part.
#ifdef SKEPU_HYBRID_USE_CUDA
					res = this->CUDA(cpuSize, gpuSize, ei, ai, ci, res, args...);
#else
					res = this->CL(cpuSize, gpuSize, ei, ai, ci, res, args...);
#endif
				}
				else {
					// CPU threads
					const size_t first = myId * q;
					const size_t last = (myId + 1) * q + (myId == numCPUThreads - 1 ? rest : 0);
					
					Ret psum = F::forward(MapFunc::OMP, skepu2::Index1D{first}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
					for (size_t i = first+1; i < last; ++i)
					{
						Temp tempMap = F::forward(MapFunc::OMP, skepu2::Index1D{i}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
						psum = ReduceFunc::OMP(psum, tempMap);
					}
					parsums[myId] = psum;
				}
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				res = ReduceFunc::OMP(res, parsum);
			
			return res;
		}
		
	} // namespace backend
} // namespace skepu2

#endif

