/*! \file scan_hy.inl
 *  \brief Contains the definitions of Hybrid execution specific member functions for the Scan skeleton.
 * 
 * This implementation will split the Scan into several blocks, one for each CPU thread and one for the GPU, like:
 * 
 *        CPU threads:       |          GPU
 *   #1 | #2 | #3 | #4 | #5  | 
 *  ....|....|....|....|.... | ........................
 *   ...| ...| ...| ...| ... |  .......................
 *    ..|  ..|  ..|  ..|  .. |   ......................
 *     .|   .|   .|   .|   . |    .....................
 *     a|   b|   c|   d|   e |     ....................
 *                           |      ...................
 *  ---- OpenMP Barrier -----|       ..................
 *                           |        .................
 *      Master thread:       |         ................
 *         abcde             |          ...............
 *          ....             |           ..............
 *           ...             |            .............
 *            ..             |             ............
 *             .             |              ...........
 *         ABCDE             |               ..........
 *                           |                .........
 *  ---- OpenMP Barrier -----|                 ........
 *                           |                  .......
 *      CPU threads:         |                   ......
 *  (#1)| #2 | #3 | #4 | #5  |                    .....
 *      | +A | +B | +C | +D  |                     ....
 *      | .. | .. | .. | ..  |                      ...
 *      | .. | .. | .. | ..  |                       ..
 *      | .. | .. | .. | ..  |                        .
 *      | .. | .. | .. | ..  |                        f
 *      | +A | +B | +C | +D  |          +E
 * 
 * All threads will first preform a local Scan on their block, for the CPU threads resulting in the block sums a-e (i.e. where c is the sum[1] 
 * of all elements of thread #3's block and so on) and on the GPU to the block sum f. The master thread will then do a Scan of these total 
 * sums of the CPU blocks. This will result in an array with 5 elements, A-E. These five elements are the missing sums for the global Scan 
 * of blocks corresponding to threads #2-#5 and the GPU. These are then added to the blocks (Thread #1 is already done, as it has no preceeding 
 * block, and is thus not missing any values for the global Scan). The add part will take much longer time on CPU (denoted +A to +D) than on GPU 
 * (denoted +E), thus is the GPU not included in the \em OpenMP \em barriers when the array of missing block values are calculated. The GPU is 
 * instead synchronized with the CPUs through a lock to make sure it doesn't start its add pass before the missing values are properly calculated.
 * 
 * [1] Sum is used here as if the Scan was using an add user function, i.e. implementing a prefix sum.
 * 
 */

#ifdef SKEPU_HYBRID

namespace skepu
{
	namespace backend
	{
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::Hybrid(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			// Make sure we are properly synched with device data
			res.getParent().invalidateDeviceData();
			arg.getParent().updateHost();
			
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize*size;
			const size_t gpuSize = size-cpuSize;
			
			DEBUG_TEXT_LEVEL1("Hybrid Scan: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuSize == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid Scan: Too small GPU size, fall back to CPU-only.");
				this->OMP(size, res, arg, mode, initial);
				return;
			}
			else if(cpuSize < 2) {
				DEBUG_TEXT_LEVEL1("Hybrid Scan: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				this->CU(size, res, arg, mode, initial);
#else
				this->CL(size, res, arg, mode, initial);
#endif
				return;
			}
			
			// Setup parameters needed to parallelize with OpenMP
			const size_t minThreads = 2; // At least 2 threads, one for CPU one for GPU
			const size_t maxThreads = cpuSize/2 + 1; // Max cpuSize/2 threads for CPU part plus one thread for taking care of GPU
			omp_set_num_threads(std::max(minThreads, std::min(this->m_selected_spec->CPUThreads(), maxThreads)));
			const size_t nthr = omp_get_max_threads();
			const size_t numCPUThreads = nthr - 1; // One thread is used for GPU
			
			const size_t q = cpuSize / numCPUThreads;
			const size_t rest = cpuSize % numCPUThreads;
			
			// Array to store partial thread results in.
			std::vector<T> offset_array(numCPUThreads);
			
			// Process first element here
			*res = (mode == ScanMode::Inclusive) ? *arg++ : initial;
			
			omp_set_nested(true);
			
			omp_lock_t gpuAddLock;
			omp_init_lock(&gpuAddLock);
			omp_set_lock(&gpuAddLock);
			
#pragma omp parallel num_threads(2)
			{
				
				if(omp_get_thread_num() == 0) {
					// GPU thread
					size_t inOffset = (mode == ScanMode::Inclusive) ? 1 : 0; // Needed because arg ptr is moved one step when mode == Inclusive
					T gpuStartValue = arg(cpuSize-1); // Only used when mode == Exclusive really
#ifdef SKEPU_HYBRID_USE_CUDA
					this->CU(gpuSize, res+cpuSize, arg+cpuSize-inOffset, mode, gpuStartValue);
#else
					this->CL(gpuSize, res+cpuSize, arg+cpuSize-inOffset, mode, gpuStartValue);
#endif
					
					omp_set_lock(&gpuAddLock); // Make sure CPU threads has finished updating the offset_array
#ifdef SKEPU_HYBRID_USE_CUDA
					const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
					scanAddGPU_CU(numDevices, gpuSize, res+cpuSize, offset_array[numCPUThreads-1]);
#else
					const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
					scanAddGPU_CL(numDevices, gpuSize, res+cpuSize, offset_array[numCPUThreads-1]);
#endif
// 					omp_unset_lock(&gpuAddLock); // Don't bother to unlock
				}
				else {
					// Start CPU threads
#pragma omp parallel num_threads(numCPUThreads)
					{
						const size_t myId = omp_get_thread_num();
						
						const size_t first = myId * q;
						const size_t last = (myId + 1) * q + ((myId == numCPUThreads- 1) ? rest : 0);
						
						
						// First let each thread make their own scan and saved the result in a partial result array.
						if (myId != 0) 
							res(first) = arg(first-1);
						
						for (size_t i = first + 1; i < last; ++i) {
							res(i) = ScanFunc::OMP(res(i-1), arg(i-1));
						}
						offset_array[myId] = res(last-1);
#pragma omp barrier
						
						// Let the master thread scan the partial result array
#pragma omp master
						{
							for (size_t i = 1; i < numCPUThreads; ++i) {
								offset_array[i] = ScanFunc::OMP(offset_array[i-1], offset_array[i]);
							}
							// Signal GPU thread that we are done updating the offset_array
							omp_unset_lock(&gpuAddLock);
						}
						
#pragma omp barrier
						if (myId != 0) {
							// CPU threads, except first one, which is already done.
							// Add the scanned partial results to each threads work batch.
							for (size_t i = first; i < last; ++i) {
								res(i) = ScanFunc::OMP(res(i), offset_array[myId-1]);
							}
						}
					} // END omp parallel(numCPUThreads) block
					
				} // END else
				
			} // END omp parallel(2) block
			
			omp_destroy_lock(&gpuAddLock);
		}

		
#ifdef SKEPU_HYBRID_USE_CUDA
		/**
		 * Utility function to apply the user function with \em value to \em size number of elements in memory pointed to by \em OutIterator. 
		 * Divides the work between \em numDevices GPUs using CUDA.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::scanAddGPU_CU(size_t numDevices, size_t size, OutIterator res, T value) 
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			for (size_t i = 0; i < numDevices; ++i) {
				const size_t numElements = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = std::min(numElements / numThreads + (numElements % numThreads == 0 ? 0:1), this->m_selected_spec->GPUBlocks());
				
				DEBUG_TEXT_LEVEL1("Hybrid-CUDA Scan (add device sums): device " << i << ", numElem = " << numElements << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads << "\n");
				
				typename OutIterator::device_pointer_type_cu out_mem_p = res.getParent().updateDevice_CU((res + i * numElemPerSlice).getAddress(), numElements, i, AccessMode::ReadWrite);
				cudaSetDevice(i);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_scan_add_kernel<<< numBlocks, numThreads, 0, (this->m_environment->m_devices_CU.at(i)->m_streams[0]) >>>
					(out_mem_p->getDeviceDataPointer(), value, numElements);
#else
				this->m_cuda_scan_add_kernel<<< numBlocks, numThreads >>>
					(out_mem_p->getDeviceDataPointer(), value, numElements);
#endif
			}
			cudaSetDevice(m_environment->bestCUDADevID);
		}
#endif // SKEPU_HYBRID_USE_CUDA


#ifndef SKEPU_HYBRID_USE_CUDA // i.e. if OpenCL
		/**
		 * Utility function to apply the user function with \em value to \em size number of elements in memory pointed to by \em OutIterator. 
		 * Divides the work between \em numDevices GPUs using OpenCL.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::scanAddGPU_CL(size_t numDevices, size_t size, OutIterator res, T value) 
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			for (size_t i = 0; i < numDevices; ++i) {
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElements = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = std::min(numElements / numThreads + (numElements % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks());
				
				DEBUG_TEXT_LEVEL1("Hybrid-OpenCL Scan (add device sums): device " << i << ", numElem = " << numElements << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads << "\n");
				
				typename OutIterator::device_pointer_type_cl outMemP = res.getParent().updateDevice_CL((res + i * numElemPerSlice).getAddress(), numElements, device, false);
				
				CLKernel::scanAdd(i, numThreads, numBlocks * numThreads, outMemP, value, numElements);
			}
		}
#endif // ifndef SKEPU_HYBRID_USE_CUDA

		
	}
}

#endif // SKEPU_HYBRID
