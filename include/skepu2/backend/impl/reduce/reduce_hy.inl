/*! \file reduce_omp.inl
*  \brief Contains the definitions of Hybrid execution specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_HYBRID

#include <omp.h>

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Performs the Reduction on a whole Matrix. Returns a \em SkePU vector of reduction result.
		 *  Using \em Hybrid execution backend.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		void Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::Hybrid(Vector<T> &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			// Partition the workload
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuRows = cpuPartitionSize*rows;
			const size_t gpuRows = rows-cpuRows;
			
			DEBUG_TEXT_LEVEL1("Hybrid Reduce (Matrix 1D): rows = " << rows << ", cols = " << cols << " CPU partition: " << (100.0f*cpuPartitionSize) << "% \n");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuRows == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid Reduce: Too small GPU size, fall back to CPU-only.");
				this->OMP(res, arg);
				return;
			}
			else if(cpuRows == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid Reduce: Too small CPU size, fall back to GPU-only.");
				VectorIterator<T> it = res.begin();
#ifdef SKEPU_HYBRID_USE_CUDA
				this->CU(it, arg.begin(), rows);
#else
				this->CL(it, arg.begin(), rows);
#endif
				return;
			}
			
			// Make sure we are properly synched with device data
			arg.updateHost();
			
			Vector<T> result(rows);
			T *data = arg.getAddress();
			
			// Set up thread indexing
			const size_t minThreads = 2; // At least one thread for CPU part and one for GPU part
			const size_t maxThreads = cpuRows + 1; // Max one per thread per row, plus one thread for the GPU
			omp_set_num_threads(std::max(minThreads, std::min(maxThreads, this->m_selected_spec->CPUThreads())));
			const size_t numCPUThreads = omp_get_max_threads() - 1;
			
			// Now, we safely assume that there are at least as number of rows to process as #threads available
			// schedule rows to each thread
			const size_t rowsPerThread = cpuRows / numCPUThreads;
			const size_t restRows = cpuRows % numCPUThreads;
			
#pragma omp parallel
			{
				const size_t myId = omp_get_thread_num();
				const size_t lastThreadId = omp_get_num_threads() - 1;
				
				if(myId == lastThreadId) {
					// Last thread takes care of the GPU part
					VectorIterator<T> it = res.begin() + cpuRows;
#ifdef SKEPU_HYBRID_USE_CUDA
					this->CU(it, arg.begin(cpuRows), gpuRows);
#else
					this->CL(it, arg.begin(cpuRows), gpuRows);
#endif
				} 
				else {
					// CPU threads
					// we divide the "N" remainder rows to first "N" threads instead of giving it to last thread to achieve better load balancing
					
					size_t firstRow = myId * rowsPerThread;
					size_t lastRow;
					
					if(myId != 0)
						firstRow += (myId < restRows) ? myId : restRows;
					
					if(myId < restRows)
						lastRow = firstRow+rowsPerThread + 1;
					else
						lastRow = firstRow+rowsPerThread;
					
					for (size_t r = firstRow; r < lastRow; ++r) {
						size_t base = r*cols;
						T psum = data[base];
						for(size_t c = 1; c < cols; ++c)
						{
							psum = ReduceFunc::OMP(psum, data[base+c]);
						}
						res(r) = psum;
					}
					
				} // END CPU part
			} // END pragma omp parallel
			
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements. Returns a scalar result. Divides the elements between the 
		 *  CPU and a GPU backend and performs reduction of the parts in parallel. The results from each processing unit 
		 *  are then reduced on the CPU.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::Hybrid(size_t size, T &res, Iterator arg)
		{
			// Partition the workload
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize*size;
			const size_t gpuSize = size-cpuSize;
			
			DEBUG_TEXT_LEVEL1("Hybrid Reduce (Vector): size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "% \n");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuSize == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid Reduce: Too small GPU size, fall back to CPU-only.");
				return this->OMP(size, res, arg);
			}
			else if(cpuSize < 2) {
				DEBUG_TEXT_LEVEL1("Hybrid Reduce: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				return this->CU(size, res, arg);
#else
				return this->CL(size, res, arg);
#endif
			}
			
			// Set up thread indexing
			const size_t minThreads = 2; // At least 2 threads, one for CPU one for GPU
			const size_t maxThreads = cpuSize/2 + 1; // Max cpuSize/2 threads for CPU part plus one thread for taking care of GPU
			omp_set_num_threads(std::max(minThreads, std::min(maxThreads, this->m_selected_spec->CPUThreads())));
			const size_t numCPUThreads = omp_get_max_threads() - 1; // One thread is used for GPU
			const size_t q = cpuSize / numCPUThreads;
			const size_t rest = cpuSize % numCPUThreads;
			
			// Make sure we are properly synched with device data
			arg.getParent().updateHost();
			
			std::vector<T> parsums(numCPUThreads+1);
			
#pragma omp parallel
			{
				size_t myId = omp_get_thread_num();
				
				if(myId == 0) {
					// Let first thread handle the GPU part
#ifdef SKEPU_HYBRID_USE_CUDA
					parsums[numCPUThreads] = this->CU(gpuSize, arg[cpuSize], arg+cpuSize);
#else
					parsums[numCPUThreads] = this->CL(gpuSize, arg[cpuSize], arg+cpuSize);
#endif
				}
				else {
					// CPU threads
					myId--; // Reindex CPU threads, 0...numCPUThreads-1
					const size_t first = myId * q;
					const size_t last = (myId+1) * q + ((myId == numCPUThreads-1) ? rest : 0);
					
					T psum = arg(first);
					for (size_t i = first+1; i < last; ++i)
						psum = ReduceFunc::OMP(psum, arg(i));
					parsums[myId] = psum;
					
				}
			}
			
			for (auto it = parsums.begin(); it != parsums.end(); ++it)
				res = ReduceFunc::OMP(res, *it);
			
			return res;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on a
		 *  input Matrix. Returns a scalar result.
		 *  Using the \em Hybrid execution backend.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::Hybrid(T &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			// Partition the workload
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuRows = cpuPartitionSize*rows;
			const size_t gpuRows = rows-cpuRows;
			
			DEBUG_TEXT_LEVEL1("Hybrid Reduce (Matrix 2D): rows = " << rows << ", cols = " << cols << " CPU partition: " << (100.0f*cpuPartitionSize) << "% \n");

			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuRows == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid Reduce: Too small GPU size, fall back to CPU-only.");
				return this->OMP(res, arg);
			}
			else if(cpuRows == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid Reduce: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				return this->CU(res, arg.begin(), rows);
#else
				return this->CL(res, arg.begin(), rows);
#endif
			}
			
			// Make sure we are properly synched with device data
			arg.updateHost();
			T *data = arg.getAddress();
			
			// Set up thread indexing
			const size_t minThreads = 2; // At least one thread for CPU part and one for GPU part
			const size_t maxThreads = cpuRows + 1; // Max one per thread per row, plus one thread for the GPU
			omp_set_num_threads(std::max(minThreads, std::min(maxThreads, this->m_selected_spec->CPUThreads())));
			const size_t numCPUThreads = omp_get_max_threads() - 1;
			
			// Now, we safely assume that there are at least as number of rows to process as #threads available
			// schedule rows to each thread
			const size_t rowsPerThread = cpuRows / numCPUThreads;
			const size_t restRows = cpuRows % numCPUThreads;
			
			std::vector<T> threadRes(numCPUThreads);
			
#pragma omp parallel
			{
				const size_t myId = omp_get_thread_num();
				const size_t lastThreadId = omp_get_num_threads() - 1;
				
				if(myId == lastThreadId) {
					// Last thread takes care of the GPU part
#ifdef SKEPU_HYBRID_USE_CUDA
					res = this->CU(res, arg.begin(cpuRows), gpuRows);
#else
					res = this->CL(res, arg.begin(cpuRows), gpuRows);
#endif
				} 
				else {
					// CPU threads
					// we divide the "N" remainder rows to first "N" threads instead of giving it to last thread to achieve better load balancing
					
					size_t firstRow = myId * rowsPerThread;
					size_t lastRow;
					
					if (myId != 0)
						firstRow += (myId < restRows) ? myId : restRows;
					
					if (myId < restRows)
						lastRow = firstRow+rowsPerThread + 1;
					else
						lastRow = firstRow+rowsPerThread;
					
					// Row wise reduction
					std::vector<T> rowResults(lastRow-firstRow);
					for (size_t r = firstRow; r < lastRow; ++r)
					{
						const size_t base = r*cols;
						T psum = data[base];
						for (size_t c = 1;  c < cols; ++c)
							psum = ReduceFuncRowWise::OMP(psum, data[base+c]);
						rowResults[r - firstRow] = psum;
					}
					
					// Colume wise reduction
					T parred = rowResults[0];
					for(size_t i = 1; i < rowResults.size(); ++i)
						parred = ReduceFuncColWise::OMP(parred, rowResults[i]);
					threadRes[myId] = parred;
					
				}
			} // End omp parallel
			
			
			// Reduce result of each thread, sequentially
			for (auto it = threadRes.begin(); it != threadRes.end(); ++it)
				res = ReduceFuncColWise::OMP(res, *it);
			
			return res;
		}
		
		
		
	} // end namespace backend
} // end namespace skepu2

#endif

