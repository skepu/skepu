/*! \file reduce_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		/*!
		 *  Performs the Reduction on a whole Matrix. Returns a \em SkePU vector of reduction result.
		 *  Using \em OpenMP as backend.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		void Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::OMP(Vector<T> &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			
			DEBUG_TEXT_LEVEL1("OpenMP Reduce (Matrix 1D): rows = " << rows << ", cols = " << cols << "\n");
			
			// Make sure we are properly synched with device data
			arg.updateHost();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			T *data = arg.getAddress();
			
#pragma omp parallel for schedule(runtime)
			for (size_t row = 0; row < rows; ++row)
			{
				size_t base = row * cols;
				T parsum = data[base];
				for (size_t col = 1; col < cols; ++col)
					parsum = ReduceFunc::OMP(parsum, data[base + col]);
				res(row) = parsum;
			}
			SKEPU_TRACE_CALL(trace_handle, "Reduce 1D-Partial", this, {rows, cols}, {res.getParent().getObjectID()}, {arg.getParent().getObjectID()}, {}, {});
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements. Returns a scalar result. Divides the elements among all
		 *  \em OpenMP threads and does reduction of the parts in parallel. The results from each thread are then
		 *  reduced on the CPU.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		Scalar<typename ReduceFunc::Ret> Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::OMP(size_t size, T &res, Iterator arg)
		{
			DEBUG_TEXT_LEVEL1("OpenMP Reduce (Vector): size = " << size << "\n");
			
			// Make sure we are properly synched with device data
			arg.getParent().updateHost();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			std::vector<T> parsums(std::min<size_t>(size, omp_get_max_threads()));
			bool first = true;
			
#pragma omp parallel for schedule(runtime) firstprivate(first)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				if (first)
				{
					parsums[myid] = arg(i);
					first = false;
				}
				else
					parsums[myid] = ReduceFunc::OMP(parsums[myid], arg(i));
			}
			
			for (auto const& el : parsums)
				res = ReduceFunc::OMP(res, el);

			auto result = Scalar<T>(res);
			SKEPU_TRACE_CALL(trace_handle, "Reduce 1D-Full", this, {size}, tracing::scalar_labels(result), {arg.getParent().getObjectID()}, {}, {});
			return result;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on a
		 *  input Matrix. Returns a scalar result.
		 *  Using the \em OpenMP as backend.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::OMP(T &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			DEBUG_TEXT_LEVEL1("OpenMP Reduce (Matrix 2D): rows = " << rows << ", cols = " << cols << "\n");
		
			// Make sure we are properly synched with device data
			arg.updateHost();
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			T *data = arg.getAddress();
			std::vector<T> rowsums(rows);
			
			// First row-wise
#pragma omp parallel for schedule(runtime)
			for (size_t row = 0; row < rows; ++row)
			{
				size_t base = row * cols;
				T rowsum = data[base];
				for (size_t col = 1; col < cols; ++col)
					rowsum = ReduceFuncRowWise::OMP(rowsum, data[base + col]);
				rowsums[row] = rowsum;
			}
			
			// Then partial col-wise
			std::vector<T> parsums(std::min<size_t>(rows, omp_get_max_threads()));
			bool first = true;
#pragma omp parallel for schedule(runtime) firstprivate(first)
			for (size_t i = 0; i < rows; ++i)
			{
				size_t myid = omp_get_thread_num();
				if (first)
				{
					parsums[myid] = rowsums[i];
					first = false;
				}
				else
					parsums[myid] = ReduceFuncColWise::OMP(parsums[myid], rowsums[i]);
			}
			
			// Final col-wise sequential reduction
			for (auto const& el : parsums)
				res = ReduceFuncColWise::OMP(res, el);
				
			auto result = Scalar<T>(res);
			SKEPU_TRACE_CALL(trace_handle, "Reduce 2D", this, {rows, cols}, tracing::scalar_labels(result), {arg.getParent().getObjectID()}, {}, {});
			return result;
		}
		
		
	} // end namespace backend
} // end namespace skepu

#endif
