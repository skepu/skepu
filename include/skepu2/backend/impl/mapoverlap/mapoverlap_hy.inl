/*! \file mapoverlap_hy.inl
*  \brief Contains the definitions of Hybrid execution specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_HYBRID

#include <omp.h>

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Performs the MapOverlap on a range of elements using \em Hybrid backend and a seperate output range.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_Hybrid(Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			const size_t overlap = this->m_overlap;
			const size_t size = arg.size();
			const size_t stride = 1;
			
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuSize = cpuPartitionSize*size;
			const size_t gpuSize = size-cpuSize;
			const size_t nthr = this->m_selected_spec->CPUThreads();
			const size_t numCPUThreads = nthr-1;

			DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: size = " << size << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuSize < 32) { // Not smaller than a warp (=32 threads)
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small GPU size, fall back to CPU-only.");
				this->vector_OpenMP(res, arg, ai, ci, args...);
				return;
			}
			else if(cpuSize < numCPUThreads) {
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				this->vector_CUDA(0, res, arg, ai, ci, args...);
#else
				this->vector_OpenCL(0, res, arg, ai, ci, args...);
#endif
				return;
			}
			
			T start[3*overlap];
			
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			omp_set_nested(true);
			
#pragma omp parallel num_threads(2)
			{
				if(omp_get_thread_num() == 0) {
					// Let first thread handle GPU
#ifdef SKEPU_HYBRID_USE_CUDA
					this->vector_CUDA(cpuSize, res, arg, ai, ci, args...);
#else
					this->vector_OpenCL(cpuSize, res, arg, ai, ci, args...);
#endif
				}
				else {
					// CPU threads
#pragma omp parallel for num_threads(numCPUThreads)
					for (size_t i = 0; i < overlap; ++i)
					{
						switch (this->m_edge)
						{
						case Edge::Cyclic:
							start[i] = arg(size + i  - overlap);
							break;
						case Edge::Duplicate:
							start[i] = arg(0);
							break;
						case Edge::Pad:
							start[i] = this->m_pad;
						}
					}
					
					for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
						start[i] = arg(j);
					
					for (size_t i = 0; i < overlap; ++i)
						res(i) = MapOverlapFunc::OMP(overlap, stride, &start[i + overlap],
								get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
						
#pragma omp parallel for num_threads(numCPUThreads)
					for (size_t i = overlap; i < cpuSize; ++i)
						res(i) = MapOverlapFunc::OMP(overlap, stride, &arg(i),
							get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				} // end else
				
			} // end omp parallel num_threads(2)
			
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em Hybrid backend with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_Hybrid(skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			const size_t rowWidth = arg.total_cols();
			const size_t stride = 1;
			
			const float cpuPartitionSize = this->m_selected_spec->CPUPartitionRatio();
			const size_t cpuRows = cpuPartitionSize*arg.total_rows();
			const size_t gpuRows = arg.total_rows()-cpuRows;
			const size_t nthr = this->m_selected_spec->CPUThreads();
			const size_t numCPUThreads = nthr-1;
			
			DEBUG_TEXT_LEVEL1("Hybrid row-wise MapOverlap: rows = " << arg.total_rows() << " CPU partition: " << (100.0f*cpuPartitionSize) << "%");
			
			// If one partition is considered too small, fall back to GPU-only or CPU-only
			if(gpuRows == 0) {
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small GPU size, fall back to CPU-only.");
				this->rowwise_OpenMP(res, arg, ai, ci, args...);
				return;
			}
			else if(cpuRows < numCPUThreads) {
				DEBUG_TEXT_LEVEL1("Hybrid MapOverlap: Too small CPU size, fall back to GPU-only.");
#ifdef SKEPU_HYBRID_USE_CUDA
				this->rowwise_CUDA(arg.total_rows(), res, arg, ai, ci, args...);
#else
				this->rowwise_OpenCL(arg.total_rows(), res, arg, ai, ci, args...);
#endif
				return;
			}
			
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			size_t overlap = this->m_overlap;
			T start[3*overlap], end[3*overlap];
			
			const Ret *inputBegin = arg.getAddress() + gpuRows*rowWidth;
			const Ret *inputEnd = inputBegin + cpuRows*rowWidth;
			Ret *out = res.getAddress() + gpuRows*rowWidth;
			
			omp_set_nested(true);
			
			// Let GPU take the _first_ gpuRows of the matrix, and the CPU the _last_ cpuRows.
#pragma omp parallel num_threads(2)
			{
				if(omp_get_thread_num() == 1) {
					// Let last thread handle GPU
#ifdef SKEPU_HYBRID_USE_CUDA
					this->rowwise_CUDA(gpuRows, res, arg, ai, ci, args...);
#else
					this->rowwise_OpenCL(gpuRows, res, arg, ai, ci, args...);
#endif
				}
				else {
					// CPU Part
					for (size_t row = gpuRows; row < arg.total_rows(); ++row)
					{
						inputEnd = inputBegin + rowWidth;
						
#pragma omp parallel for num_threads(numCPUThreads)
						for (size_t i = 0; i < overlap; ++i)
						{
							switch (this->m_edge)
							{
							case Edge::Cyclic:
								start[i] = inputEnd[i  - overlap];
								end[3*overlap-1 - i] = inputBegin[overlap-i-1];
								break;
							case Edge::Duplicate:
								start[i] = inputBegin[0];
								end[3*overlap-1 - i] = inputEnd[-1];
								break;
							case Edge::Pad:
								start[i] = this->m_pad;
								end[3*overlap-1 - i] = this->m_pad;
								break;
							}
						}
						
						for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
							start[i] = inputBegin[j];
						
						for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
							end[i] = inputEnd[j - 2*overlap];
						
						for (size_t i = 0; i < overlap; ++i)
							out[i] = MapOverlapFunc::OMP(overlap, stride, &start[i + overlap], get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
							
#pragma omp parallel for num_threads(numCPUThreads)
						for (size_t i = overlap; i < rowWidth - overlap; ++i)
							out[i] = MapOverlapFunc::OMP(overlap, stride, &inputBegin[i], get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
							
						for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
							out[i] = MapOverlapFunc::OMP(overlap, stride, &end[i + 2 * overlap - rowWidth], get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
						
						inputBegin += rowWidth;
						out += rowWidth;
					}
				} // end CPU part
			} // END omp parallel
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em Hybrid backend with a seperate output range.
		 *  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_Hybrid(skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			std::cout << "WARNING: colwise_Hybrid is not implemented for Hybrid exection yet. Will run OpenMP version." << std::endl;
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			size_t overlap = this->m_overlap;
			size_t size = arg.size();
			T start[3*overlap], end[3*overlap];
			
			size_t rowWidth = arg.total_cols();
			size_t colWidth = arg.total_rows();
			size_t stride = rowWidth;
			
			const Ret *inputBegin = arg.getAddress();
			const Ret *inputEnd = inputBegin + size;
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
			for (size_t col = 0; col < arg.total_cols(); ++col)
			{
				inputEnd = inputBegin + (rowWidth * (colWidth-1));
				
#pragma omp parallel for
				for (size_t i = 0; i < overlap; ++i)
				{
					switch (this->m_edge)
					{
					case Edge::Cyclic:
						start[i] = inputEnd[(i+1-overlap)*stride];
						end[3*overlap-1 - i] = inputBegin[(overlap-i-1)*stride];
						break;
					case Edge::Duplicate:
						start[i] = inputBegin[0];
						end[3*overlap-1 - i] = inputEnd[0]; // hmmm...
						break;
					case Edge::Pad:
						start[i] = this->m_pad;
						end[3*overlap-1 - i] = this->m_pad;
						break;
					}
				}
				
				for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
					start[i] = inputBegin[j*stride];
				
				for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
					end[i] = inputEnd[(j - 2*overlap + 1)*stride];
				
				for (size_t i = 0; i < overlap; ++i)
					res(i * stride + col) = MapOverlapFunc::OMP(overlap, 1, &start[i + overlap],
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
#pragma omp parallel for
				for (size_t i = overlap; i < colWidth - overlap; ++i)
					res(i * stride + col) = MapOverlapFunc::OMP(overlap, stride, &inputBegin[i*stride],
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					
				for (size_t i = colWidth - overlap; i < colWidth; ++i)
					res(i * stride + col) = MapOverlapFunc::OMP(overlap, 1, &end[i + 2 * overlap - colWidth],
						get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				
				inputBegin += 1;
			}
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_Hybrid(skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			std::cout << "WARNING: helper_Hybrid is not implemented for Hybrid exection yet. Will run OpenMP version." << std::endl;
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			res.invalidateDeviceData();
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
			const size_t overlap_x = this->m_overlap_x;
			const size_t overlap_y = this->m_overlap_y;
			const size_t rows = res.total_rows();
			const size_t cols = res.total_cols();
			const size_t in_cols = arg.total_cols();
			
#pragma omp parallel for
			for (size_t i = 0; i < rows; i++)
				for (size_t j = 0; j < cols; j++)
					res(i, j) = MapOverlapFunc::CPU(overlap_x, overlap_y, in_cols, &arg((i + overlap_y) * in_cols + (j + overlap_x)), get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
		}
		
	} // namespace backend
} // namespace skepu2

#endif
