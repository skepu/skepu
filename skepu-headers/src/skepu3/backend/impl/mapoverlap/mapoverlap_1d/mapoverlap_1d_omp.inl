/*! \file mapoverlap_1d_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapOverlap1D skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{

        /*!
		 *  Performs the MapOverlap on a range of elements using \em OpenMP as backend and a seperate output range.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			const int overlap = this->m_overlap[0];
			const size_t size = arg.size();
			const size_t stride = 1;
			
			auto random_pre = this->template prepareRandom<MapOverlapFunc::randomCount>(overlap);
			size_t threads = std::min<size_t>(size - 2*overlap, omp_get_max_threads());
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(size - 2*overlap, threads);
			auto random_post = this->template prepareRandom<MapOverlapFunc::randomCount>(overlap);
			
			if (this->m_edge != Edge::None)
			{
				T start[3*overlap], end[3*overlap];

#pragma omp parallel for schedule(runtime)
				for (size_t i = 0; i < overlap; ++i)
				{
					switch (this->m_edge)
					{
					case Edge::Cyclic:
						start[i] = arg(size + i  - overlap);
						end[3*overlap-1 - i] = arg(overlap-i-1);
						break;
					case Edge::Duplicate:
						start[i] = arg(0);
						end[3*overlap-1 - i] = arg(size-1);
						break;
					case Edge::Pad:
						start[i] = this->m_pad;
						end[3*overlap-1 - i] = this->m_pad;
						break;
					default:
						break;
					}
				}
				
				for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
					start[i] = arg(j);
				
				for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
					end[i] = arg(j + size - 2*overlap);
				
				for (size_t i = 0; i < overlap; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random_pre, Region1D<T>{overlap, stride, &start[i + overlap]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
				}
					
#pragma omp parallel for schedule(runtime)
				for (size_t i = overlap; i < size - overlap; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random(omp_get_thread_num()), Region1D<T>{overlap, stride, &arg(i)},
						get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
				}
					
				for (size_t i = size - overlap; i < size; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random_post, Region1D<T>{overlap, stride, &end[i + 2 * overlap - size]},
						get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
				}
			}

			else // Edge::None
			{
#pragma omp parallel for schedule(runtime)
				for (size_t i = 0; i < size - overlap * 2; ++i)
				{
					auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random(omp_get_thread_num()), Region1D<T>{overlap, stride, &arg(i + overlap)},
						get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
				}
			}

			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 1D", this, {size},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenMP with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			int overlap = this->m_overlap[0];
			size_t size = arg.size();
			T start[3*overlap], end[3*overlap];
			
			size_t rowWidth = arg.total_cols();
			size_t colWidth = arg.total_rows();
			size_t stride = 1;
			
			const T *inputBegin = arg.getAddress();
			const T *inputEnd = inputBegin + size;
			
			for (size_t row = 0; row < colWidth; ++row)
			{
				auto random_pre = this->template prepareRandom<MapOverlapFunc::randomCount>(overlap);
				size_t threads = std::min<size_t>(rowWidth - 2*overlap, omp_get_max_threads());
				auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(rowWidth - 2*overlap, threads);
				auto random_post = this->template prepareRandom<MapOverlapFunc::randomCount>(overlap);
				
				inputEnd = inputBegin + rowWidth;
				
				if (this->m_edge != Edge::None)
				{
#pragma omp parallel for schedule(runtime)
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
						default:
							break;
						}
					}
					
					for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
						start[i] = inputBegin[j];
					
					for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
						end[i] = inputEnd[j - 2*overlap];
					
					for (size_t i = 0; i < overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random_pre, Region1D<T>{overlap, stride, &start[i + overlap]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
					}
						
#pragma omp parallel for schedule(runtime)
					for (size_t i = overlap; i < rowWidth - overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random(omp_get_thread_num()), Region1D<T>{overlap, stride, &inputBegin[i]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
					}
						
					for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random_post, Region1D<T>{overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
					}
				}

				else // Edge::None
				{
					size_t out_cols = get<0>(std::forward<CallArgs>(args)...).total_cols();
#pragma omp parallel for schedule(runtime)
					for (size_t i = 0; i < out_cols; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random(omp_get_thread_num()), Region1D<T>{overlap, stride, &inputBegin[i + overlap]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
					}
				}
				
				inputBegin += rowWidth;
			}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap RowWise", this, {colWidth, rowWidth},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenMP with a seperate output range.
		 *  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			int overlap = this->m_overlap[0];
			size_t size = arg.size();
			T start[3*overlap], end[3*overlap];
			
			size_t rowWidth = arg.total_cols();
			size_t colWidth = arg.total_rows();
			size_t stride = rowWidth;
			
			const T *inputBegin = arg.getAddress();
			const T *inputEnd = inputBegin + size;

			for (size_t col = 0; col < arg.total_cols(); ++col)
			{
				auto random_pre = this->template prepareRandom<MapOverlapFunc::randomCount>(overlap);
				size_t threads = std::min<size_t>(colWidth - 2*overlap, omp_get_max_threads());
				auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(colWidth - 2*overlap, threads);
				auto random_post = this->template prepareRandom<MapOverlapFunc::randomCount>(overlap);
				
				inputEnd = inputBegin + (rowWidth * (colWidth-1));
				
				if (this->m_edge != Edge::None)
				{
#pragma omp parallel for schedule(runtime)
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
						default:
							break;
						}
					}
					
					for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
						start[i] = inputBegin[j*stride];
					
					for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
						end[i] = inputEnd[(j - 2*overlap + 1)*stride];
					
					for (size_t i = 0; i < overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random_pre, Region1D<T>{overlap, 1, &start[i + overlap]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
					}
						
#pragma omp parallel for schedule(runtime)
					for (size_t i = overlap; i < colWidth - overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random(omp_get_thread_num()), Region1D<T>{overlap, stride, &inputBegin[i*stride]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
					}
						
					for (size_t i = colWidth - overlap; i < colWidth; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random_post, Region1D<T>{overlap, 1, &end[i + 2 * overlap - colWidth]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
					}
				}

				else // Edge::None
				{
					size_t out_rows = get<0>(std::forward<CallArgs>(args)...).total_rows();
#pragma omp parallel for schedule(runtime)
					for (size_t i = 0; i < out_rows; ++i)
					{
						auto res = F::forward(MapOverlapFunc::OMP, Index1D{i}, random(omp_get_thread_num()), Region1D<T>{overlap, stride, &inputBegin[(i + overlap)*stride]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
					}
				}
				
				inputBegin += 1;
			}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap ColWise", this, {colWidth, rowWidth},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}

    } // backend

} // skepu

#endif // SKEPU_OPENMP