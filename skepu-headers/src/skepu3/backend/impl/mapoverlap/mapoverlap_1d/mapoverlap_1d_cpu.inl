/*! \file mapoverlap_1d_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapOverlap1D skeleton.
 */

namespace skepu
{
	namespace backend
	{

        template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
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
			
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(size); // TODO: fix order and edge mode
			
			if (this->m_edge != Edge::None)
			{
				T start[3*overlap], end[3*overlap];
				
				for (size_t i = 0; i < overlap; ++i)
				{
					switch (this->m_edge)
					{
					case Edge::Cyclic:
						start[i] = arg[size + i  - overlap];
						end[3*overlap-1 - i] = arg[overlap-i-1];
						break;
					case Edge::Duplicate:
						start[i] = arg[0];
						end[3*overlap-1 - i] = arg[size-1];
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
					auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, 1, &start[i + overlap]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
				}
				
				for (size_t i = overlap; i < size - overlap; ++i)
				{
					auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, 1, arg.getAddress() + i}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
				}
				
				for (size_t i = size - overlap; i < size; ++i)
				{
					auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, 1, &end[i + 2 * overlap - size]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
				}
			}

			else
			{
				for (size_t i = 0; i < size - overlap * 2; ++i)
					{
						if (p == Parity::None || index_parity(p, i))
						{
							const size_t in_index = i + overlap;
							auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, 1, arg.getAddress() + in_index}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
						}
					}
			}

			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 1D", this, {size},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
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
			T start[3*overlap], end[3*overlap];
			
			const size_t rowWidth = arg.total_cols();
			const size_t colWidth = arg.total_rows();
			const size_t stride = rowWidth;
			
			const T *inputBegin = arg.getAddress();
			const T *inputEnd = inputBegin + size;
			
			size_t final_size = arg.total_cols() * colWidth;
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(colWidth);
			
			for (size_t col = 0; col < arg.total_cols(); ++col)
			{
				inputEnd = inputBegin + rowWidth * (colWidth - 1);
				
				if (this->m_edge != Edge::None)
				{
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
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, 1, &start[i + overlap]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
					}
						
					for (size_t i = overlap; i < colWidth - overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, stride, &inputBegin[i*stride]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
					}
						
					for (size_t i = colWidth - overlap; i < colWidth; ++i)
					{
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, 1, &end[i + 2 * overlap - colWidth]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
					}
				}

				else // Edge::None
				{
					size_t out_rows = get<0>(std::forward<CallArgs>(args)...).total_rows();
					for (size_t i = 0; i < out_rows; ++i)
					{
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, stride, &inputBegin[(i + overlap)*stride]},
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
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
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
			
			size_t final_size = arg.total_rows() * rowWidth;
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(final_size);
			
			for (size_t row = 0; row < arg.total_rows(); ++row)
			{
				inputEnd = inputBegin + rowWidth;
				
				if (this->m_edge != Edge::None)
				{
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
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, stride, &start[i + overlap]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
					}
						
					for (size_t i = overlap; i < rowWidth - overlap; ++i)
					{
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, stride, &inputBegin[i]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
					}
						
					for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
					{
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, stride, &end[i + 2 * overlap - rowWidth]},
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
					}
				}

				else // Edge::none
				{
					size_t out_cols = get<0>(std::forward<CallArgs>(args)...).total_cols();
					for (size_t i = 0; i < out_cols; ++i)
					{
						auto res = F::forward(MapOverlapFunc::CPU, Index1D{i}, random, Region1D<T>{overlap, stride, &inputBegin[i + overlap]},
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

    } // backend

} // skepu