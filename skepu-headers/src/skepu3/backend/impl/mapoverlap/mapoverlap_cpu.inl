/*! \file mapoverlap_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapOverlap skeleton.
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
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
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
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
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
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			// Sync with device data
			arg.updateHost();
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
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
				
				inputBegin += rowWidth;
			}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap RowWise", this, {colWidth, rowWidth},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			
			Region2D<T> region{arg, this->m_overlap_y, this->m_overlap_x, this->m_edge, this->m_pad};
			
			Index2D start{0, 0}, end{size_i, size_j};
			if (this->m_edge == Edge::None)
			{
				start = Index2D{(size_t)this->m_overlap_y, (size_t)this->m_overlap_x};
				end = Index2D{size_i - this->m_overlap_y, size_j - this->m_overlap_x};
			}
			
			size_t final_size = (end.row - start.row) * (end.col - start.col);
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(final_size);
			
			for (size_t i = start.row; i < end.row; i++)
				for (size_t j = start.col; j < end.col; j++)
					if (p == Parity::None || index_parity(p, i, j))
					{
						region.idx = Index2D{i,j};
						auto res = F::forward(MapOverlapFunc::CPU, Index2D{i,j}, random, region, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j)..., res);
					}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 2D", this, {size_i, size_j},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI,  typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			
			Region3D<T> region{arg, this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_edge, this->m_pad};
			
			Index3D start{0, 0, 0}, end{size_i, size_j, size_k};
			if (this->m_edge == Edge::None)
			{
				start = Index3D{(size_t)this->m_overlap_i, (size_t)this->m_overlap_j, (size_t)this->m_overlap_k};
				end = Index3D{size_i - this->m_overlap_i, size_j - this->m_overlap_j, size_k - this->m_overlap_k};
			}
			
			size_t final_size = (end.i - start.i) * (end.j - start.j) * (end.k - start.k);
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(final_size);
			
			for (size_t i = start.i; i < end.i; i++)
				for (size_t j = start.j; j < end.j; j++)
					for (size_t k = start.k; k < end.k; k++)
						if (p == Parity::None || index_parity(p, i, j, k))
						{
							region.idx = Index3D{i,j,k};
							auto res = F::forward(MapOverlapFunc::CPU, Index3D{i,j,k}, random, region, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j, k)..., res);
						}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 3D", this, {size_i, size_j, size_k},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
		
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
			size_t size_l = get<0>(std::forward<CallArgs>(args)...).size_l();
			
			const std::string Name = isPool ? "MapPool4D" : "MapOverlap4D";
			DEBUG_TEXT_LEVEL1("CPU " << Name << ": size = " << size_i << " x " << size_j << " x " << size_k << " x " << size_l);
			DEBUG_TEXT_LEVEL1("CPU " << Name << ": kernel = " << this->m_overlap[0] << " x " << this->m_overlap[1] << " x " << this->m_overlap[2] << " x " << this->m_overlap[3]);
			DEBUG_TEXT_LEVEL1("CPU " << Name << ": strides = " << this->m_strides[0] << " x " << this->m_strides[1] << " x " << this->m_strides[2] << " x " << this->m_strides[3]);
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<outArity>(std::forward<CallArgs>(args)...);
			
			RegionType region{arg,
				this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3],
				this->m_edge, this->m_pad,
				(size_i - arg.size_i()) / 2 * this->m_overlap[0], (size_j - arg.size_j()) / 2 * this->m_overlap[1],
				(size_k - arg.size_k()) / 2 * this->m_overlap[2], (size_l - arg.size_l()) / 2 * this->m_overlap[3]
			};
		/*	Index4D start{0, 0, 0, 0}, end{size_i, size_j, size_k, size_l};
			if (this->m_edge == Edge::None)
			{
				start = Index4D{(size_t)this->m_overlap_i, (size_t)this->m_overlap_j, (size_t)this->m_overlap_k, (size_t)this->m_overlap_l};
				end = Index4D{size_i - this->m_overlap_i, size_j - this->m_overlap_j, size_k - this->m_overlap_k, size_l - this->m_overlap_l};
			}
			
			size_t final_size = (end.i - start.i) * (end.j - start.j) * (end.k - start.k) * (end.l - start.l);*/
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(size_i * size_j * size_k * size_l);
			
			for (size_t i = 0; i < size_i; i++)
				for (size_t j = 0; j < size_j; j++)
					for (size_t k = 0; k < size_k; k++)
						for (size_t l = 0; l < size_l; l++)
							if (p == Parity::None || index_parity(p, i, j, k, l))
							{
								region.idx = Index4D{
									i * this->m_strides[0],// + ((this->m_edge == Edge::None) ? this->m_overlap[0] : 0),
									j * this->m_strides[1],// + ((this->m_edge == Edge::None) ? this->m_overlap[1] : 0),
									k * this->m_strides[2],// + ((this->m_edge == Edge::None) ? this->m_overlap[2] : 0),
									l * this->m_strides[3]// + ((this->m_edge == Edge::None) ? this->m_overlap[3] : 0)
								};
								auto res = F::forward(
									MapOverlapFunc::CPU, Index4D{i,j,k,l}, random, region,
									get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
									get<CI>(std::forward<CallArgs>(args)...)...
								);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j, k, l)..., res);
							}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 4D", this, {size_i, size_j, size_k, size_l},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
	} // namespace backend
} // namespace skepu
