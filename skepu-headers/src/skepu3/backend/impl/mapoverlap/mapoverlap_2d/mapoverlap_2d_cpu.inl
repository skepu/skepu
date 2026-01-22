/*! \file mapoverlap_2d_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapOverlap2D skeleton.
 */

namespace skepu
{
	namespace backend
	{

        template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			
			Region2D<T> region{arg, this->m_overlap[1], this->m_overlap[0], this->m_edge, this->m_pad};
			
			Index2D start{0, 0}, end{size_i, size_j};
			if (this->m_edge == Edge::None)
			{
				start = Index2D{(size_t)this->m_overlap[1], (size_t)this->m_overlap[0]};
				end = Index2D{size_i - (size_t)this->m_overlap[1], size_j - (size_t)this->m_overlap[0]};
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

    } // backend

} // skepu