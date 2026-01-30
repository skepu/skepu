/*! \file mapoverlap_3d_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapOverlap3D skeleton.
 */

namespace skepu
{
	namespace backend
	{

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
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			
			Region3D<T> region{arg, this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_edge, this->m_pad};
			
			Index3D offset{0, 0, 0};
			if (this->m_edge == Edge::None)
			{
				offset.i = this->m_overlap[0];
				offset.j = this->m_overlap[1];
				offset.k = this->m_overlap[2];
			}
			
			size_t final_size = size_i * size_j * size_k;
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(final_size);
			
			for (size_t i = 0; i < size_i; i++)
				for (size_t j = 0; j < size_j; j++)
					for (size_t k = 0; k < size_k; k++)
						if (p == Parity::None || index_parity(p, i, j, k))
						{
							region.idx = Index3D{(i + offset.i) * this->m_strides[0], (j + offset.j) * this->m_strides[1], (k + offset.k) * this->m_strides[2]};
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

    } // backend

} // skepu