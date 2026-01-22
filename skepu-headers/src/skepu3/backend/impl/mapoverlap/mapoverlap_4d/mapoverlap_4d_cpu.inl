/*! \file mapoverlap_4d_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapOverlap4D skeleton.
 */

namespace skepu
{
	namespace backend
	{

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
			size_t size_l = get<0>(std::forward<CallArgs>(args)...).size_l();
			
			const std::string Name = this->isPool ? "MapPool4D" : "MapOverlap4D";
			DEBUG_TEXT_LEVEL1("CPU " << Name << ": size = " << size_i << " x " << size_j << " x " << size_k << " x " << size_l);
			DEBUG_TEXT_LEVEL1("CPU " << Name << ": kernel = " << this->m_overlap[0] << " x " << this->m_overlap[1] << " x " << this->m_overlap[2] << " x " << this->m_overlap[3]);
			DEBUG_TEXT_LEVEL1("CPU " << Name << ": strides = " << this->m_strides[0] << " x " << this->m_strides[1] << " x " << this->m_strides[2] << " x " << this->m_strides[3]);
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			
			Region4D<T> region{arg,
				this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3],
				this->m_edge, this->m_pad,
				(size_i - arg.size_i()) / 2 * this->m_overlap[0], (size_j - arg.size_j()) / 2 * this->m_overlap[1],
				(size_k - arg.size_k()) / 2 * this->m_overlap[2], (size_l - arg.size_l()) / 2 * this->m_overlap[3]
			};
		/*	Index4D start{0, 0, 0, 0}, end{size_i, size_j, size_k, size_l};
			if (this->m_edge == Edge::None)
			{
				start = Index4D{(size_t)this->m_overlap[0], (size_t)this->m_overlap[1], (size_t)this->m_overlap[2], (size_t)this->m_overlap[3]};
				end = Index4D{size_i - this->m_overlap[0], size_j - this->m_overlap[1], size_k - this->m_overlap[2], size_l - this->m_overlap[3]};
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

    } // backend

} // skepu