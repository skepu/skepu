/*! \file mapoverlap_4d_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapOverlap4D skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			const size_t out_size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			const size_t out_size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			const size_t out_size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
			const size_t out_size_l = get<0>(std::forward<CallArgs>(args)...).size_l();

			const std::string Name = this->isPool ? "MapPool4D" : "MapOverlap4D";
			DEBUG_TEXT_LEVEL1("OpenMP " << Name << ": size = " << out_size_i << " x " << out_size_j << " x " << out_size_k << " x " << out_size_l);
			DEBUG_TEXT_LEVEL1("OpenMP " << Name << ": kernel = " << this->m_overlap[0] << " x " << this->m_overlap[1] << " x " << this->m_overlap[2] << " x " << this->m_overlap[3]);
			DEBUG_TEXT_LEVEL1("OpenMP " << Name << ": strides = " << this->m_strides[0] << " x " << this->m_strides[1] << " x " << this->m_strides[2] << " x " << this->m_strides[3]);


			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);

			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

			RegionType region{arg,
				this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3],
				this->m_edge, this->m_pad,
				(out_size_i - arg.size_i()) / 2 * this->m_overlap[0], (out_size_j - arg.size_j()) / 2 * this->m_overlap[1],
				(out_size_k - arg.size_k()) / 2 * this->m_overlap[2], (out_size_l - arg.size_l()) / 2 * this->m_overlap[3]
			};

			const size_t threads = std::min<size_t>(out_size_i, omp_get_max_threads());
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(out_size_i * out_size_j * out_size_k * out_size_l, threads, out_size_j * out_size_k * out_size_l);

			long offset[4] = {0, 0, 0, 0};
			if (!this->isPool)
			{
				offset[0] = region_offset(out_size_i, arg.size_i(), this->m_overlap[0], this->m_edge);
				offset[1] = region_offset(out_size_j, arg.size_j(), this->m_overlap[1], this->m_edge);
				offset[2] = region_offset(out_size_k, arg.size_k(), this->m_overlap[2], this->m_edge);
				offset[3] = region_offset(out_size_l, arg.size_l(), this->m_overlap[3], this->m_edge);
			}

#pragma omp parallel for schedule(runtime) firstprivate(region)
			for (size_t i = 0; i < out_size_i; i++)
				for (size_t j = 0; j < out_size_j; j++)
					for (size_t k = 0; k < out_size_k; k++)
						for (size_t l = 0; l < out_size_l; l++)
							if (p == Parity::None || index_parity(p, i, j, k, l))
							{
								region.idx = Index4D{
                                    (size_t)((long)(i * this->m_strides[0]) + offset[0]),
										(size_t)((long)(j * this->m_strides[1]) + offset[1]),
										(size_t)((long)(k * this->m_strides[2]) + offset[2]),
										(size_t)((long)(l * this->m_strides[3]) + offset[3])
                                };
								auto res = F::forward(
									MapOverlapFunc::OMP, Index4D{i + get<0>(std::forward<CallArgs>(args)...).indexOrigin(), j, k, l}, random(omp_get_thread_num()), region,
									get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...
								);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j, k, l)..., res);
							}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 4D", this, {out_size_i, out_size_j, out_size_k, out_size_l},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...),
				this->template randomID<MapOverlapFunc::randomCount>()
			);
		}

    } // backend

} // skepu

#endif // SKEPU_OPENMP