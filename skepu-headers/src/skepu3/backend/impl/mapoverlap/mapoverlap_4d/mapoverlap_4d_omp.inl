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

			// (mirror mode: the edge mode None with an input of the size of the output; only the interior of the output is computed, the elements near the edges keep their values and their random numbers are skipped)
			// The rows of the loop below are the units of the random numbers (the interior ones in mirror mode: they get the numbers one after the other)
			const size_t first_i = this->m_mirror_mode ? this->m_overlap[0] : 0, first_j = this->m_mirror_mode ? this->m_overlap[1] : 0;
			const size_t first_k = this->m_mirror_mode ? this->m_overlap[2] : 0, first_l = this->m_mirror_mode ? this->m_overlap[3] : 0;
			const size_t rows = out_size_i - 2 * first_i;
			const size_t work_unit = (out_size_j - 2 * first_j) * (out_size_k - 2 * first_k) * (out_size_l - 2 * first_l);
			const size_t threads = std::min<size_t>(rows, omp_get_max_threads());
			const size_t final_size = out_size_i * out_size_j * out_size_k * out_size_l;
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(rows * work_unit, threads, work_unit);
			if (this->m_mirror_mode)
				this->template prepareRandom<MapOverlapFunc::randomCount>(final_size - rows * work_unit);

			long offset[4] = {0, 0, 0, 0};
			if (!this->isPool && !this->m_mirror_mode)
			{
				offset[0] = region_offset(out_size_i, arg.size_i(), this->m_overlap[0], this->m_edge);
				offset[1] = region_offset(out_size_j, arg.size_j(), this->m_overlap[1], this->m_edge);
				offset[2] = region_offset(out_size_k, arg.size_k(), this->m_overlap[2], this->m_edge);
				offset[3] = region_offset(out_size_l, arg.size_l(), this->m_overlap[3], this->m_edge);
			}

#pragma omp parallel for schedule(runtime) firstprivate(region)
			for (size_t i = first_i; i < out_size_i - first_i; i++)
				for (size_t j = first_j; j + first_j < out_size_j; j++)
					for (size_t k = first_k; k + first_k < out_size_k; k++)
						for (size_t l = first_l; l + first_l < out_size_l; l++)
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