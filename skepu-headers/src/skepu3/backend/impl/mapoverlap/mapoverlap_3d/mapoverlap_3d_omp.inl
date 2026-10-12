/*! \file mapoverlap_3d_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapOverlap3D skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI,  typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t out_size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			size_t out_size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			size_t out_size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			// (with the update modes Red and Black, and in mirror mode, some elements of the output are not computed: they keep their values)
			if (p != Parity::None || this->m_mirror_mode)
				pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			
			RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_edge, this->m_pad};
			

			// (mirror mode: the edge mode None with an input of the size of the output; only the interior of the output is computed, the elements near the edges keep their values and their random numbers are skipped)
			// The rows of the loop below are the units of the random numbers (the interior ones in mirror mode: they get the numbers one after the other)
			const size_t first_i = this->m_mirror_mode ? this->m_overlap[0] : 0, first_j = this->m_mirror_mode ? this->m_overlap[1] : 0, first_k = this->m_mirror_mode ? this->m_overlap[2] : 0;
			const size_t rows = out_size_i - 2 * first_i;
			const size_t work_unit = (out_size_j - 2 * first_j) * (out_size_k - 2 * first_k);
			size_t threads = std::min<size_t>(rows, omp_get_max_threads());
			size_t final_size = out_size_i * out_size_j * out_size_k;
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(rows * work_unit, threads, work_unit);
			if (this->m_mirror_mode)
				this->template prepareRandom<MapOverlapFunc::randomCount>(final_size - rows * work_unit);
			
            Index3D offset{0, 0, 0};
            if (!this->isPool && !this->m_mirror_mode)
            {
                if (this->m_edge == Edge::None)
                {
                    offset.i = this->m_overlap[0];
                    offset.j = this->m_overlap[1];
                    offset.k = this->m_overlap[2];
                }
                else
                {
                    offset.i = (arg.size_i() - out_size_i) / 2;
                    offset.j = (arg.size_j() - out_size_j) / 2;
                    offset.k = (arg.size_k() - out_size_k) / 2;
                }
            }

#pragma omp parallel for schedule(runtime) firstprivate(region)
			for (size_t i = first_i; i < out_size_i - first_i; i++)
				for (size_t j = first_j; j + first_j < out_size_j; j++)
					for (size_t k = first_k; k + first_k < out_size_k; k++)
						if (p == Parity::None || index_parity(p, i, j, k))
						{
							region.idx = Index3D{(i + offset.i) * this->m_strides[0], (j + offset.j) * this->m_strides[1], (k + offset.k) * this->m_strides[2]};
							auto res = F::forward(MapOverlapFunc::OMP, Index3D{i + get<0>(std::forward<CallArgs>(args)...).indexOrigin(), j, k}, random(omp_get_thread_num()), region, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j, k)..., res);
						}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 3D", this, {out_size_i, out_size_j, out_size_k},
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