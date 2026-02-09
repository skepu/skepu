/*! \file mapoverlap_2d_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapOverlap2D skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{

        template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			size_t out_size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
			size_t out_size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
			
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI-InArity-OutArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
			
			RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_edge, this->m_pad};
			
			Index2D offset{0, 0};
            if (!this->isPool)
            {
                if (this->m_edge == Edge::None)
                {
                    offset.row = this->m_overlap[0];
                    offset.col = this->m_overlap[1];
                }
                else
                {
                    offset.row = (arg.size_i() - out_size_i) / 2;
                    offset.col = (arg.size_j() - out_size_j) / 2;
                }
            }

			size_t threads = std::min<size_t>(out_size_i, omp_get_max_threads());
			size_t final_size = out_size_i * out_size_j;
			size_t work_unit = out_size_j;
			auto random = this->template prepareRandom<MapOverlapFunc::randomCount>(final_size, threads, work_unit);
			
#pragma omp parallel for schedule(runtime) firstprivate(region)
			for (size_t i = 0; i < out_size_i; i++)
				for (size_t j = 0; j < out_size_j; j++)
					if (p == Parity::None || index_parity(p, i, j))
					{
						region.idx = Index2D{(i + offset.row) * this->m_strides[0], (j + offset.col) *this->m_strides[1]};
						auto res = F::forward(MapOverlapFunc::OMP, Index2D{i,j}, random(omp_get_thread_num()), region, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
						SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j)..., res);
					}
			SKEPU_TRACE_CALL(trace_handle, "MapOverlap 2D", this, {out_size_i, out_size_j},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}

    } // backend

} // skepu

#endif // SKEPU_OPENMP