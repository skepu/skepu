/*! \file map_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the MapPairs skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel>
		::OMP(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapPairs: hsize = " << Hsize << ", vsize = " << Vsize);
			
			// Sync with device data
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);
			pack_expand((get<HEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<VEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			size_t threads = std::min<size_t>(Vsize, omp_get_max_threads());
			auto random = this->template prepareRandom<MapPairsFunc::randomCount>(Vsize * Hsize, threads, Hsize);
			
#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < Vsize; ++i)
			{
				for (size_t j = 0; j < Hsize; ++j)
				{
					Index2D index{ i, j };
					auto res = F::forward(MapPairsFunc::OMP, index, random(omp_get_thread_num()),
						get<VEI>(std::forward<CallArgs>(args)...)(i)...,
						get<HEI>(std::forward<CallArgs>(args)...)(j)...,
						get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-Varity-Harity-outArity>(typename MapPairsFunc::ProxyTags{}), index)...,
						get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN_INPLACE(this->m_in_place, get<OI>(std::forward<CallArgs>(args)...)(i, j)..., res);
				}
			}
			SKEPU_TRACE_CALL(trace_handle, "MapPairs", this, {Hsize, Vsize},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<VEI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...,
					get<HEI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
	}
}

#endif // SKEPU_OPENMP
