/*! \file map_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP Map: label = '" << this->getLabel() << "' size = " << size << " threads = " << omp_get_max_threads());

			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);

			SKEPU_TRACE_START_EVENT(trace_handle);
			size_t threads = std::min<size_t>(size, omp_get_max_threads());
			auto random = this->template prepareRandom<MapFunc::randomCount>(size, threads);

#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < size; ++i)
			{
				auto index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
				auto res = F::forward(MapFunc::OMP, index, random(omp_get_thread_num()),
					get<EI>(std::forward<CallArgs>(args)...)(i)...,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity-outArity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
				SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
			}

			SKEPU_TRACE_CALL(trace_handle, "Map", this, {size},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...),
				this->template randomID<MapFunc::randomCount>()
			);
		}
	}
}

#endif // SKEPU_OPENMP
