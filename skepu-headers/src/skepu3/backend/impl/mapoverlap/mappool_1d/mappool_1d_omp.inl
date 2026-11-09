#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPool1D<MapFunc, CUDAKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapPool1D: size = " << size << " threads = " << omp_get_max_threads());

			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-1-outArity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-1-outArity])), 0)...);
			pack_expand((get<OI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(), 0)...);

			SKEPU_TRACE_START_EVENT(trace_handle);
			const T *input = get<outArity>(std::forward<CallArgs>(args)...).getAddress();
			size_t threads = std::min<size_t>(size, omp_get_max_threads());
			auto random = this->template prepareRandom<MapFunc::randomCount>(size, threads);

#pragma omp parallel for schedule(runtime)
			for (size_t i = 0; i < size; ++i)
			{
				auto index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
				Pool1D<T> pool(this->m_pool_size, 1, input + i * this->m_stride);
				auto res = F::forward(MapFunc::OMP, index, random(omp_get_thread_num()), pool,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-1-outArity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
				SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
			}

			SKEPU_TRACE_CALL(trace_handle, "MapPool 1D", this, {size},
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
