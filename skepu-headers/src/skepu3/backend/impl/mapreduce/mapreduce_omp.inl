/*! \file mapreduce_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapReduce skeleton.
*/

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>
#include <vector>

namespace skepu
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
		scalar_wrapper_t<typename MapFunc::Ret> MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapReduce (mode A): size = " << size);
			// Sync with device data
			pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);

			SKEPU_TRACE_START_EVENT(trace_handle);

			std::vector<Ret> parsums(std::min<size_t>(size, omp_get_max_threads()));
			bool first = true;
			
			size_t threads = std::min<size_t>(size, omp_get_max_threads());
			auto random = this->template prepareRandom<MapFunc::randomCount>(size, threads);
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel for schedule(runtime) firstprivate(first)
			for (size_t i = 0; i < size; ++i)
			{
				size_t myid = omp_get_thread_num();
				TempIndexType index;
				if constexpr (MapFunc::indexed && sizeof...(EI) > 0)
					index = (get<0>(std::forward<CallArgs>(args)...) + i).getIndex();
				else
					index = make_index(defaultDim{}, i, this->default_size_j, this->default_size_k, this->default_size_l);
				
				Ret tempMap = F::forward(MapFunc::OMP,
					index, random(omp_get_thread_num()),
					get<EI>(std::forward<CallArgs>(args)...)(i)...,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
				if (first)
				{
					parsums[myid] = tempMap;
					first = false;
				}
				else
					pack_expand((get_or_return<OI>(parsums[myid]) = ReduceFunc::OMP(get_or_return<OI>(parsums[myid]), get_or_return<OI>(tempMap)), 0)...);
			}
			
			// Final Reduce sequentially
			for (Ret const& parsum : parsums)
				pack_expand((get_or_return<OI>(res) = ReduceFunc::OMP(get_or_return<OI>(res), get_or_return<OI>(parsum)), 0)...);

			auto result = scalar_wrapper_t<typename MapFunc::Ret>(res);
			SKEPU_TRACE_CALL(trace_handle, "MapReduce", this, {size}, tracing::scalar_output_labels(result),
				{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
			return result;
		}
		
	} // namespace backend
} // namespace skepu

#endif
