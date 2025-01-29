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
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CLKernel>
		::OMP(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP MapPairsReduce: hsize = " << Hsize << ", vsize = " << Vsize);
			
			// Sync with device data
			pack_expand((get<HEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<VEI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().invalidateDeviceData(hasWriteAccess(MapPairsFunc::anyAccessMode[AI-Varity-Harity])), 0)...);
			SKEPU_TRACE_START_EVENT(trace_handle);
			
			size_t threads = std::min<size_t>(Vsize, omp_get_max_threads());
			auto random = (this->m_mode == ReduceMode::RowWise)
				? this->template prepareRandom<MapPairsFunc::randomCount>(Vsize * Hsize, threads, Hsize)
				: this->template prepareRandom<MapPairsFunc::randomCount>(Hsize * Vsize, threads, Vsize);
			
			if (this->m_mode == ReduceMode::RowWise)
#pragma omp parallel for schedule(runtime)
				for (size_t i = 0; i < Vsize; ++i)
				{
					pack_expand((get<OI>(std::forward<CallArgs>(args)...)(i) = get_or_return<OI>(this->m_start), 0)...);
					for (size_t j = 0; j < Hsize; ++j)
					{
						auto index = Index2D { i, j };
						auto temp = F::forward(MapPairsFunc::OMP, Index2D{ i, j }, random(omp_get_thread_num()),
							get<VEI>(std::forward<CallArgs>(args)...)(i)...,
							get<HEI>(std::forward<CallArgs>(args)...)(j)...,
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
							get<CI>(std::forward<CallArgs>(args)...)...
						);
						pack_expand((get<OI>(std::forward<CallArgs>(args)...)(i) = ReduceFunc::CPU(get<OI>(std::forward<CallArgs>(args)...)(i), get_or_return<OI>(temp)), 0)...);
					}
				}
			else if (this->m_mode == ReduceMode::ColWise)
#pragma omp parallel for schedule(runtime)
				for (size_t j = 0; j < Hsize; ++j) // TODO: optimize?
				{
					pack_expand((get<OI>(std::forward<CallArgs>(args)...)(j) = get_or_return<OI>(this->m_start), 0)...);
					for (size_t i = 0; i < Vsize; ++i)
					{
						auto index = Index2D { i, j };
						auto temp = F::forward(MapPairsFunc::OMP, Index2D{ i, j }, random(omp_get_thread_num()),
							get<VEI>(std::forward<CallArgs>(args)...)(i)...,
							get<HEI>(std::forward<CallArgs>(args)...)(j)...,
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
							get<CI>(std::forward<CallArgs>(args)...)...
						);
						pack_expand((get<OI>(std::forward<CallArgs>(args)...)(j) = ReduceFunc::CPU(get<OI>(std::forward<CallArgs>(args)...)(j), get_or_return<OI>(temp)), 0)...);
					}
				}
			SKEPU_TRACE_CALL(trace_handle, "MapPairsReduce", this, {Hsize, Vsize},
				{get<OI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<VEI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...,
					get<HEI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
				{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
				tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...)
			);
		}
		
	} // namespace backend
} // namespace skepu

#endif
