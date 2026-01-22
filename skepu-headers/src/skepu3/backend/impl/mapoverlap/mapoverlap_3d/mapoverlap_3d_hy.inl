/*! \file mapoverlap_3d_hy.inl
*  \brief Contains the definitions of Hybrid execution specific member functions for the MapOverlap3D skeleton.
 */

#ifdef SKEPU_HYBRID

#include <omp.h>

namespace skepu
{
	namespace backend
	{

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_Hybrid(skepu::Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			std::cout << "WARNING: helper_Hybrid is not implemented for Hybrid exection yet. Will run OpenMP version." << std::endl;
			
			this->helper_OpenMP(p, oi, ei, ai, ci, std::forward<CallArgs>(args)...);
		}

    } // backend

} // skepu

#endif // SKEPU_HYBRID