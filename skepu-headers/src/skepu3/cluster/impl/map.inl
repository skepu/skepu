#ifndef MAP_INL
#define MAP_INL

#include <utility>

#include <skepu3/cluster/map.hpp>
#include <starpu.h>
#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<size_t arity,
		         typename MapFunc,
		         typename CUDAKernel,
		         typename CLKernel>
		Map<arity,MapFunc,CUDAKernel,CLKernel>
		::Map(CUDAKernel kernel) : m_cuda_kernel { kernel } {}


		template<size_t arity,
		         typename MapFunc,
		         typename CUDAKernel,
		         typename CLKernel>
		void Map<arity,MapFunc,CUDAKernel,CLKernel>
		::setDefaultSize(size_t x, size_t y)
		{
			this->m_default_size_x = x;
			this->m_default_size_y = y;
		}


		template<size_t arity,
		         typename MapFunc,
		         typename CUDAKernel,
		         typename CLKernel>
		template<size_t... EI,
		         size_t... AI,
		         size_t... CI,
		         typename... CallArgs>
		void Map<arity,MapFunc,CUDAKernel,CLKernel>
		::backendDispatch(pack_indices<EI...>,
		                  pack_indices<AI...>,
		                  pack_indices<CI...>,
		                  Size2D size,
		                  MatrixIterator<T> res,
		                  CallArgs&&... args)
		{
			auto elemwise_iterators = std::make_tuple(get<EI>(args...).begin()...);
			this->element_aligned(size,
			                      res,
			                      std::get<EI>(elemwise_iterators)...,
			                      get<AI>(args...)...,
			                      get<CI>(args...)...);
		}


static int iteration = 0;
		template<size_t arity,
		         typename MapFunc,
		         typename CUDAKernel,
		         typename CLKernel>
		template<typename MatT,
		         size_t... RI,
		         size_t... EI,
		         size_t... CI,
		         typename... Uniform>
		void Map<arity,MapFunc,CUDAKernel,CLKernel>
		::cpu(const void* /* self */,
		      Size2D size,
		      Offset2D global_offset,
		      MatT && bufs, // skepu::Mat<T>...
		      pack_indices<RI...>,
		      pack_indices<EI...>,
		      pack_indices<CI...>,
		      Uniform... args)
		{
			auto & res = std::get<0>(bufs);

			for(size_t row = 0; row < size.row; ++row) {
				#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
				for(size_t col = 0; col < size.col; ++col) {
					Index2D i {};
					i.col = col;
					i.row = row;
					i.i = i.row*res.ld + i.col;
					res[i.i] =
						F::forward(MapFunc::OMP,
						           global_offset + i,
						           std::get<EI>(bufs)[i.i]...,
						           std::get<CI>(bufs)...,
						           args...);
				}
			}
		}
	} // backend
} // skepu

#endif /* MAP_INL */
