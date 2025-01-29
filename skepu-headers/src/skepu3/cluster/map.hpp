#ifndef MAP_HPP
#define MAP_HPP

#include <starpu.h>
#include <skepu3/impl/meta_helpers.hpp>
#include <skepu3/cluster/index.hpp>
#include <skepu3/cluster/cluster_meta_helpers.hpp>
#include <skepu3/cluster/matrix.hpp>
#include <skepu3/cluster/matrix_iterator.hpp>
#include <skepu3/cluster/multivariant_task.hpp>
#include <cstddef>

#include <tuple>

namespace skepu
{
	namespace backend
	{
		template<size_t arity,
		         typename MapFunc,
		         typename CUDAKernel,
		         typename CLKernel>
		class Map :
			cluster::multivariant_task<std::tuple<typename MapFunc::Ret>,
			                           typename MapFunc::ElwiseArgs,
			                           typename skepu::helpers
			                           ::mat_tuple_to_raw_type_tuple<
				                           typename MapFunc::ContainerArgs>::unwrapped,
			                           typename MapFunc::UniformArgs,
			                           Map<arity,
			                               MapFunc,
			                               CUDAKernel,
			                               CLKernel>>
		{
			// Types
			using Self = Map<arity, MapFunc, CUDAKernel, CLKernel>;
			using T = typename MapFunc::Ret;
			using F = ConditionalIndexForwarder<MapFunc::indexed,
			                                    decltype(&MapFunc::CPU)>;

		public:
			//static constexpr auto skeletonType = SkeletonType::Map;
			using ResultArgs = std::tuple<T>;
			using ElwiseArgs = typename MapFunc::ElwiseArgs;
			using ContainerArgs = typename MapFunc::ContainerArgs;
			using UniformArgs = typename MapFunc::UniformArgs;
		private:

			static constexpr size_t numArgs =
				MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
			static constexpr size_t anyArity =
				std::tuple_size<typename MapFunc::ContainerArgs>::value;

			// Members
			CUDAKernel m_cuda_kernel;
			size_t m_default_size_x;
			size_t m_default_size_y;



			// Plumbing functions

			template<size_t... EI,
			         size_t... AI,
			         size_t... CI,
			         typename Iterator,
			         typename... CallArgs>
			void STARPU(size_t size,
			            pack_indices<EI...>,
			            pack_indices<AI...>,
			            pack_indices<CI...>,
			            Iterator res,
			            CallArgs&&... args);


			template<size_t... EI,
			         size_t... AI,
			         size_t... CI,
			         typename... CallArgs>
			void backendDispatch(pack_indices<EI...> ei,
			                     pack_indices<AI...> ai,
			                     pack_indices<CI...> ci,
			                     Size2D size,
			                     MatrixIterator<T> res,
			                     CallArgs&&... args);
		public:
			// CPU implementation
			template<typename MatT,
			         size_t... RI,
			         size_t... EI,
			         size_t... CI,
			         typename... Uniform>
			static void cpu(const void * self,
			                Size2D size,
			                Offset2D global_offset,
			                MatT && bufs,
			                pack_indices<RI...>,
			                pack_indices<EI...>,
			                pack_indices<CI...>,
			                Uniform... args);


			//static constexpr auto skeletonType = SkeletonType::Map;
			static constexpr bool prefers_matrix = MapFunc::prefersMatrix;

			static constexpr typename
			make_pack_indices<arity, 0>::type elwise_indices{};

			static constexpr typename
			make_pack_indices<arity + anyArity, arity>::type any_indices{};

			static constexpr typename
			make_pack_indices<numArgs, arity + anyArity>::type const_indices{};

			Map(CUDAKernel kernel);
			void setDefaultSize(size_t x, size_t y = 0);


			template<template<class> class Container,
			         typename... CallArgs,
			         REQUIRES(is_skepu_container<Container<T>>::value)>
			Container<T> &operator()(Container<T> &res, CallArgs&&... args)
				{
					this->backendDispatch(elwise_indices,
					                      any_indices,
					                      const_indices,
					                      res.size2D(),
					                      res.begin(),
					                      args...);
					return res;
				}


			template<template<class> class Iterator,
			         typename... CallArgs,
			         REQUIRES(is_skepu_matrix_iterator<Iterator<T>>::value)>
			Iterator<T> operator()(Iterator<T> res,
			                       Iterator<T> res_end,
			                       CallArgs&&... args)
				{
					this->backendDispatch(elwise_indices,
					                      any_indices,
					                      const_indices,
					                      res % res_end,
					                      res,
					                      args...);
					return res;
				}
		};
	} // backend
} // skepu

#include <skepu3/cluster/impl/map.inl>

#endif // MAP_HPP
