#pragma once
#ifndef SKEPU_CLUSTER_MAP_HPP
#define SKEPU_CLUSTER_MAP_HPP 1

#include <tuple>
#include <starpu_mpi.h>

#include <omp.h>

#ifdef SKEPU_CUDA
#include <cuda.h>
#endif // SKEPU_CUDA

#include "skepu3/cluster/cluster.hpp"
#include "skepu3/cluster/common.hpp"
#include "skepu3/cluster/skeletons/skeleton_base.hpp"
#include "skepu3/cluster/skeletons/skeleton_task.hpp"
#include "skepu3/cluster/skeletons/skeleton_utils.hpp"

namespace skepu {
namespace backend {
namespace starpu {

template<typename MapFunc, typename CUKernel>
struct map
{
	typedef ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>
		F;

	template<
		typename Buffers,
		typename Iterator,
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename... CallArgs>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		Iterator begin,
		size_t count,
		CallArgs &&... args) noexcept
	-> void
	{
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t i = 0; i < count; ++i)
		{
			auto res =
				F::forward(
					MapFunc::OMP,
					(begin +i).index(),
					// Elwise elements are also raw pointers.
					std::get<EI>(buffers)[i]...,
					// Set MatRow to correct position. Does nothing for others.
					cluster::advance(std::get<CI>(buffers), i)...,
					args...);
			std::tie(std::get<RI>(buffers)[i]...) = res;
		}
	}

	#ifdef SKEPU_CUDA
	CUKernel static cu_kernel;

	template<
		typename Buffers,
		typename Iterator,
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename... CallArgs>
	auto static
	CUDA(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		Iterator begin,
		size_t count,
		CallArgs &&... args) noexcept
	-> void
	{
		dim3 block{std::min<unsigned int>(count, 1024)};
		dim3 grid{(unsigned int)ceil((double)count/1024)};
		auto stream = starpu_cuda_get_local_stream();
		size_t size_j = begin.getParent().size_j();
		size_t size_k = begin.getParent().size_k();
		size_t size_l = begin.getParent().size_l();
		
		StrideList<sizeof...(RI) + sizeof...(EI)> strides{};

		cu_kernel<<<grid, block, 0, stream>>>(
			std::get<RI>(buffers)...,
			SKEPU_PRNG_PLACEHOLDER,
			std::get<EI>(buffers)...,
			std::get<CI>(buffers)...,
			args...,
			size_j,
			size_k,
			size_l,
			count,
			begin.offset(),
			strides
		);

		cudaStreamSynchronize(stream);
	}
	#endif
};

#ifdef SKEPU_CUDA
template<typename MapFunc, typename CUKernel>
CUKernel map<MapFunc, CUKernel>::cu_kernel = 0;
#endif

} // namespace _starpu

template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
class Map
: public cluster::skeleton_task<
		starpu::map<MapFunc, CUDAKernel>,
		typename cluster::result_tuple<typename MapFunc::Ret>::type,
		typename MapFunc::ElwiseArgs,
		typename MapFunc::ContainerArgs,
		typename MapFunc::UniformArgs>
{
	typedef typename MapFunc::Ret T;

	typedef cluster::skeleton_task<
			starpu::map<MapFunc, CUDAKernel>,
			typename cluster::result_tuple<typename MapFunc::Ret>::type,
			typename MapFunc::ElwiseArgs,
			typename MapFunc::ContainerArgs,
			typename MapFunc::UniformArgs>
		skeleton_task;

	static constexpr size_t inArgs =
		MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
	static constexpr size_t outArgs =
		MapFunc::outArity;
	static constexpr size_t numArgs =
		inArgs + outArgs;
	static constexpr size_t anyArity =
		std::tuple_size<typename MapFunc::ContainerArgs>::value;

	static constexpr typename make_pack_indices<outArgs>::type out_indices{};
	static constexpr
		typename make_pack_indices<arity + outArgs, outArgs>::type elwise_indices{};
	static constexpr
			typename
				make_pack_indices<arity + anyArity + outArgs, arity + outArgs>::type
		any_indices{};
	static constexpr
		typename make_pack_indices<anyArity, 0>::type ptag_indices{};
	static constexpr
			typename make_pack_indices<numArgs, arity + anyArity + outArgs>::type
		const_indices{};

public:
	Map(CUDAKernel kernel)
	: skeleton_task("SkePU Map")
	{
		#ifdef SKEPU_CUDA
		starpu::map<MapFunc, CUDAKernel>::cu_kernel = kernel;
		#endif
	}

	~Map() noexcept
	{
		skepu::cluster::barrier();
	}

	template<typename... Args>
	auto tune(Args&&...) noexcept -> void {}

	template<typename... CallArgs>
	auto
	operator()(CallArgs&&... args) noexcept
	-> decltype(get<0>(args...))
	{
		static_assert(sizeof...(CallArgs) == numArgs,
			"Number of arguments not matching Map function");

		auto & res = get<0>(args...);

		backendDispatch(
			out_indices,
			elwise_indices,
			any_indices,
			const_indices,
			ptag_indices,
			res.begin(),
			res.end(),
			std::forward<CallArgs>(args)...);

		return res;
	}

	template<typename Iterator, typename... CallArgs,
		REQUIRES(is_skepu_iterator<Iterator, T>::value)>
	auto
	operator()(Iterator begin, Iterator end, CallArgs&&... args) noexcept
	-> Iterator
	{
		static_assert(sizeof...(CallArgs) == numArgs -1,
			"Number of arguments not matching Map function");

		backendDispatch(
			out_indices,
			elwise_indices,
			any_indices,
			const_indices,
			ptag_indices,
			begin,
			end,
			begin,
			std::forward<CallArgs>(args)...);
		return begin;
	}

private:
	template<
		size_t ... OI,
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename Iterator,
		typename ... CallArgs>
	auto
	STARPU(
		pack_indices<OI...>,
		pack_indices<EI...>,
		pack_indices<AI...>,
		pack_indices<CI...>,
		pack_indices<PI...>,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> void
	{
		auto constexpr static proxy_tags = typename MapFunc::ProxyTags{};

		// Since attribute maybe_unsed is not available until C++17, we use this
		// trick instead to get rid of the unused variable warnings for proxy_tags.
		if(sizeof proxy_tags)
			;

		// The proxy elements require the data to be gathered on all nodes.
		// Except for MatRow, which will be partitioned.
		pack_expand((
			skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(proxy_tags)),0)...);

		// And the same goes for the elwise arguments.
		// TODO: They should be realinged so that element i in the elwise is on the
		// same rank as element i in the result. This will ensure that the number of
		// tasks generated is as low as possible.
		pack_expand((cont::getParent(get<EI>(args...)).partition(),0)...);

		auto filter_parts =
			std::max<size_t>({
				cont::getParent(get<OI>(args...)).min_filter_parts()...,
				cont::getParent(get<EI>(args...)).min_filter_parts()...,
				cluster::min_filter_parts_container_arg(
					cont::getParent(get<AI>(args...)),
					std::get<PI>(proxy_tags))...});

		pack_expand(
			(cont::getParent(get<OI>(args...)).filter(filter_parts), 0)...,
			(cont::getParent(get<EI>(args...)).filter(filter_parts), 0)...,
			(cluster::filter(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(proxy_tags),
				filter_parts), 0)...);

		// The result will be partitioned.
		pack_expand(
			(cont::getParent(get<OI>(args...)).partition(),
			(cont::getParent(get<OI>(args...)).invalidate_local_storage()),0)...);

		while(begin != end)
		{
			auto pos = begin.offset();

			auto task_count = std::min({
				(size_t)(end - begin),
				begin.getParent().block_count_from(pos),
				cont::getParent(get<EI>(args...)).block_count_from(pos)...});

			auto handles =
				std::make_tuple(
					cont::getParent(get<OI>(args...)).handle_for(pos)...,
					cont::getParent(get<EI>(args...)).handle_for(pos)...,
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(proxy_tags),
						pos)...);
			auto call_back_args = std::make_tuple(begin, task_count);

			this->schedule(
				handles,
				begin,
				task_count,
				std::forward<decltype(get<CI>(args...))>(get<CI>(args...))...);

			begin += task_count;
		}
	}

	template<
		size_t ... OI,
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename Iterator,
		typename ... CallArgs>
	auto
	backendDispatch(
		pack_indices<OI...> oi,
		pack_indices<EI...> ei,
		pack_indices<AI...> ai,
		pack_indices<CI...> ci,
		pack_indices<PI...> pi,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> void
	{
		/* TODO:
		 * This should be more involved with possible backend selection. We only
		 * have OMP at this point, so we just redirect the call.
		 */
		STARPU(
			oi,
			ei,
			ai,
			ci,
			pi,
			begin,
			end,
			std::forward<CallArgs>(args)...);
	}
};

} // namespace backend
} // namespace skepu

#endif // SKEPU_CLUSTER_MAP_HPP
