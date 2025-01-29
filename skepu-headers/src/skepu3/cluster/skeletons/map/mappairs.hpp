#pragma once
#ifndef SKEPU_CLUSTER_SKELETONS_MAPPAIRS_HPP
#define SKEPU_CLUSTER_SKELETONS_MAPPAIRS_HPP 1

#include <omp.h>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>
#include "skepu3/cluster/skeletons/skeleton_base.hpp"
#include "skepu3/cluster/skeletons/skeleton_task.hpp"
#include "skepu3/cluster/skeletons/skeleton_utils.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename Func, typename CUKernel>
struct map_pairs
{
	typedef ConditionalIndexForwarder<
			Func::indexed,
			decltype(&Func::CPU)>
		F;

	template<
		size_t ... RHI,
		size_t ... HI,
		size_t ... CI,
		size_t ... VEI,
		size_t ... HEI,
		typename Buffers,
		typename ... Args>
	auto static
	run(
		pack_indices<RHI...>,
		pack_indices<HI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		pack_indices<VEI...>,
		pack_indices<HEI...>,
		size_t start_row,
		size_t rows,
		size_t width,
		Args && ... args) noexcept
	-> void
	{
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t i = 0; i < rows; ++i)
		{
			auto row_offset = i * width;
			for(size_t j = 0; j < width; ++j)
			{
				auto res =
					F::forward(
						Func::OMP, Index2D{start_row + i, j},
						std::get<VEI>(buffers)[i]...,
						std::get<HEI>(buffers)[j]...,
						cluster::advance(std::get<CI>(buffers), i)...,
						args...);
				std::tie(
					std::get<RHI>(buffers)[row_offset + j]...) = res;
			}
		}
	}

	#ifdef SKEPU_CUDA
	CUKernel static cu_kernel;

	template<
		size_t ... RHI,
		size_t ... HI,
		size_t ... CI,
		size_t ... VEI,
		size_t ... HEI,
		typename Buffers,
		typename ... Args>
	auto static
	CUDA(
		pack_indices<RHI...>,
		pack_indices<HI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		pack_indices<VEI...>,
		pack_indices<HEI...>,
		size_t start_row,
		size_t rows,
		size_t width,
		Args && ... args) noexcept
	-> void
	{
		std::cerr << "[SkePU][MapPairs] CUDA kernel!\n";
		unsigned int count = rows * width;
		dim3 block{std::min<unsigned int>(count, 1024)};
		dim3 grid{count / block.x};
		if(count % grid.x)
			++grid.x;
		auto stream = starpu_cuda_get_local_stream();

		cu_kernel<<<grid, block, 0, stream>>>(
			std::get<RHI>(buffers)...,
			std::get<VEI>(buffers)...,
			std::get<HEI>(buffers)...,
			std::get<CI>(buffers)...,
			args...,
			rows,
			width,
			start_row * width);

		cudaStreamSynchronize(stream);
	}
	#endif
};

#ifdef SKEPU_CUDA
template<typename MapFunc, typename CUKernel>
CUKernel map_pairs<MapFunc, CUKernel>::cu_kernel = 0;
#endif

} // namespace _starpu

template<
	size_t Varity,
	size_t Harity,
	typename MapPairsFunc,
	typename CUDAKernel,
	typename CLKernel>
class MapPairs
:	private cluster::skeleton_task<
		_starpu::map_pairs<MapPairsFunc, CUDAKernel>,
		typename cluster::result_tuple<typename MapPairsFunc::Ret>::type,
		typename MapPairsFunc::ElwiseArgs,
		typename MapPairsFunc::ContainerArgs,
		typename MapPairsFunc::UniformArgs>
{
	typedef typename MapPairsFunc::Ret T;
	typedef cluster::skeleton_task<
			_starpu::map_pairs<MapPairsFunc, CUDAKernel>,
			typename cluster::result_tuple<typename MapPairsFunc::Ret>::type,
			typename MapPairsFunc::ElwiseArgs,
			typename MapPairsFunc::ContainerArgs,
			typename MapPairsFunc::UniformArgs>
		skeleton_task;

	static constexpr size_t outArity = MapPairsFunc::outArity;
	static constexpr size_t numArgs =
		MapPairsFunc::totalArity - (MapPairsFunc::indexed ? 1 : 0) + outArity;
	static constexpr size_t anyArity =
		std::tuple_size<typename MapPairsFunc::ContainerArgs>::value;

	typedef std::tuple<T> ResultArg;
	typedef typename MapPairsFunc::ElwiseArgs ElwiseArgs;
	typedef typename MapPairsFunc::ContainerArgs ContainerArgs;
	typedef typename MapPairsFunc::UniformArgs UniformArgs;

	auto static constexpr out_indices =
		typename make_pack_indices<outArity, 0>::type{};
	auto static constexpr Velwise_indices =
		typename make_pack_indices<outArity + Varity, outArity>::type{};
	auto static constexpr Helwise_indices =
		typename make_pack_indices<outArity + Varity + Harity, outArity + Varity>
			::type{};
	auto static constexpr elwise_indices =
			typename make_pack_indices<outArity + Varity + Harity, outArity>::type{};
	auto static constexpr any_indices =
		typename make_pack_indices<
				outArity + Varity + Harity + anyArity,
				outArity + Varity + Harity>
			::type{};
	auto static constexpr const_indices =
		typename make_pack_indices<numArgs, outArity + Varity + Harity + anyArity>
			::type{};
	typename MapPairsFunc::ProxyTags proxy_tags{};
	auto static constexpr proxy_tag_indices =
		typename make_pack_indices<
			std::tuple_size<typename MapPairsFunc::ProxyTags>::value>::type{};

public:
	static constexpr auto skeletonType = SkeletonType::MapPairs;

	MapPairs(CUDAKernel kernel)
	: skeleton_task("MapPairs")
	{
		#ifdef SKEPU_CUDA
		_starpu::map_pairs<MapPairsFunc, CUDAKernel>::cu_kernel = kernel;
		#endif
	}

	~MapPairs() noexcept
	{
		skepu::cluster::barrier();
	}

	template<typename... CallArgs>
	auto
	operator()(CallArgs&&... args)
	-> typename std::add_lvalue_reference<decltype(get<0>(args...))>::type
	{
		this->backendDispatch(
			out_indices,
			Velwise_indices,
			Helwise_indices,
			any_indices,
			const_indices,
			proxy_tag_indices,
			get<0>(args...).size_i(),
			get<0>(args...).size_j(),
			std::forward<CallArgs>(args)...);
		return get<0>(args...);
	}

private:
	template<
		size_t... OI,
		size_t... VEI,
		size_t... HEI,
		size_t... AI,
		size_t... CI,
		size_t ... PI,
		typename... CallArgs>
	auto
	backendDispatch(
		pack_indices<OI...>,
		pack_indices<VEI...> vei,
		pack_indices<HEI...> hei,
		pack_indices<AI...>,
		pack_indices<CI...>,
		pack_indices<PI...>,
		size_t Vsize,
		size_t Hsize,
		CallArgs&&... args)
	-> void
	{
		if(disjunction((get<OI>(std::forward<CallArgs>(args)...).size_i() < Vsize)...)
				|| disjunction((get<OI>(std::forward<CallArgs>(args)...).size_j() < Hsize)...))
			SKEPU_ERROR("Non-matching output container sizes");

		if(disjunction((get<VEI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
			SKEPU_ERROR("Non-matching vertical container sizes");

		if(disjunction((get<HEI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
			SKEPU_ERROR("Non-matching horizontal container sizes");

		/* Data locality
		 * The proxy's will be gathered or partitioned depending on if it is a
		 * Matrow proxy or not.
		 * The output matrices will be partitioned, and so will the row element
		 * wise arguments.
		 * The column wise element arguments will however have to be gathered,
		 * since they need to be accessed in their entirety on every node during
		 * the map.
		 */
		pack_expand(
			(skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(proxy_tags)),0)...,
			(cont::getParent(get<OI>(args...)).partition(),0)...,
			(cont::getParent(get<VEI>(args...)).partition(),0)...,
			(cont::getParent(get<HEI>(args...)).allgather(),0)...);

		auto parts =
			std::max<size_t>({
				cont::getParent(get<OI>(args...)).min_filter_parts()...,
				cont::getParent(get<VEI>(args...)).min_filter_parts()...,
				cont::getParent(get<HEI>(args...)).min_filter_parts()...,
				cluster::min_filter_parts_container_arg(
					cont::getParent(get<AI>(args...)),
					std::get<PI>(proxy_tags))...});

		pack_expand(
			(cont::getParent(get<OI>(args...)).filter(parts), 0)...,
			(cont::getParent(get<VEI>(args...)).filter(parts), 0)...,
			(cont::getParent(get<HEI>(args...)).filter(parts), 0)...,
			(cluster::filter(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(proxy_tags),
				parts), 0)...);

		pack_expand(
			(cont::getParent(get<OI>(args...)).invalidate_local_storage(),0)...);

		auto cols = get<0>(args...).size_j();
		for(size_t row = 0; row < Vsize;)
		{
			// So much work for the task row count...
			size_t task_size =
				cont::getParent(get<0>(args...)).block_count_from(row * cols) / cols;
			auto handles =
				std::make_tuple(
					cont::getParent(get<OI>(args...)).handle_for_row(row)...,
					cont::getParent(get<VEI>(args...)).handle_for(row)...,
					cont::getParent(get<HEI>(args...)).local_storage_handle()...,
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(proxy_tags),
						row)...);

			skeleton_task::schedule(
				handles,
				vei,
				hei,
				row,
				task_size,
				Hsize,
				std::forward<decltype(get<CI>(args...))>(get<CI>(args...))...);

			row += task_size;
		}
	}
};

} // namespace backend
} // namespace skepu

#endif //  SKEPU_CLUSTER_SKELETONS_MAPPAIRS_HPP
