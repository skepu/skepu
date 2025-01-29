#pragma once
#ifndef SKEPU_CLUSTER_SKELETONS_MAPPAIRSREDUCE_H
#define SKEPU_CLUSTER_SKELETONS_MAPPAIRSREDUCE_H 1

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>
#include "skepu3/cluster/skeletons/reduce/reduce_mode.hpp"
#include "skepu3/cluster/skeletons/skeleton_base.hpp"
#include "skepu3/cluster/skeletons/skeleton_task.hpp"
#include "skepu3/cluster/skeletons/skeleton_utils.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename MapFunc, typename ReduceFunc>
struct map_pairs_reduce_rowwise
{
	typedef
			ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>
		F;

	template<
		size_t ... RHI,
		size_t ... HI,
		size_t ... CI,
		size_t ... HEI,
		size_t ... VEI,
		typename Buffers,
		typename start_value_t,
		typename ... Args>
	auto static
	run(
		pack_indices<RHI...>,
		pack_indices<HI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		pack_indices<VEI...>,
		pack_indices<HEI...>,
		start_value_t start_value,
		size_t start_row,
		size_t task_rows,
		size_t task_cols,
		Args && ... args) noexcept
	-> void
	{
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t row = 0; row < task_rows; ++row)
		{
			pack_expand((
				std::get<RHI>(buffers)[row] = get_or_return<RHI>(start_value),0)...);
			for(size_t col = 0; col < task_cols; ++col)
			{
				auto map_res =
					F::forward(
						MapFunc::OMP, Index2D{start_row + row, col},
						std::get<VEI>(buffers)[row]...,
						std::get<HEI>(buffers)[col]...,
						cluster::advance(std::get<CI>(buffers), row)...,
						args...);
				pack_expand(
					(std::get<RHI>(buffers)[row] =
						ReduceFunc::OMP(
							std::get<RHI>(buffers)[row],
							get_or_return<RHI>(map_res)))...);
			}
		}
	}
};

template<typename MapFunc, typename ReduceFunc>
struct map_pairs_reduce_colwise
{
	typedef
			ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>
		F;

	template<
		size_t ... RHI,
		size_t ... HI,
		size_t ... CI,
		size_t ... HEI,
		size_t ... VEI,
		typename Buffers,
		typename start_value_t,
		typename ... Args>
	auto static
	run(
		pack_indices<RHI...>,
		pack_indices<HI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		pack_indices<VEI...>,
		pack_indices<HEI...>,
		start_value_t start_value,
		size_t start_col,
		size_t task_cols,
		size_t task_rows,
		Args && ... args) noexcept
	-> void
	{
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t col = 0; col < task_cols; ++col)
		{
			pack_expand((
				std::get<RHI>(buffers)[col] = get_or_return<RHI>(start_value),0)...);
			for(size_t row = 0; row < task_rows; ++row)
			{
				auto map_res =
					F::forward(
						MapFunc::OMP, Index2D{row, start_col + col},
						std::get<VEI>(buffers)[row]...,
						std::get<HEI>(buffers)[col]...,
						cluster::advance(std::get<CI>(buffers), row)...,
						args...);
				pack_expand(
					((std::get<RHI>(buffers)[col] =
						ReduceFunc::OMP(
							std::get<RHI>(buffers)[col],
							get_or_return<RHI>(map_res))),0)...);
			}
		}
	}
};

} // namespace _starpu

template<
		size_t Varity,
		size_t Harity,
		typename MapFunc,
		typename ReduceFunc,
		typename CUDAKernel,
		typename CLKernel>
class MapPairsReduce
: private cluster::skeleton_task<
		_starpu::map_pairs_reduce_rowwise<MapFunc, ReduceFunc>,
		typename cluster::result_tuple<typename MapFunc::Ret>::type,
		typename MapFunc::ElwiseArgs,
		typename MapFunc::ContainerArgs,
		typename MapFunc::UniformArgs>,
	private cluster::skeleton_task<
		_starpu::map_pairs_reduce_colwise<MapFunc, ReduceFunc>,
		typename cluster::result_tuple<typename MapFunc::Ret>::type,
		typename MapFunc::ElwiseArgs,
		typename MapFunc::ContainerArgs,
		typename MapFunc::UniformArgs>
{
	typedef typename MapFunc::Ret Ret;
	typedef std::tuple<> ResultArg;
	typedef typename MapFunc::ElwiseArgs ElwiseArgs;
	typedef typename MapFunc::ContainerArgs ContainerArgs;
	typedef typename MapFunc::UniformArgs UniformArgs;
	static constexpr bool prefers_matrix = MapFunc::prefersMatrix;
	typedef cluster::skeleton_task<
			_starpu::map_pairs_reduce_rowwise<MapFunc, ReduceFunc>,
			typename cluster::result_tuple<Ret>::type,
			ElwiseArgs,
			ContainerArgs,
			UniformArgs>
		rowwise_task;
	typedef cluster::skeleton_task<
			_starpu::map_pairs_reduce_colwise<MapFunc, ReduceFunc>,
			typename cluster::result_tuple<typename MapFunc::Ret>::type,
			ElwiseArgs,
			ContainerArgs,
			UniformArgs>
		colwise_task;

	static constexpr size_t out_arity = MapFunc::outArity;
	static constexpr size_t numArgs =
		MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
	static constexpr size_t anyArity =
		std::tuple_size<typename MapFunc::ContainerArgs>::value;

	auto static constexpr out_indices =
		typename make_pack_indices<out_arity, 0>::type{};
	auto static constexpr Velwise_indices =
		typename make_pack_indices<out_arity + Varity, out_arity>::type{};
	auto static constexpr Helwise_indices =
		typename make_pack_indices<out_arity + Harity + Varity, out_arity + Varity>
			::type{};
	auto static constexpr elwise_indices =
		typename make_pack_indices<out_arity + Harity + Varity, out_arity>::type{};
	auto static constexpr any_indices =
		typename make_pack_indices<
				out_arity + Varity + Harity + anyArity, out_arity + Varity + Harity>
			::type{};
	auto static constexpr const_indices =
		typename make_pack_indices<
				out_arity + numArgs,
				out_arity + Varity + Harity + anyArity>
			::type{};

	auto static constexpr proxy_tag_indices =
		typename make_pack_indices<
			std::tuple_size<typename MapFunc::ProxyTags>::value>::type{};
	typename MapFunc::ProxyTags proxy_tags{};

	typedef typename parameter_type<MapFunc::indexed
			? 1
			: 0, decltype(&MapFunc::CPU)>::type
		First;

	size_t default_size_i;
	size_t default_size_j;

	ReduceMode m_mode;
	Ret m_start;

public:
	static constexpr auto skeletonType = SkeletonType::MapPairsReduce;

	MapPairsReduce(CUDAKernel)
	: rowwise_task("MapPairsReduce RowWise"),
		colwise_task("MapPairsReduce ColWise"),
		default_size_i(0),
		default_size_j(0),
		m_mode(ReduceMode::RowWise),
		m_start(Ret())
	{}

	~MapPairsReduce() noexcept
	{
		skepu::cluster::barrier();
	}

	void setStartValue(Ret val)
	{
		this->m_start = val;
	}

	void setDefaultSize(size_t i, size_t j = 0)
	{
		this->default_size_i = i;
		this->default_size_j = j;
	}

	auto
	setReduceMode(ReduceMode mode) noexcept
	-> void
	{
		m_mode = mode;
	}

	template<typename... CallArgs>
	auto
	operator()(CallArgs &&... args) noexcept
	-> typename std::add_lvalue_reference<decltype(get<0>(args...))>::type
	{
		switch(m_mode)
		{
		case ReduceMode::RowWise:
			rowwise_dispatch(
				out_indices,
				Velwise_indices,
				Helwise_indices,
				any_indices,
				const_indices,
				proxy_tag_indices,
				std::forward<CallArgs>(args)...);
			break;
		case ReduceMode::ColWise:
			colwise_dispatch(
				out_indices,
				Velwise_indices,
				Helwise_indices,
				any_indices,
				const_indices,
				proxy_tag_indices,
				std::forward<CallArgs>(args)...);
			break;
		default:
			SKEPU_ERROR("Reduce mode not supported.");
		}
		return get<0>(args...);
	}

private:
	template<
		size_t ... OI,
		size_t ... VEI,
		size_t ... HEI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename ... CallArgs>
	void rowwise_dispatch(
		pack_indices<OI...>,
		pack_indices<VEI...> vei,
		pack_indices<HEI...> hei,
		pack_indices<AI...>,
		pack_indices<CI...>,
		pack_indices<PI...>,
		CallArgs &&... args) noexcept
	{
		size_t Vsize = get<0>(get<VEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_i);
		size_t Hsize = get<0>(get<HEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_j);

		if(disjunction((get<OI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
			SKEPU_ERROR("Non-matching output container size");

		if(disjunction((get<VEI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
			SKEPU_ERROR("Non-matching container sizes");

		if(disjunction((get<HEI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
			SKEPU_ERROR("Non-matching container sizes");

		// Filters are not supported here, make sure containers are not filtered...
		pack_expand(
			(cont::getParent(get<OI>(args...)).filter(0),0)...,
			(cont::getParent(get<VEI>(args...)).filter(0),0)...,
			(cont::getParent(get<HEI>(args...)).filter(0),0)...,
			(cont::getParent(get<AI>(args...)).filter(0),0)...);

		pack_expand(
			(rowwise_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(proxy_tags)),0)...,
			(cont::getParent(get<OI>(args...)).partition(),0)...,
			(cont::getParent(get<VEI>(args...)).partition(),0)...,
			(cont::getParent(get<HEI>(args...)).allgather(),0)...,
			(cont::getParent(get<OI>(args...)).invalidate_local_storage(),0)...);

		for(size_t row = 0; row < Vsize;)
		{
			size_t task_size =
				cont::getParent(get<0>(args...)).block_count_from(row);
			auto handles =
				std::make_tuple(
					cont::getParent(get<OI>(args...)).handle_for(row)...,
					cont::getParent(get<VEI>(args...)).handle_for(row)...,
					cont::getParent(get<HEI>(args...)).local_storage_handle()...,
					rowwise_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(proxy_tags),
						row)...);

			rowwise_task::schedule(
				handles,
				vei,
				hei,
				m_start,
				row,
				task_size,
				Hsize,
				std::forward<decltype(get<CI>(args...))>(get<CI>(args...))...);

			row += task_size;
		}
	}

	template<
		size_t ... OI,
		size_t ... VEI,
		size_t ... HEI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename ... CallArgs>
	void colwise_dispatch(
		pack_indices<OI...>,
		pack_indices<VEI...> vei,
		pack_indices<HEI...> hei,
		pack_indices<AI...>,
		pack_indices<CI...>,
		pack_indices<PI...>,
		CallArgs &&... args) noexcept
	{
		size_t Vsize = get<0>(get<VEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_i);
		size_t Hsize = get<0>(get<HEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_j);

		if(disjunction((get<OI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
			SKEPU_ERROR("Non-matching output container size");

		if(disjunction((get<VEI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
			SKEPU_ERROR("Non-matching container sizes");

		if(disjunction((get<HEI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
			SKEPU_ERROR("Non-matching container sizes");

		pack_expand(
			// Filters are not supported here yet
			(cont::getParent(get<OI>(args...)).filter(0),0)...,
			(cont::getParent(get<VEI>(args...)).filter(0),0)...,
			(cont::getParent(get<HEI>(args...)).filter(0),0)...,
			(cont::getParent(get<AI>(args...)).filter(0),0)...,

			/* Might need a version for MatCol that does not allgather in this case.
			(colwise_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(proxy_tags)),0)...,
			 */
			(cont::getParent(get<AI>(args...)).allgather(),0)...,
			(cont::getParent(get<OI>(args...)).partition(),0)...,
			(cont::getParent(get<VEI>(args...)).allgather(),0)...,
			(cont::getParent(get<HEI>(args...)).partition(),0)...,
			(cont::getParent(get<OI>(args...)).invalidate_local_storage(),0)...);

		for(size_t col = 0; col < Hsize;)
		{
			size_t task_size =
				cont::getParent(get<0>(args...)).block_count_from(col);
			auto handles =
				std::make_tuple(
					cont::getParent(get<OI>(args...)).handle_for(col)...,
					cont::getParent(get<VEI>(args...)).local_storage_handle()...,
					cont::getParent(get<HEI>(args...)).handle_for(col)...,
					colwise_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(proxy_tags),
						col)...);

			auto call_back_args =
				std::make_tuple(
					vei,
					hei,
					m_start,
					col,
					task_size,
					Vsize);

			colwise_task::schedule(
				handles,
				vei,
				hei,
				m_start,
				col,
				task_size,
				Vsize,
				std::forward<decltype(get<CI>(args...))>(get<CI>(args...))...);

			col += task_size;
		}
	}
};

} // end namespace backend
} // end namespace skepu

#endif // SKEPU_CLUSTER_SKELETONS_MAPPAIRSREDUCE_H
