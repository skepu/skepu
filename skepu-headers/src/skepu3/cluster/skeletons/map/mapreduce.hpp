#pragma once
#ifndef SKEPU_CLUSTER_MAPREDUCE_HPP
#define SKEPU_CLUSTER_MAPREDUCE_HPP 1

#include <algorithm>
#include <omp.h>
#include <set>

#include <skepu3/cluster/common.hpp>
#include <skepu3/cluster/cluster.hpp>
#include "skepu3/cluster/skeletons/skeleton_base.hpp"
#include "skepu3/cluster/skeletons/skeleton_task.hpp"
#include "skepu3/cluster/skeletons/skeleton_utils.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename MapFunc, typename ReduceFunc>
struct map_reduce
{
	typedef ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>
		F;
	typedef typename MapFunc::Ret T;
	typedef index_dimension<
			typename std::conditional<MapFunc::indexed, typename MapFunc::IndexType,
			skepu::Index1D>::type>
		defaultDim;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		size_t ... OI,
		typename Buffers,
		typename Iterator,
		typename ... CallArgs>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		pack_indices<OI...>,
		Iterator begin,
		size_t count,
		CallArgs && ... args) noexcept
	-> void
	{
		auto threads = std::min<size_t>(count, starpu_combined_worker_get_size());
		omp_set_num_threads(threads);

		std::vector<T> res_v(threads);
		#pragma omp parallel
		{
			size_t tid = omp_get_thread_num();
			res_v[tid] =
				F::forward(
					MapFunc::OMP,
					(begin + tid).index(),
					// Elwise elements are also raw pointers.
					std::get<EI>(buffers)[tid]...,
					// Set MatRow to correct position. Does nothing for others.
					cluster::advance(std::get<CI>(buffers), tid)...,
					args...);
			#pragma omp for
			for(size_t i = threads; i < count; ++i)
			{
				auto res =
					F::forward(
						MapFunc::OMP,
						(begin +i).index(),
						// Elwise elements are also raw pointers.
						std::get<EI>(buffers)[i]...,
						// Set MatRow to correct position. Does nothing for others.
						cluster::advance(std::get<CI>(buffers),i)...,
						args...);
				pack_expand((
					get_or_return<OI>(res_v[tid]) =
						ReduceFunc::OMP(
							get_or_return<OI>(res_v[tid]),
							get_or_return<OI>(res)), 0)...);
			}
		}

		auto & res = std::get<0>(buffers)[0];
		res = res_v[0];
		for(size_t i(1); i < threads; ++i)
			pack_expand((
				get_or_return<OI>(res) =
					ReduceFunc::OMP(
						get_or_return<OI>(res),
						get_or_return<OI>(res_v[i])), 0)...);
	}

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		size_t ... OI,
		typename Buffers,
		typename ... CallArgs>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		pack_indices<OI...>,
		size_t offset,
		size_t count,
		size_t size_j,
		size_t size_k,
		size_t size_l,
		CallArgs && ... args) noexcept
	-> void
	{
		auto threads =
			std::min<size_t>(count, starpu_combined_worker_get_size());
		omp_set_num_threads(threads);

		std::vector<T> res_v(threads);
		#pragma omp parallel
		{
			size_t tid = omp_get_thread_num();
			//auto res =
			res_v[tid] =
				F::forward(
					MapFunc::OMP,
					make_index(defaultDim{}, offset + tid, size_j, size_k, size_l),
					// Set MatRow to correct position. Does nothing for others.
					cluster::advance(std::get<CI>(buffers), tid)...,
					args...);

			#pragma omp for
			for(size_t i = threads; i < count; ++i)
			{
				auto res =
						F::forward(
							MapFunc::OMP,
							make_index(defaultDim{}, offset +i, size_j, size_k, size_l),
							// Set MatRow to correct position. Does nothing for others.
							cluster::advance(std::get<CI>(buffers), i)...,
							args...);
				pack_expand((
					get_or_return<OI>(res_v[tid]) =
						ReduceFunc::OMP(
							get_or_return<OI>(res_v[tid]),
							get_or_return<OI>(res)), 0)...);
			}
		}

		auto & res = std::get<0>(buffers)[0];
		res = res_v[0];
		for(size_t i(1); i < threads; ++i)
			pack_expand((
				get_or_return<OI>(res) =
					ReduceFunc::OMP(
						get_or_return<OI>(res),
						get_or_return<OI>(res_v[i])), 0)...);
	}
};

} // namespace _starpu

template<
	size_t arity,
	typename MapFunc,
	typename ReduceFunc,
	typename CUDAKernel,
	typename CUDAReduceKernel,
	typename CLKernel>
class MapReduce
: public cluster::skeleton_task<
		_starpu::map_reduce<MapFunc, ReduceFunc>,
		std::tuple<typename MapFunc::Ret>,
		typename MapFunc::ElwiseArgs,
		typename MapFunc::ContainerArgs,
		typename MapFunc::UniformArgs>
{
	static constexpr auto skeletonType = SkeletonType::MapReduce;
	typedef typename MapFunc::Ret Ret;

	typedef cluster::skeleton_task<
			_starpu::map_reduce<MapFunc, ReduceFunc>,
			std::tuple<typename MapFunc::Ret>,
			typename MapFunc::ElwiseArgs,
			typename MapFunc::ContainerArgs,
			typename MapFunc::UniformArgs>
		skeleton_task;

	static constexpr size_t numArgs =
		MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
	static constexpr size_t anyArity =
		std::tuple_size<typename MapFunc::ContainerArgs>::value;

	size_t constexpr static nresult = MapFunc::outArity;
	size_t constexpr static nelwise =
		std::tuple_size<typename MapFunc::ElwiseArgs>::value;
	size_t constexpr static ncontainer =
		std::tuple_size<typename MapFunc::ContainerArgs>::value;
	size_t constexpr static nuniform =
		std::tuple_size<typename MapFunc::UniformArgs>::value;

	static constexpr typename make_pack_indices<MapFunc::outArity>::type
		out_indices{};
	static constexpr typename make_pack_indices<arity, 0>::type
		elwise_indices{};
	static constexpr typename
			make_pack_indices<arity + anyArity, arity>::type
		any_indices{};
	static constexpr typename
			make_pack_indices<numArgs, arity + anyArity>::type
		const_indices{};
	auto static constexpr ptag_indices =
		typename make_pack_indices<anyArity>::type{};

	typedef index_dimension<
			typename std::conditional<MapFunc::indexed, typename MapFunc::IndexType,
			skepu::Index1D>::type>
		defaultDim;

	size_t default_size_i;
	size_t default_size_j;
	size_t default_size_k;
	size_t default_size_l;
	Ret m_start{};
	Ret m_result;
	std::vector<starpu_data_handle_t> m_result_handles;

public:
	MapReduce(CUDAKernel, CUDAReduceKernel)
	: skeleton_task("Skepu MapReduce"), m_result_handles(cluster::mpi_size())
	{
		auto rank = cluster::mpi_rank();
		for(size_t i(0); i < cluster::mpi_size(); ++i)
		{
			auto & handle = m_result_handles[i];
			int home_node = -1;
			Ret * ptr = 0;
			if(i == rank)
			{
				home_node = STARPU_MAIN_RAM;
				ptr = &m_result;
			}

			starpu_variable_data_register(
				&handle, home_node, (uintptr_t)ptr, sizeof(Ret));
			starpu_mpi_data_register(
				handle, cluster::mpi_tag(), i);
		}
	}

	~MapReduce() noexcept
	{
		for(auto & handle : m_result_handles)
			starpu_data_unregister_no_coherency(handle);
		skepu::cluster::barrier();
	}

	void setStartValue(Ret val)
	{
		m_start = val;
	}

	void setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
	{
		default_size_i = i;
		default_size_j = j;
		default_size_k = k;
		default_size_l = l;
	}

	template<
		typename T,
		template<class> class Container,
		typename ... CallArgs,
		REQUIRES(is_skepu_container<Container<T>>::value)>
	auto
	operator()(Container<T> & arg1, CallArgs && ... args) noexcept
	-> Ret
	{
		static_assert(sizeof...(CallArgs) == numArgs -1,
			"Number of arguments not matching Map function");

		return
			backendDispatch(
				elwise_indices,
				any_indices,
				const_indices,
				arg1.begin(),
				arg1.end(),
				arg1,
				std::forward<CallArgs>(args)...);
	}

	template<
		typename T,
		template<typename>class Iterator,
		typename ... CallArgs,
		REQUIRES(is_skepu_iterator<Iterator<T>, T>::value)>
	auto
	operator()(Iterator<T> begin, Iterator<T> end, CallArgs && ... args) noexcept
	-> Ret
	{
		static_assert(sizeof...(CallArgs) == numArgs -1,
			"Number of arguments not matching Map function");

		return
			backendDispatch(
				elwise_indices,
				any_indices,
				const_indices,
				begin,
				end,
				begin,
				std::forward<CallArgs>(args)...);
	}

	template<typename ... CallArgs>
	auto
	operator()(CallArgs && ... args) noexcept
	-> Ret
	{
		static_assert(sizeof...(CallArgs) == numArgs,
			"Number of arguments not matching Map function");

		size_t size = default_size_i;
		if(defaultDim::value >= 2)
			size *= default_size_j;
		if(defaultDim::value >= 3)
			size *= default_size_k;
		if(defaultDim::value >= 4)
			size *= default_size_l;

		return
			backendDispatch(
				any_indices,
				const_indices,
				size,
				std::forward<CallArgs>(args)...);
	}

private:
	template<
		size_t ... EI,
		size_t ... AI,
		size_t ... CI,
		typename Iterator,
		typename ... CallArgs>
	auto
	backendDispatch(
		pack_indices<EI...> ei,
		pack_indices<AI...> ai,
		pack_indices<CI...> ci,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> Ret
	{
		if(disjunction((
				cont::getParent(get<EI>(args...)).size() < (end - begin))...))
			SKEPU_ERROR("Non-matching container sizes");

		return
			STARPU(
				out_indices,
				ei,
				ai,
				ci,
				ptag_indices,
				begin,
				end,
				std::forward<CallArgs>(args)...);
	}

	template<
		size_t ... AI,
		size_t ... CI,
		typename ... CallArgs>
	auto
	backendDispatch(
		pack_indices<AI...> ai,
		pack_indices<CI...> ci,
		size_t size,
		CallArgs && ... args) noexcept
	-> Ret
	{
		return
			STARPU(
				out_indices,
				ai,
				ci,
				ptag_indices,
				size,
				std::forward<CallArgs>(args)...);
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
	STARPU(
		pack_indices<OI...> oi,
		pack_indices<EI...>,
		pack_indices<AI...>,
		pack_indices<CI...>,
		pack_indices<PI...>,
		Iterator begin,
		Iterator end,
		CallArgs && ... args) noexcept
	-> Ret
	{
		auto static constexpr pt = typename MapFunc::ProxyTags{};

		pack_expand(
			// Overpartiniing not supported yet.
			(cont::getParent(get<EI>(args...)).filter(0),0)...,
			(cont::getParent(get<AI>(args...)).filter(0),0)...,

			(skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(pt)),0)...,
			(cont::getParent(get<EI>(args...)).partition(),0)...);

		std::set<size_t> scheduled_ranks;
		while(begin != end)
		{
			auto pos = begin.offset();
			auto task_size = std::min({
				(size_t)(end - begin),
				cont::getParent(get<EI>(args...)).block_count_from(pos)...});
			auto rank = starpu_mpi_data_get_rank(begin.getParent().handle_for(pos));
			auto handles =
				std::make_tuple(
					m_result_handles[rank],
					cont::getParent(get<EI>(args...)).handle_for(pos)...,
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(pt),
						pos)...);

			skeleton_task::schedule(
				handles,
				oi,
				begin,
				task_size,
				std::forward<decltype(get<CI>(args...))>(get<CI>(args...))...);

			scheduled_ranks.insert(rank);
			begin += task_size;
		}

		Ret res = m_start;
		for(auto rank : scheduled_ranks)
		{
			auto & handle = m_result_handles[rank];
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
			starpu_data_acquire(handle, STARPU_RW);
			pack_expand((
				get_or_return<OI>(res) =
					ReduceFunc::CPU(
						get_or_return<OI>(res),
						get_or_return<OI>(
							*(Ret *)starpu_data_get_local_ptr(handle))),0)...);
			starpu_data_release(handle);
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
		}

		return res;
	}

	template<
		size_t ... OI,
		size_t ... AI,
		size_t ... CI,
		size_t ... PI,
		typename ... CallArgs>
	auto
	STARPU(
		pack_indices<OI...> oi,
		pack_indices<AI...>,
		pack_indices<CI...>,
		pack_indices<PI...>,
		size_t size,
		CallArgs && ... args) noexcept
	-> Ret
	{
		auto static constexpr pt = typename MapFunc::ProxyTags{};
		pack_expand(
			(cont::getParent(get<AI>(args...)).filter(0), 0)...,

			(skeleton_task::handle_container_arg(
				cont::getParent(get<AI>(args...)),
				std::get<PI>(pt)),0)...);
		size_t task_size = size / cluster::mpi_size();
		if(size - (task_size * cluster::mpi_size()))
			++task_size;
		size_t pos = 0;
		size_t rank = 0;

		for(; pos < size; pos += task_size, ++rank)
		{
			auto handles =
				std::make_tuple(
					m_result_handles[rank],
					skeleton_task::container_handle(
						cont::getParent(get<AI>(args...)),
						std::get<PI>(pt),
						pos)...);
			size_t count = std::min(task_size, size - (rank * task_size));
			auto call_back_args =
				std::make_tuple(
					oi,
					pos,
					count,
					default_size_j,
					default_size_k,
					default_size_l);

			skeleton_task::schedule(
				handles,
				oi,
				pos,
				count,
				default_size_j,
				default_size_k,
				default_size_l,
				std::forward<decltype(get<CI>(args...))>(get<CI>(args...))...);
		}

		Ret res = m_start;
		for(rank = 0; rank * task_size < size; ++rank)
		{
			auto handle = m_result_handles[rank];
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
			starpu_data_acquire(handle, STARPU_RW);
			Ret part_res = *(Ret *)starpu_data_get_local_ptr(handle);
			pack_expand((
				get_or_return<OI>(res) =
					ReduceFunc::CPU(
						get_or_return<OI>(res),
						get_or_return<OI>(part_res)),0)...);
			starpu_data_release(handle);
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
		}

		return res;
	}
};

} // namespace backend
} // namespace skepu

#endif// SKEPU_CLUSTER_MAPREDUCE_HPP
