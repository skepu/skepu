#pragma once
#ifndef SKEPU_CLUSTER_SKELETONS_SKELETON_TASK_HPP
#define SKEPU_CLUSTER_SKELETONS_SKELETON_TASK_HPP 1

#include <tuple>

#include <starpu_mpi.h>

#include "skepu3/cluster/common.hpp"
#include "skepu3/cluster/containers/proxies.hpp"
#include "skeleton_base.hpp"
#include "skeleton_utils.hpp"
#include "task_helpers.hpp"

namespace skepu {
namespace cluster {

/** Encapsulating class for starpu tasks.
 * This class encapsulates some data access patterns common to all skeletons,
 * using the curiously recccuring template pattern. This places some
 * requirements on its subclasses.
 *
 * Each parameter is a tuple of the underlying datatype of each
 * desired buffer.
 *
 * \author Henrik Henriksson
 * \author Johan Ahlqvist
 *
 * \tparam ResultArgs			STARPU_W
 * \tparam ElwiseArgs			STARPU_R
 * \tparam ContainerArgs	STARPU_R
 * \tparam UniformArgs
 *
 */
template<
	typename CallBackFn,
	typename ResultArgs,
	typename ElwiseArgs,
	typename ContainerArgs,
	typename UniformArgs>
class skeleton_task
: public virtual backend::SkeletonBase
{
	starpu_codelet cl;

	static constexpr size_t n_result = std::tuple_size<ResultArgs>::value;
	static constexpr size_t n_elwise = std::tuple_size<ElwiseArgs>::value;
	static constexpr size_t n_container = std::tuple_size<ContainerArgs>::value;
	static constexpr size_t n_uniform = std::tuple_size<UniformArgs>::value;
	static constexpr size_t n_handles = n_result + n_elwise + n_container;


	static constexpr
			typename make_pack_indices<n_handles, 0>::type
		handle_indices{};
	static constexpr
			typename make_pack_indices<n_result, 0>::type
		result_handle_indices{};
	static constexpr
			typename make_pack_indices<n_result+n_elwise, n_result>::type
		elwise_handle_indices{};
	static constexpr
			typename make_pack_indices<
				n_result+n_elwise+n_container,n_result+n_elwise>::type
		container_handle_indices{};

	static constexpr typename make_pack_indices<n_result>::type ri{};
	static constexpr typename make_pack_indices<n_elwise>::type ei{};
	static constexpr typename make_pack_indices<n_container>::type ci{};
	static constexpr typename make_pack_indices<n_uniform>::type ui{};

protected:
	skeleton_task(char const * name)
	{
		starpu_codelet_init(&cl);
		cl.nbuffers = STARPU_VARIABLE_NBUFFERS;
		cl.max_parallelism = INT_MAX;
		cl.type = STARPU_FORKJOIN; // For OpenMP
		cl.cpu_funcs_name[0] = name;
	}

	template<typename Container, typename ProxyTag>
	auto
	container_handle(Container & c, ProxyTag, size_t)
	-> starpu_data_handle_t
	{
		return c.local_storage_handle();
	}

	template<typename Container>
	auto
	container_handle(Container & c, skepu::ProxyTag::MatRow const &, size_t row)
	-> starpu_data_handle_t
	{
		return c.handle_for_row(row);
	}

	template<typename Container, typename ProxyTag>
	auto
	handle_container_arg(Container & c, ProxyTag) noexcept
	-> void
	{
		c.allgather();
	}

	template<typename Container>
	auto
	handle_container_arg(Container & m, skepu::ProxyTag::MatRow)
	-> void
	{
		m.partition();
	}

	template<
		typename ... Handles,
		typename ... Args>
	auto
	schedule(
		std::tuple<Handles...> & handles,
		Args && ... args)
	-> void
	{
		cl.cpu_funcs[0] = starpu_task_callback<Args...>;

		#ifdef SKEPU_CUDA
		cl.cuda_funcs[0] = starpu_cu_callback<Args...>;
		#endif

		auto bs = SkeletonBase::m_bs;
		if(!bs)
			bs = &internalGlobalBackendSpecAccessor();

		if(bs)
		{
			using type = skepu::Backend::Type;
			switch(bs->getType())
			{
			#ifdef SKEPU_CUDA
			case type::CUDA:
				cl.where = STARPU_CUDA;
				break;
			#endif
			case type::OpenMP:
			case type::CPU:
				cl.where = STARPU_CPU;
				break;
			case type::Auto:
				cl.where = STARPU_CPU|STARPU_CUDA;
				break;
			default:
			{
				std::cerr << "[SkePU][skeleton_task] ERROR: "
					"Selected Backend::Type is not supported with StarPU-MPI.\n";
				std::abort();
			}};
		}
		else
		{
			cl.where = STARPU_CPU;

			#ifdef SKEPU_CUDA
			cl.where |= STARPU_CUDA;
			#endif
		}

		schedule(
			result_handle_indices,
			elwise_handle_indices,
			container_handle_indices,
			handles,
			std::forward<Args>(args)...);
	}

private:
	template<
		size_t ... OI,
		size_t ... EI,
		size_t ... CI,
		typename ... Handles,
		typename ... Args>
	auto
	schedule(
		pack_indices<OI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		std::tuple<Handles...> & handles,
		Args && ... args)
	-> void
	{

		auto starpu_args =
			std::tuple_cat(
				util::build_write_args(
					ri,
					std::tie(std::get<OI>(handles)...)),
				util::build_read_args(
					typename make_pack_indices<n_elwise + n_container>::type{},
					std::tie(
						std::get<EI>(handles)...,
						std::get<CI>(handles)...)),
				util::build_value_args(std::forward<Args>(args)...));
		typedef decltype(starpu_args) starpu_args_t;
		auto constexpr num_args = std::tuple_size<starpu_args_t>::value;

		schedule(
			typename make_pack_indices<num_args>::type{},
			starpu_args);
	}

	template<
		size_t ... I,
		typename ... StarPU_Args>
	auto
	schedule(
		pack_indices<I...>,
		std::tuple<StarPU_Args...> & args) noexcept
	-> void
	{
		starpu_mpi_task_insert(
			MPI_COMM_WORLD,
			&cl,
			std::get<I>(args)...,
			0);
	}

	template<typename ... Args>
	auto static
	starpu_task_callback(void ** buffers, void * args) noexcept
	-> void
	{
		auto static constexpr ai =
			typename make_pack_indices<sizeof...(Args)>::type{};

		handle_callback<Args...>(
			handle_indices,
			ri,
			ei,
			ci,
			ai,
			buffers,
			args);
	}

	template<
		typename ... Args,
		size_t ... HI,
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		size_t ... AI>
	auto static
	handle_callback(
		pack_indices<HI...>,
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		pack_indices<AI...>,
		void ** buffers,
		void * args_buffer) noexcept
	-> void
	{
		auto args = std::tuple<typename std::decay<Args>::type...>{};
		util::extract_value_args(
			args_buffer,
			std::get<AI>(args)...);

		// Since attribute maybe_unsed is not available until C++17, we use this
		// trick instead to get rid of the unused variable warnings for cbargs and
		// uniform_args.
		if(sizeof args)
			;

		typedef decltype(
				std::make_tuple(
					typename std::add_pointer<
							typename base_type<
									typename std::tuple_element<RI, ResultArgs>::type>
								::type>
						::type{}...,
					typename std::add_pointer<
							typename base_type<
									typename std::tuple_element<EI, ElwiseArgs>::type>
								::type>
						::type{}...,
					typename std::tuple_element<CI, ContainerArgs>::type{}...))
			buffers_type;

		buffers_type containers(
			prepare_buffer(
				(typename std::tuple_element<HI, buffers_type>::type *)0,
				buffers[HI])...);
		CallBackFn::run(
			result_handle_indices,
			elwise_handle_indices,
			container_handle_indices,
			containers,
			std::get<AI>(args)...);
	}

	#ifdef SKEPU_CUDA
	template<typename ... Args>
	auto static
	starpu_cu_callback(void ** buffers, void * args) noexcept
	-> void
	{
		auto static constexpr ai =
			typename make_pack_indices<sizeof...(Args)>::type{};

		handle_cu_callback<Args...>(
			handle_indices,
			ri,
			ei,
			ci,
			ai,
			buffers,
			args);
	}

	template<
		typename ... Args,
		size_t ... HI,
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		size_t ... AI>
	auto static
	handle_cu_callback(
		pack_indices<HI...>,
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		pack_indices<AI...>,
		void ** buffers,
		void * args_buffer) noexcept
	-> void
	{
		auto args = std::tuple<typename std::decay<Args>::type...>{};
		util::extract_value_args(
			args_buffer,
			std::get<AI>(args)...);

		// Since attribute maybe_unsed is not available until C++17, we use this
		// trick instead to get rid of the unused variable warnings for cbargs and
		// uniform_args.
		if(sizeof args)
			;

		typedef decltype(std::tuple_cat(
				std::make_tuple(
					typename std::add_pointer<
						decltype(std::get<RI>(ResultArgs{}))>::type{}...),
				std::make_tuple(
					typename std::add_pointer<
						decltype(std::get<EI>(ElwiseArgs{}))>::type{}...),
				ContainerArgs{}))
			buffers_type;

		buffers_type containers(
			prepare_buffer(
				(typename std::tuple_element<HI, buffers_type>::type *)0,
				buffers[HI])...);
		CallBackFn::CUDA(
			result_handle_indices,
			elwise_handle_indices,
			container_handle_indices,
			std::forward<buffers_type>(containers),
			std::get<AI>(args)...);
	}
	#endif
};

} // namespace cluster
} // namespace skepu

#endif // SKEPU_CLUSTER_SKELETONS_SKELETON_TASK_HPP
