#pragma once
#ifndef SKEPU_CLUSTER_TASK_HELPERS_HPP
#define SKEPU_CLUSTER_TASK_HELPERS_HPP 1

#include <skepu3/cluster/common.hpp>

#include <starpu.h>

namespace skepu {
namespace util {

auto inline
build_read_arg(starpu_data_handle_t handle) noexcept
-> decltype(std::make_tuple(STARPU_R, handle))
{
	return std::make_tuple(STARPU_R, handle);
};

template<
	size_t ... I,
	typename ... Handles>
auto inline
build_read_args(
	pack_indices<I...>,
	std::tuple<Handles...> const & handles) noexcept
-> decltype(
		std::tuple_cat(
			build_read_arg(std::get<I>(handles))...))
{
	return std::tuple_cat(build_read_arg(std::get<I>(handles))...);
}

auto inline
build_write_arg(starpu_data_handle_t handle) noexcept
-> decltype(std::make_tuple(STARPU_W, handle))
{
	return std::make_tuple(STARPU_W, handle);
};

template<
	size_t ... I,
	typename ... Handles>
auto inline
build_write_args(
	pack_indices<I...>,
	std::tuple<Handles...> const & handles) noexcept
-> decltype(
		std::tuple_cat(
			build_write_arg(std::get<I>(handles))...))
{
	return std::tuple_cat(build_write_arg(std::get<I>(handles))...);
}

template<typename T>
auto inline
build_value_arg(T && value) noexcept
-> decltype(std::make_tuple(STARPU_VALUE, &value, sizeof(T)))
{
	return std::make_tuple(STARPU_VALUE, &value, sizeof(T));
};

template<typename ... Args>
auto inline
build_value_args(
	Args &&... args) noexcept
-> decltype(
		std::tuple_cat(
			build_value_arg(args)...))
{
	return std::tuple_cat(build_value_arg(args)...);
}

template<typename ... Args>
auto inline
extract_value_args(
	void * args_buffer,
	Args &&... args) noexcept
-> void
{
	starpu_codelet_unpack_args(
		args_buffer,
		&(args)...,
		0);
}

} // namespace util
} // namespace skepu

#endif // SKEPU_CLUSTER_TASK_HELPERS_HPP
