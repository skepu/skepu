#pragma once
#ifndef SKEPU_CLUSTER_EXTERNAL_HPP
#define SKEPU_CLUSTER_EXTERNAL_HPP 1

#include "cluster.hpp"
#include "common.hpp"

namespace skepu {
namespace label {

struct read {};
struct write {};

} // namespace label

template<
	typename ... Containers,
	REQUIRES_VALUE(are_skepu_containers<Containers...>)>
auto inline constexpr
read(Containers && ... cs) noexcept
-> std::tuple<label::read, Containers...>
{
	return std::tuple<label::read, Containers...>(label::read{}, cs...);
}

template<
	typename ... Containers,
	REQUIRES_VALUE(are_skepu_containers<Containers...>)>
auto inline constexpr
write(
	Containers && ... cs) noexcept
-> std::tuple<label::write, Containers...>
{
	return std::tuple<label::write, Containers...>(label::write{}, cs...);
}

template<
	size_t ... CI,
	typename ... Containers>
auto inline
gather_to_root(
	pack_indices<CI...>,
	std::tuple<Containers...> & cs) noexcept
-> void
{
	pack_expand((cont::getParent(std::get<CI>(cs)).gather_to_root(),0)...);
}

template<
	size_t ... CI,
	typename ... Containers>
auto inline
make_writeable(
	pack_indices<CI...>,
	std::tuple<Containers...> & cs) noexcept
-> void
{
	pack_expand((cont::getParent(std::get<CI>(cs)).make_ext_w(),0)...);
}

template<
	size_t ... CI,
	typename ... Containers>
auto inline
scatter_from_root(
	pack_indices<CI...>,
	std::tuple<Containers...> & cs)
-> void
{
	pack_expand((cont::getParent(std::get<CI>(cs)).scatter_from_root(),0)...);
}

template<typename OP>
auto inline
external(OP && op) noexcept(noexcept(op))
-> void
{
	if(!cluster::mpi_rank())
		op();
}

template<
	typename ... ContR,
	typename OP>
auto inline
external(
	std::tuple<label::read, ContR...> cr,
	OP && op) noexcept(noexcept(op))
-> void
{
	auto static constexpr read_indices =
		typename make_pack_indices<sizeof...(ContR) +1, 1>::type{};

	gather_to_root(read_indices, cr);

	if(!cluster::mpi_rank())
		op();
}

template<
	typename OP,
	typename ... ContW>
auto inline
external(
	OP && op,
	std::tuple<label::write, ContW...> cw) noexcept(noexcept(op))
-> void
{
	auto static constexpr write_indices =
		typename make_pack_indices<sizeof...(ContW) +1, 1>::type{};

	make_writeable(write_indices, cw);

	if(!cluster::mpi_rank())
		op();

	scatter_from_root(write_indices, cw);
}

template<
	typename ... ContR,
	typename OP,
	typename ... ContW>
auto inline
external(
	std::tuple<label::read, ContR...> cr,
	OP && op,
	std::tuple<label::write, ContW...> cw) noexcept(noexcept(op))
-> void
{
	auto static constexpr read_indices =
		typename make_pack_indices<sizeof...(ContR) +1, 1>::type{};
	auto static constexpr write_indices =
		typename make_pack_indices<sizeof...(ContW) +1, 1>::type{};

	gather_to_root(read_indices, cr);
	make_writeable(write_indices, cw);

	if(!cluster::mpi_rank())
		op();

	scatter_from_root(write_indices, cw);
}

} // namespace skepu

#endif // SKEPU_CLUSTER_EXTERNAL_HPP
