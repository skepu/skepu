#pragma once

#include <tuple>

#include "skepu3/mpi/vector.hpp"

namespace skepu
{
	// Code that cannot be run in parallel, for example with files or with data in a structure of the program. It runs on
	// the first process only (the other processes skip it), with the containers that it reads brought to that process
	// and the containers that it writes given back to their owners afterwards. A container that the code uses must be
	// listed: in skepu::read(...) if the code only reads it, in skepu::write(...) if it changes it.

	template<typename... Containers>
	auto inline read(Containers &&... cs) noexcept -> std::tuple<label::read, Containers...>
	{
		return std::tuple<label::read, Containers...>(label::read{}, cs...);
	}

	template<typename... Containers>
	auto inline write(Containers &&... cs) noexcept -> std::tuple<label::write, Containers...>
	{
		return std::tuple<label::write, Containers...>(label::write{}, cs...);
	}

	namespace mpi
	{
		template<typename Tuple, size_t... I>
		auto external_begin(Tuple &cs, std::index_sequence<I...>)
		{
			return std::make_tuple(std::get<I + 1>(cs).external_begin()...);
		}

		template<typename Tuple, typename States, size_t... I>
		void external_end_read(Tuple &cs, States &states, std::index_sequence<I...>)
		{
			(std::get<I + 1>(cs).external_end_read(std::get<I>(states)), ...);
		}

		template<typename Tuple, size_t... I>
		void external_end_write(Tuple &cs, std::index_sequence<I...>)
		{
			(std::get<I + 1>(cs).external_end_write(), ...);
		}

		template<typename OP>
		void run_on_root(OP &&op)
		{
			if (mpi::rank() == 0)
			{
				++external_depth();
				op();
				--external_depth();
			}
		}
	}

	template<typename OP>
	void inline external(OP &&op)
	{
		mpi::run_on_root(op);
	}

	template<typename... ContR, typename OP>
	void inline external(std::tuple<label::read, ContR...> cr, OP &&op)
	{
		constexpr auto indices = std::index_sequence_for<ContR...>{};
		auto states = mpi::external_begin(cr, indices);
		mpi::run_on_root(op);
		mpi::external_end_read(cr, states, indices);
	}

	template<typename OP, typename... ContW>
	void inline external(OP &&op, std::tuple<label::write, ContW...> cw)
	{
		constexpr auto indices = std::index_sequence_for<ContW...>{};
		mpi::external_begin(cw, indices);
		mpi::run_on_root(op);
		mpi::external_end_write(cw, indices);
	}

	template<typename... ContR, typename OP, typename... ContW>
	void inline external(std::tuple<label::read, ContR...> cr, OP &&op, std::tuple<label::write, ContW...> cw)
	{
		constexpr auto read_indices = std::index_sequence_for<ContR...>{};
		constexpr auto write_indices = std::index_sequence_for<ContW...>{};
		auto states = mpi::external_begin(cr, read_indices);
		mpi::external_begin(cw, write_indices);
		mpi::run_on_root(op);
		mpi::external_end_read(cr, states, read_indices);
		mpi::external_end_write(cw, write_indices);
	}

	// With a description of the code (for the trace of a single node: not used here)

	template<typename OP>
	void inline external(std::string /*description*/, OP &&op SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		external(std::forward<OP>(op));
	}

	template<typename... ContR, typename OP>
	void inline external(std::string /*description*/, std::tuple<label::read, ContR...> cr, OP &&op SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		external(cr, std::forward<OP>(op));
	}

	template<typename OP, typename... ContW>
	void inline external(std::string /*description*/, OP &&op, std::tuple<label::write, ContW...> cw SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		external(std::forward<OP>(op), cw);
	}

	template<typename... ContR, typename OP, typename... ContW>
	void inline external(std::string /*description*/, std::tuple<label::read, ContR...> cr, OP &&op, std::tuple<label::write, ContW...> cw SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		external(cr, std::forward<OP>(op), cw);
	}
}
