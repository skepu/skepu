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

		template<typename Tuple, size_t... I>
		std::vector<int> ids_of(Tuple &cs, std::index_sequence<I...>)
		{
			return std::vector<int>{std::get<I + 1>(cs).getObjectID()...};
		}

		// What all the forms of external do: the containers that are read are brought to the first process, which runs the code, the
		// containers that are written are given back to their owners, and the call is recorded in the trace (as a call, with its
		// number: the processes make the same calls in the same order; the gather is before the code and the scatter after it).
		template<typename... ContR, typename OP, typename... ContW>
		void external_core(const std::string &label, const char *file, int line, std::tuple<label::read, ContR...> &cr, OP &&op, std::tuple<label::write, ContW...> &cw)
		{
			constexpr auto read_indices = std::index_sequence_for<ContR...>{};
			constexpr auto write_indices = std::index_sequence_for<ContW...>{};
			trace::CallScope scope("external");
			auto states = mpi::external_begin(cr, read_indices);
			mpi::external_begin(cw, write_indices);
			if (mpi::rank() == 0)
				scope.work_begin();
			mpi::run_on_root(op);
			if (mpi::rank() == 0)
				scope.work_finish();
			scope.work_finished();
			mpi::external_end_read(cr, states, read_indices);
			mpi::external_end_write(cw, write_indices);
			scope.finish_external(label, file, line, mpi::ids_of(cw, write_indices), mpi::ids_of(cr, read_indices));
		}
	}

	// (the single node library does not record the call of an external that has no label and no container)
	template<typename OP>
	void inline external(OP &&op)
	{
		mpi::run_on_root(op);
	}

	template<typename... ContR, typename OP>
	void inline external(std::tuple<label::read, ContR...> cr, OP &&op SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		std::tuple<label::write> no_writes{label::write{}};
		mpi::external_core("[external]", SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS, cr, op, no_writes);
	}

	template<typename OP, typename... ContW>
	void inline external(OP &&op, std::tuple<label::write, ContW...> cw SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		std::tuple<label::read> no_reads{label::read{}};
		mpi::external_core("[external]", SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS, no_reads, op, cw);
	}

	template<typename... ContR, typename OP, typename... ContW>
	void inline external(std::tuple<label::read, ContR...> cr, OP &&op, std::tuple<label::write, ContW...> cw SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		mpi::external_core("[external]", SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS, cr, op, cw);
	}

	// With a description of the code: the label of the call in the trace

	template<typename OP>
	void inline external(std::string label, OP &&op SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		std::tuple<label::read> no_reads{label::read{}};
		std::tuple<label::write> no_writes{label::write{}};
		mpi::external_core(label, SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS, no_reads, op, no_writes);
	}

	template<typename... ContR, typename OP>
	void inline external(std::string label, std::tuple<label::read, ContR...> cr, OP &&op SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		std::tuple<label::write> no_writes{label::write{}};
		mpi::external_core(label, SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS, cr, op, no_writes);
	}

	template<typename OP, typename... ContW>
	void inline external(std::string label, OP &&op, std::tuple<label::write, ContW...> cw SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		std::tuple<label::read> no_reads{label::read{}};
		mpi::external_core(label, SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS, no_reads, op, cw);
	}

	template<typename... ContR, typename OP, typename... ContW>
	void inline external(std::string label, std::tuple<label::read, ContR...> cr, OP &&op, std::tuple<label::write, ContW...> cw SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM)
	{
		mpi::external_core(label, SKEPU_EXTERNAL_SOURCE_LOCATION_ARGS, cr, op, cw);
	}
}
