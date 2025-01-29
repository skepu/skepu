#pragma once
#ifndef SKEPU_STARPU_VECTOR_PARTITION_HPP
#define SKEPU_STARPU_VECTOR_PARTITION_HPP 1

#include <starpu_mpi.h>

#include "../partition.hpp"

namespace skepu {
namespace util {

template<typename T>
class vector_partition : private partition_base<T>
{
	typedef partition_base<T> base;

public:
	vector_partition() noexcept
	: base(starpu_vector_filter_block)
	{}

	vector_partition(size_t count) noexcept
	: base(starpu_vector_filter_block)
	{
		init(count);
	}

	vector_partition(vector_partition const & other) noexcept
	: vector_partition(other.m_size)
	{
		base::copy(other);
	}

	vector_partition(vector_partition && other) noexcept
	: base(std::move(other))
	{}

	~vector_partition() noexcept = default;

	auto
	init(size_t count) noexcept
	-> void
	{
		set_sizes(count);
		if(base::m_external)
			alloc_local_storage();
		else
			alloc_partitions();
	}

	auto
	init(T * ptr, size_t count, bool dealloc_mdata) noexcept
	-> void
	{
		set_sizes(count);
		base::m_dealloc_mdata = dealloc_mdata;
		register_local_storage(ptr);
		alloc_partitions();
		base::m_data_valid = true;
		base::m_part_valid = false;
	}

	auto
	operator=(vector_partition const & other) noexcept
	-> vector_partition &
	{
		this->~vector_partition();
		new(this) vector_partition(other);

		return *this;
	}

	auto
	operator=(vector_partition && other) noexcept
	-> vector_partition &
	{
		this->~vector_partition();
		new(this) vector_partition(std::move(other));

		return *this;
	}

	auto
	getParent() noexcept
	-> vector_partition &
	{
		return *this;
	}

	auto
	size_i() const noexcept
	-> size_t
	{
		return base::m_size;
	}

	auto
	size_j() const noexcept
	-> size_t
	{
		return 0;
	}

	auto
	size_k() const noexcept
	-> size_t
	{
		return 0;
	}

	auto
	size_l() const noexcept
	-> size_t
	{
		return 0;
	}

	using base::operator();
	using base::allgather;
	using base::block_count_from;
	using base::capacity;
	using base::data;
	using base::fill;
	using base::filter;
	using base::gather_to_root;
	using base::invalidate_local_storage;
	using base::local_storage_handle;
	using base::handle_for;
	using base::make_ext_w;
	using base::min_filter_parts;
	using base::num_parts;
	using base::part_offset;
	using base::partition;
	using base::randomize;
	using base::scatter_from_root;
	using base::set;
	using base::size;

private:
	auto
	alloc_local_storage() noexcept
	-> void override
	{
		if(!base::m_size)
			return;

		if(!base::m_data)
			base::m_data = new T[base::m_size];

		if(!base::m_external && !base::m_data_handle)
		{
			starpu_data_handle_t & handle = base::m_data_handle;
			starpu_vector_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)(base::m_data),
				base::m_size,
				sizeof(T));
			starpu_mpi_data_register(
				base::m_data_handle,
				cluster::mpi_tag(),
				STARPU_MPI_PER_NODE);
		}
	}

	auto
	alloc_partitions() noexcept
	-> void override
	{
		if(!base::m_size)
			return;

		if(!base::m_part_data)
			base::m_part_data = new T[base::m_part_size];

		for(size_t i(0); i < base::m_handles.size(); ++i)
		{
			auto & handle = base::m_handles[i];
			if(handle)
				continue;

			if(i == cluster::mpi_rank())
				starpu_register(base::m_part_data, base::m_part_size, handle, i);
			else
				starpu_register(0, base::m_part_size, handle, i);
		}
	}

	auto
	get_ptr(starpu_data_handle_t & handle) noexcept
	-> T * override
	{
		return (T *)starpu_vector_get_local_ptr(handle);
	}

	auto
	register_local_storage(T * ptr) noexcept
	-> void
	{
		base::m_data = ptr;
		if(!base::m_external && !base::m_data_handle)
		{
			starpu_data_handle_t & handle = base::m_data_handle;
			starpu_vector_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)(base::m_data),
				base::m_size,
				sizeof(T));
			starpu_mpi_data_register(
				base::m_data_handle,
				cluster::mpi_tag(),
				STARPU_MPI_PER_NODE);
		}
	}

	auto
	set_sizes(size_t count) noexcept
	-> void
	{
		base::m_size = count;
		base::m_filter_block_size = 1;
		if(count)
		{
			base::m_part_size = count / cluster::mpi_size();
			if(count % cluster::mpi_size())
				++base::m_part_size;
			base::m_capacity = base::m_part_size * cluster::mpi_size();
		}
	}

	auto
	starpu_register(
		T * ptr,
		size_t count,
		starpu_data_handle_t & handle,
		int rank) noexcept
	-> void
	{
		if(ptr)
			starpu_vector_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)ptr,
				count,
				sizeof(T));
		else
			starpu_vector_data_register(
				&handle,
				-1, // StarPU takes care of data management for other ranks handles.
				0,
				count,
				sizeof(T));

		starpu_mpi_data_register(
			handle,
			cluster::mpi_tag(),
			rank);
	}

	auto
	update_sizes() noexcept
	-> void override
	{
		auto size = base::m_size;
		starpu_data_handle_t size_handle;
		starpu_variable_data_register(
			&size_handle,
			STARPU_MAIN_RAM,
			(uintptr_t)&size,
			sizeof(size_t));
		starpu_mpi_data_register(size_handle, cluster::mpi_tag(), 0);
		starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, size_handle);
		starpu_data_acquire(size_handle, STARPU_RW);

		set_sizes(size);

		starpu_data_release(size_handle);
		starpu_data_unregister_no_coherency(size_handle);

		alloc_partitions();
		alloc_local_storage();
	}
};

} // namespace util
} // namespace skepu

#endif // SKEPU_STARPU_VECTOR_PARTITION_HPP
