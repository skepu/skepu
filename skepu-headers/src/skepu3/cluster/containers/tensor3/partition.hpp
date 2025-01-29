#pragma once
#ifndef SKEPU_STARPU_TENSOR3_PARTITION_HPP
#define SKEPU_STARPU_TENSOR3_PARTITION_HPP 1

#include <starpu_mpi.h>

#include <skepu3/cluster/common.hpp>
#include "../partition.hpp"

namespace skepu {
namespace util {

template<typename T>
class tensor3_partition : private partition_base<T>
{
	typedef partition_base<T> base;

	size_t m_size_i;
	size_t m_size_j;
	size_t m_size_k;
	size_t m_size_jk;
	size_t m_part_i;

public:
	typedef skepu::Index3D index_type;

	tensor3_partition() noexcept
	: base(starpu_block_filter_depth_block)
	{}

	tensor3_partition(size_t i, size_t j, size_t k) noexcept
	: base(starpu_block_filter_depth_block)
	{
		init(i, j, k);
	}

	tensor3_partition(tensor3_partition const & other) noexcept
	: tensor3_partition(other.m_size_i, other.m_size_j, other.m_size_k)
	{
		base::copy(other);
	}

	tensor3_partition(tensor3_partition && other) noexcept
	: base(std::move(other)),
		m_size_i(other.m_size_i),
		m_size_j(other.m_size_j),
		m_size_k(other.m_size_k),
		m_size_jk(other.m_size_jk),
		m_part_i(other.m_part_i)
	{}

	~tensor3_partition() noexcept = default;

	auto
	init(size_t i, size_t j, size_t k) noexcept
	-> void
	{
		m_size_i = i;
		m_size_j = j;
		m_size_k = k;
		m_size_jk = j*k;
		set_sizes();

		if(base::m_external)
			alloc_local_storage();
		else
			alloc_partitions();
	}

	auto
	init(T * ptr, size_t i, size_t j, size_t k, bool dealloc_mdata) noexcept
	-> void
	{
		base::m_dealloc_mdata = dealloc_mdata;
		m_size_i = i;
		m_size_j = j;
		m_size_k = k;
		set_sizes();
		register_local_storage(ptr);
		alloc_partitions();
		base::m_data_valid = true;
		base::m_part_valid = false;
	}

	auto
	operator=(tensor3_partition const & other) noexcept
	-> tensor3_partition &
	{
		this->~tensor3_partition();
		new(this) tensor3_partition(other);
		return *this;
	}

	auto
	operator=(tensor3_partition && other) noexcept
	-> tensor3_partition &
	{
		this->~tensor3_partition();
		new(this) tensor3_partition(std::move(other));
		return *this;
	}

	auto
	operator()(size_t i, size_t j, size_t k) noexcept
	-> T &
	{
		base::m_part_valid = false;
		return base::operator()((i * m_size_jk) + (j * m_size_k) + k);
	}

	auto
	operator()(size_t i, size_t j, size_t k) const noexcept
	-> T const &
	{
		return base::operator()((i * m_size_jk) + (j * m_size_k) + k);
	}

	auto
	block_count_i(size_t i_pos) const noexcept
	-> size_t
	{
		return
			std::min(
				m_part_i - (i_pos % m_part_i),
				m_size_i - i_pos);
	}

	auto
	block_offset_i(size_t i_pos) const noexcept
	-> size_t
	{
		return (i_pos % m_part_i) * m_size_jk;
	}

	auto
	getParent() noexcept
	-> tensor3_partition &
	{
		return *this;
	}

	auto
	handle_for_i(size_t i_pos) noexcept
	-> starpu_data_handle_t
	{
		return base::m_handles[i_pos / m_part_i];
	}

	auto
	index(size_t pos) const noexcept
	-> index_type
	{
		index_type idx;
		idx.i = pos / m_size_jk;
		pos -= idx.i * m_size_jk;
		idx.j = pos / m_size_k;
		idx.k = pos - (idx.j * m_size_k);

		return idx;
	}

	auto
	size_i() const noexcept
	-> size_t
	{
		return m_size_i;
	}

	auto
	size_j() const noexcept
	-> size_t
	{
		return m_size_j;
	}

	auto
	size_k() const noexcept
	-> size_t
	{
		return m_size_k;
	}

	auto
	size_l() const noexcept
	-> size_t
	{
		return 0;
	}

	auto
	stride_i() const noexcept
	-> size_t
	{
		return m_size_jk;
	}

	using base::operator();
	using base::allgather;
	using base::block_count_from;
	using base::capacity;
	using base::data;
	using base::fill;
	using base::filter;
	using base::gather_to_root;
	using base::local_storage_handle;
	using base::handle_for;
	using base::invalidate_local_storage;
	using base::make_ext_w;
	using base::min_filter_parts;
	using base::num_parts;
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

		if(!base::m_external)
		{
			auto & handle = base::m_data_handle;
			starpu_block_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)(base::m_data),
				m_size_k, m_size_k * m_size_j,
				m_size_k, m_size_j, m_size_i,
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

			auto ptr = (i == cluster::mpi_rank() ? base::m_part_data : 0);

			/* StarPU manages memory on nodes that does not own the data. */
			int home_node = -1;
			/* But it the rank owns the data, we manage the storage. */
			if(ptr)
				home_node = STARPU_MAIN_RAM;

			starpu_block_data_register(
				&handle,
				home_node,
				(uintptr_t)ptr,
				m_size_k, m_size_j * m_size_k,
				m_size_k, m_size_j, m_part_i,
				sizeof(T));
			starpu_mpi_data_register(
				handle,
				cluster::mpi_tag(),
				i);
		}
	}

	auto
	get_ptr(starpu_data_handle_t & handle) noexcept
	-> T * override
	{
		return (T *)starpu_block_get_local_ptr(handle);
	}

	auto
	register_local_storage(T * ptr) noexcept
	-> void
	{
		base::m_data = ptr;

		if(!base::m_external)
		{
			auto & handle = base::m_data_handle;
			starpu_block_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)(base::m_data),
				m_size_k, m_size_k * m_size_j,
				m_size_k, m_size_j, m_size_i,
				sizeof(T));
			starpu_mpi_data_register(
				base::m_data_handle,
				cluster::mpi_tag(),
				STARPU_MPI_PER_NODE);
		}
	}

	auto
	set_sizes() noexcept
	-> void
	{
		base::m_size = m_size_i * m_size_j * m_size_k;
		if(base::m_size)
		{
			auto ranks = skepu::cluster::mpi_size();
			m_part_i = m_size_i / ranks;
			if(m_size_i - (m_part_i * ranks))
				++m_part_i;
			base::m_part_size = m_part_i * m_size_j * m_size_k;
			base::m_filter_block_size = m_size_jk;
			base::m_capacity = base::m_part_size * ranks;
		}
	}

	auto
	update_sizes() noexcept
	-> void override
	{
		size_t size_arr[]{m_size_i, m_size_j, m_size_k};
		starpu_data_handle_t size_handle;
		starpu_variable_data_register(
			&size_handle,
			STARPU_MAIN_RAM,
			(uintptr_t)size_arr,
			3 * sizeof(size_t));
		starpu_mpi_data_register(size_handle, cluster::mpi_tag(), 0);
		starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, size_handle);
		starpu_data_acquire(size_handle, STARPU_RW);

		m_size_i = size_arr[0];
		m_size_j = size_arr[1];
		m_size_k = size_arr[2];

		starpu_data_release(size_handle);
		starpu_data_unregister_no_coherency(size_handle);

		set_sizes();

		alloc_partitions();
		alloc_local_storage();
	}
};

} // namespace util
} // namespace skepu

#endif // SKEPU_STARPU_TENSOR3_PARTITION_HPP
