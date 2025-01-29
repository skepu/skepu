#pragma once
#ifndef SKEPU_STARPU_MATRIX_PARTITION_HPP
#define SKEPU_STARPU_MATRIX_PARTITION_HPP 1

#include <skepu3/cluster/common.hpp>
#include <skepu3/cluster/containers/partition.hpp>

namespace skepu {
namespace util {

template<typename T>
class matrix_partition : private partition_base<T>
{
	typedef partition_base<T> base;
	size_t m_rows;
	size_t m_cols;
	size_t m_part_rows;

public:
	matrix_partition() noexcept
	: base(starpu_matrix_filter_vertical_block),
		m_rows(0),
		m_cols(0),
		m_part_rows(0)
	{}

	matrix_partition(size_t rows, size_t cols) noexcept
	: base(starpu_matrix_filter_vertical_block), m_part_rows(0)
	{
		init(rows, cols);
	}

	matrix_partition(matrix_partition const & other) noexcept
	: base(starpu_matrix_filter_vertical_block),
		m_rows(other.m_rows),
		m_cols(other.m_cols),
		m_part_rows(other.m_part_rows)
	{
		set_sizes();
		if(base::m_size)
		{
			alloc_partitions();
			base::copy(other);
		}
	}

	matrix_partition(matrix_partition && other) noexcept
	: base(std::move(other)),
		m_rows(std::move(other.m_rows)),
		m_cols(std::move(other.m_cols)),
		m_part_rows(std::move(other.m_part_rows))
	{}

	~matrix_partition() noexcept = default;

	auto
	init(size_t rows, size_t cols) noexcept
	-> void
	{
		m_rows = rows;
		m_cols = cols;

		set_sizes();
		if(base::m_external)
			alloc_local_storage();
		else
			alloc_partitions();
	}

	auto
	init(T * ptr, size_t rows, size_t cols, bool dealloc_mdata) noexcept
	-> void
	{
		base::m_dealloc_mdata = dealloc_mdata;
		m_rows = rows;
		m_cols = cols;
		set_sizes();
		register_local_storage(ptr);
		alloc_partitions();
		base::m_data_valid = true;
		base::m_part_valid = false;
	}

	auto
	operator=(matrix_partition const & other) noexcept
	-> matrix_partition &
	{
		this->~matrix_partition();
		new(this) matrix_partition(other);
		return *this;
	}

	auto
	operator=(matrix_partition && other) noexcept
	-> matrix_partition &
	{
		this->~matrix_partition();
		new(this) matrix_partition(std::move(other));
		return *this;
	}

	auto
	operator()(size_t pos) noexcept
	-> T &
	{
		return base::operator()(pos);
	}

	auto
	operator()(size_t row, size_t col) noexcept
	-> T &
	{
		return base::operator()((row * m_cols) + col);
	}

	auto
	operator()(size_t row, size_t col) const noexcept
	-> T const &
	{
		return base::operator()((row * m_cols) + col);
	}

	auto
	block_count_row(size_t row) const noexcept
	-> size_t
	{
		if(!base::m_current_filter)
			return std::min(m_part_rows - (row % m_part_rows), m_rows - row);

		std::cerr << "[SkePU][matrix_partition][block_count_row] "
			"Using partitioned partitions is not supported with this function.\n";
		std::abort();
	}

	auto
	block_offset_row(size_t row) const noexcept
	-> size_t
	{
		if(!base::m_current_filter)
			return (row % m_part_rows) * m_cols;

		std::cerr << "[SkePU][matrix_partition][block_offset_row] "
			"Using partitioned partitions is not supported with this function.\n";
		std::abort();
	}

	/* TODO: Fix flip dim bugs
	 * - Remove the filters. They won't work anyways after the dim is flipped.
	 * - Set sizes
	 * - Update all the StarPU handles
	 */
	auto
	flip_dim() noexcept
	-> void
	{
		std::swap(m_rows, m_cols);
	}

	auto
	getParent() noexcept
	-> matrix_partition &
	{
		return *this;
	}

	auto
	handle_for_row(size_t row) noexcept
	-> starpu_data_handle_t
	{
		// Refer to base::handle_for for comments regarding implementation.
		// The difference being that this function is based on rows rather than
		// elements.
		base::m_data_valid = false;

		auto block_idx = row / m_part_rows;
		if(!base::m_current_filter)
			return base::m_handles[block_idx];

		auto & filter = base::m_filters.at(base::m_current_filter);
		auto & children = filter.children[block_idx];
		auto block_row = row % m_part_rows;

		auto rest_rows = filter.rest * (filter.blocks_per_child +1);
		if(block_row < rest_rows)
			return children[block_row / (filter.blocks_per_child +1)];

		block_row -= rest_rows;
		return children[(block_row / filter.blocks_per_child) + filter.rest];
	}

	auto
	index(size_t pos) const noexcept
	-> Index2D
	{
		Index2D idx;
		idx.row = pos / m_cols;
		idx.col = pos - (idx.row * m_cols);
		return idx;
	}


	auto
	rows() const noexcept
	-> size_t
	{
		return m_rows;
	}

	auto
	cols() const noexcept
	-> size_t
	{
		return m_cols;
	}

	auto
	size_i() const noexcept
	-> size_t
	{
		return m_rows;
	}

	auto
	size_j() const noexcept
	-> size_t
	{
		return m_cols;
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

	auto
	set(size_t const row, size_t const col, T const & val)
	-> void
	{
		base::set((row * m_cols) + col, val);
	}

	template<typename Iterator>
	auto
	set(Iterator begin, Iterator end)
	-> void
	{
		base::set(begin,end);
	}

	auto
	transpose_to(matrix_partition & other) noexcept
	-> void
	{
		if(!base::m_data_valid)
			base::allgather();
		transpose_local_store_to(other);
	}

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
			auto & handle = base::m_data_handle;
			starpu_matrix_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)(base::m_data),
				m_cols,
				m_cols,
				m_rows,
				sizeof(T));
			starpu_mpi_data_register(
				base::m_data_handle,
				skepu::cluster::mpi_tag(),
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

			if(i == skepu::cluster::mpi_rank())
				starpu_register(base::m_part_data, m_part_rows, m_cols, handle, i);
			else
				starpu_register(0, m_part_rows, m_cols, handle, i);
		}
	}

	auto
	get_ptr(starpu_data_handle_t & handle) noexcept
	-> T * override
	{
		return (T *)starpu_matrix_get_local_ptr(handle);
	}

	auto
	register_local_storage(T * ptr) noexcept
	-> void
	{
		base::m_data = ptr;
		if(!base::m_external && !base::m_data_handle)
		{
			auto & handle = base::m_data_handle;
			starpu_matrix_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)(base::m_data),
				m_cols,
				m_cols,
				m_rows,
				sizeof(T));
			starpu_mpi_data_register(
				base::m_data_handle,
				skepu::cluster::mpi_tag(),
				STARPU_MPI_PER_NODE);
		}
	}

	auto
	set_sizes()
	-> void
	{
		m_part_rows = m_rows / skepu::cluster::mpi_size();
		if(m_rows % skepu::cluster::mpi_size())
			++m_part_rows;
		size_t part_size = m_part_rows * m_cols;

		base::m_size = m_rows * m_cols;
		base::m_part_size = part_size;
		base::m_capacity = part_size * skepu::cluster::mpi_size();
		base::m_filter_block_size = m_cols;
	}

	auto
	starpu_register(
		T * ptr,
		size_t const rows,
		size_t const cols,
		starpu_data_handle_t & handle,
		int rank) noexcept
	-> void
	{
		if(ptr)
			starpu_matrix_data_register(
				&handle,
				STARPU_MAIN_RAM,
				(uintptr_t)ptr,
				cols,
				cols,
				rows,
				sizeof(T));
		else
			starpu_matrix_data_register(
				&handle,
				-1, // StarPU takes care of data management for other ranks handles.
				0,
				cols,
				cols,
				rows,
				sizeof(T));

		starpu_mpi_data_register(
			handle,
			cluster::mpi_tag(),
			rank);
	}

	auto
	transpose_local_store_to(matrix_partition & other) noexcept
	-> void
	{
		if(!other.m_data)
			other.alloc_local_storage();
		starpu_data_acquire(base::m_data_handle, STARPU_RW);
		starpu_data_acquire(other.m_data_handle, STARPU_RW);

		auto data_it = base::m_data;
		for(size_t i(0); i < m_rows; ++i)
			for(size_t j(0); j < m_cols; ++j, ++data_it)
				other(j,i) = *data_it;

		starpu_data_release(other.m_data_handle);
		starpu_data_release(base::m_data_handle);

		other.m_data_valid = true;
		other.m_part_valid = false;
	}

	auto
	update_sizes() noexcept
	-> void override
	{
		size_t size_arr[]{m_rows, m_cols};
		starpu_data_handle_t size_handle;
		starpu_variable_data_register(
			&size_handle,
			STARPU_MAIN_RAM,
			(uintptr_t)size_arr,
			sizeof(size_arr));
		starpu_mpi_data_register(size_handle, cluster::mpi_tag(), 0);
		starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, size_handle);
		starpu_data_acquire(size_handle, STARPU_RW);

		m_rows = size_arr[0];
		m_cols = size_arr[1];

		starpu_data_release(size_handle);
		starpu_data_unregister_no_coherency(size_handle);

		set_sizes();

		alloc_partitions();
		alloc_local_storage();
	}
};

} // namespace util
} // namespace skepu

#endif // SKEPU_STARPU_MATRIX_PARTITION_HPP
