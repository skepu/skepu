#pragma once
#ifndef SKEPU_CLUSTER_REGION_HPP
#define SKEPU_CLUSTER_REGION_HPP 1

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>

#include "modes.hpp"

namespace skepu {
namespace util {

template<typename T>
auto inline
clamp(
	T const & value,
	T const & min,
	T const & max) noexcept
-> T
{
	return
		value < min
		? min
		: max < value ? max : value;
}

template<
	typename Region,
	typename Index>
auto inline
set_index(Region & r, Index const & idx) noexcept
-> void
{
	r.set_index(idx);
}

} // namespace util

template<typename T>
class Region1D
{
	T const * m_data_start;
	T const * m_data;
	T const * m_data_end;

	size_t const m_size;
	size_t const m_task_size;

	size_t const m_offset;

	Edge m_edge;
	Index1D m_idx;
	T m_pad;

public:
	int const oi;

	Region1D(
		Edge edge, T const & pad, size_t offset,
		T const * start, T const * data, T const * end,
		int const overlap,
		size_t size, size_t task_size) noexcept
	:	m_edge(edge), m_pad(pad), m_offset(offset),
		m_data_start(start), m_data(data), m_data_end(end),
		oi(overlap),
		m_size(size), m_task_size(task_size)
	{}

	Region1D(Region1D const &) noexcept = default;
	Region1D(Region1D &&) noexcept = default;

	auto
	operator()(int i) noexcept
	-> T const &
	{
		switch(m_edge)
		{
			case Edge::None:
			case Edge::Cyclic:
			{
				auto ptr = m_data;
				ptrdiff_t pos = m_idx.i + i;
				if(pos < 0)
					ptr = m_data_start;
				else if(pos >= m_task_size)
					ptr = m_data_end;

				return ptr[pos];
			}
			case Edge::Duplicate:
			{
				auto ptr = m_data;
				ptrdiff_t pos = m_idx.i + i;
				if(pos < 0)
				{
					if((ptrdiff_t)(pos +m_offset) < 0)
						pos = -m_offset;
					if(m_offset)
						ptr = m_data_start;
				}
				else if(pos >= m_task_size)
				{
					if(pos +m_offset >= m_size)
						pos = m_size -m_offset -1;
					if(m_offset + m_task_size != m_size)
						ptr = m_data_end;
				}

				return ptr[pos];
			}
			case Edge::Pad:
			{
				auto ptr = m_data;
				ptrdiff_t pos = m_idx.i + i;
				if(pos < 0)
				{
					if((ptrdiff_t)(pos +m_offset) < 0)
						return m_pad;
					else
						ptr = m_data_start;
				}
				else if(pos >= m_task_size)
				{
					if(pos +m_offset >= m_size)
						return m_pad;
					else
						ptr = m_data_end;
				}

				return ptr[pos];
			}
			default:
				std::cerr << "[SkePU][Region1D][operator()] "
					"Edge mode not supported.\n";
				std::abort();
		}
	}

private:
	friend auto
	util::set_index<Region1D, Index1D>(Region1D &, Index1D const &) noexcept
	-> void;

	auto
	set_index(Index1D const & idx) noexcept
	-> void
	{
		m_idx = idx;
	}
};

template<typename T>
class Region2D
{
	T const * m_data_start;
	T const * m_data;
	T const * m_data_end;

	size_t const m_size_i;
	size_t const m_task_size_i;
	size_t const m_size_j;

	size_t const m_i_offset;

	Edge m_edge;
	Index2D m_idx;
	T m_pad;

public:
	int const oi;
	int const oj;

	Region2D(
		Edge edge, T const & pad, size_t i_offset,
		T const * start, T const * data, T const * end,
		int oi, int oj,
		size_t size_i, size_t task_size_i,
		size_t size_j) noexcept
	:	m_edge(edge), m_pad(pad), m_i_offset(i_offset),
		m_data_start(start), m_data(data), m_data_end(end),
		oi(oi), oj(oj),
		m_size_i(size_i), m_task_size_i(task_size_i),
		m_size_j(size_j)
	{}

	Region2D(Region2D const &) noexcept = default;
	Region2D(Region2D &&) noexcept = default;

	auto
	operator()(int i, int j) noexcept
	-> T const &
	{
		switch(m_edge)
		{
			case Edge::None:
			case Edge::Cyclic:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.row + i;
				if(pos_i < 0)
					ptr = m_data_start;
				else if(pos_i >= m_task_size_i)
					ptr = m_data_end;

				return ptr[
					pos_i * m_size_j
					+ (m_idx.col +j + m_size_j) % m_size_j];
			}
			case Edge::Duplicate:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.row + i;
				if(pos_i < 0)
				{
					if((ptrdiff_t)(pos_i +m_i_offset) < 0)
						pos_i = -m_i_offset;
					if(m_i_offset)
						ptr = m_data_start;
				}
				else if(pos_i >= m_task_size_i)
				{
					if(pos_i +m_i_offset >= m_size_i)
						pos_i = m_size_i -m_i_offset -1;
					if(m_i_offset + m_task_size_i != m_size_i)
						ptr = m_data_end;
				}

				auto pos_j = util::clamp<ptrdiff_t>(m_idx.col + j, 0, m_size_j -1);
				return ptr[
					pos_i*m_size_j
					+ pos_j];
			}
			case Edge::Pad:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.row + i;
				if(pos_i < 0)
				{
					if((ptrdiff_t)(pos_i +m_i_offset) < 0)
						return m_pad;
					else
						ptr = m_data_start;
				}
				else if(pos_i >= m_task_size_i)
				{
					if(pos_i +m_i_offset >= m_size_i)
						return m_pad;
					else
						ptr = m_data_end;
				}

				ptrdiff_t pos_j = m_idx.col + j;
				if(pos_j < 0 || pos_j >= m_size_j)
					return m_pad;
				return ptr[pos_i*m_size_j + pos_j];
			}
			default:
				std::cerr << "[SkePU][Region2D][operator()] "
					"Edge mode not supported.\n";
				std::abort();
		}
	}

private:
	friend auto
	util::set_index<Region2D, Index2D>(Region2D &, Index2D const &) noexcept
	-> void;

	auto
	set_index(Index2D const & idx) noexcept
	-> void
	{
		m_idx = idx;
	}
};

template<typename T>
class Region3D
{
	T const * m_data_start;
	T const * m_data;
	T const * m_data_end;

	size_t const m_size_i;
	size_t const m_task_size_i;
	size_t const m_size_j;
	size_t const m_size_k;

	size_t const m_stride_i;

	size_t const m_i_offset;

	Edge m_edge;
	Index3D m_idx;
	T m_pad;

public:
	int const oi;
	int const oj;
	int const ok;

	Region3D(
		Edge edge, T const & pad, size_t i_offset,
		T const * start, T const * data, T const * end,
		int oi, int oj, int ok,
		size_t size_i, size_t task_size_i,
		size_t size_j, size_t size_k,
		size_t stride_i) noexcept
	:	m_edge(edge), m_pad(pad), m_i_offset(i_offset),
		m_data_start(start), m_data(data), m_data_end(end),
		oi(oi), oj(oj), ok(ok),
		m_size_i(size_i), m_task_size_i(task_size_i),
		m_size_j(size_j), m_size_k(size_k),
		m_stride_i(stride_i)
	{}

	Region3D(Region3D const &) noexcept = default;
	Region3D(Region3D &&) noexcept = default;

	auto
	operator()(int i, int j, int k) noexcept
	-> T const &
	{
		switch(m_edge)
		{
			case Edge::None:
			case Edge::Cyclic:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.i + i;
				if(pos_i < 0)
					ptr = m_data_start;
				else if(pos_i >= m_task_size_i)
					ptr = m_data_end;

				return ptr[
					pos_i * m_stride_i
					+ ((m_idx.j +j + m_size_j) % m_size_j) * m_size_k
					+ (m_idx.k +k + m_size_k) % m_size_k];
			}
			case Edge::Duplicate:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.i + i;
				if(pos_i < 0)
				{
					if((ptrdiff_t)(pos_i +m_i_offset) < 0)
						pos_i = -m_i_offset;
					if(m_i_offset)
						ptr = m_data_start;
				}
				else if(pos_i >= m_task_size_i)
				{
					if(pos_i +m_i_offset >= m_size_i)
						pos_i = m_size_i -m_i_offset -1;
					if(m_i_offset + m_task_size_i != m_size_i)
						ptr = m_data_end;
				}

				auto pos_j = util::clamp<ptrdiff_t>(m_idx.j + j, 0, m_size_j -1);
				auto pos_k = util::clamp<ptrdiff_t>(m_idx.k + k, 0, m_size_k -1);
				return ptr[
					pos_i*m_stride_i
					+ pos_j*m_size_k
					+ pos_k];
			}
			case Edge::Pad:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.i + i;
				if(pos_i < 0)
				{
					if((ptrdiff_t)(pos_i +m_i_offset) < 0)
						return m_pad;
					else
						ptr = m_data_start;
				}
				else if(pos_i >= m_task_size_i)
				{
					if(pos_i +m_i_offset >= m_size_i)
						return m_pad;
					else
						ptr = m_data_end;
				}

				ptrdiff_t pos_j = m_idx.j + j;
				ptrdiff_t pos_k = m_idx.k + k;
				if(pos_j < 0 || pos_j >= m_size_j
						|| pos_k < 0 || pos_k >= m_size_k)
					return m_pad;
				return ptr[pos_i*m_stride_i + pos_j*m_size_k + pos_k];
			}
			default:
				std::cerr << "[SkePU][Region3D][operator()] "
					"Edge mode not supported.\n";
				std::abort();
		}
	}

private:
	friend auto
	util::set_index<Region3D, Index3D>(Region3D &, Index3D const &) noexcept
	-> void;

	auto
	set_index(Index3D const & idx) noexcept
	-> void
	{
		m_idx = idx;
	}
};

template<typename T>
class Region4D
{
	T const * m_data_start;
	T const * m_data;
	T const * m_data_end;

	size_t const m_size_i;
	size_t const m_task_size_i;
	size_t const m_size_j;
	size_t const m_size_k;
	size_t const m_size_l;

	size_t const m_stride_i;
	size_t const m_stride_j;

	size_t const m_i_offset;

	Edge m_edge;
	Index4D m_idx;
	T m_pad;

public:
	int const oi;
	int const oj;
	int const ok;
	int const ol;

	Region4D(
		Edge edge, T const & pad, size_t i_offset,
		T const * start, T const * data, T const * end,
		int oi, int oj, int ok, int ol,
		size_t size_i, size_t task_size_i,
		size_t size_j, size_t size_k, size_t size_l,
		size_t stride_i, size_t stride_j) noexcept
	:	m_edge(edge), m_pad(pad), m_i_offset(i_offset),
		m_data_start(start), m_data(data), m_data_end(end),
		oi(oi), oj(oj), ok(ok), ol(ol),
		m_size_i(size_i), m_task_size_i(task_size_i),
		m_size_j(size_j), m_size_k(size_k), m_size_l(size_l),
		m_stride_i(stride_i), m_stride_j(stride_j)
	{}

	Region4D(Region4D const &) noexcept = default;
	Region4D(Region4D &&) noexcept = default;

	auto
	operator()(int i, int j, int k, int l) noexcept
	-> T const &
	{
		switch(m_edge)
		{
			case Edge::None:
			case Edge::Cyclic:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.i + i;
				if(pos_i < 0)
					ptr = m_data_start;
				else if(pos_i >= m_task_size_i)
					ptr = m_data_end;

				return ptr[
					pos_i * m_stride_i
					+ ((m_idx.j +j + m_size_j) % m_size_j) * m_stride_j
					+ ((m_idx.k +k + m_size_k) % m_size_k) * m_size_l
					+ (m_idx.l +l + m_size_l) % m_size_l];
			}
			case Edge::Duplicate:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.i + i;
				if(pos_i < 0)
				{
					if((ptrdiff_t)(pos_i +m_i_offset) < 0)
						pos_i = -m_i_offset;
					if(m_i_offset)
						ptr = m_data_start;
				}
				else if(pos_i >= m_task_size_i)
				{
					if(pos_i +m_i_offset >= m_size_i)
						pos_i = m_size_i -m_i_offset -1;
					if(m_i_offset + m_task_size_i != m_size_i)
						ptr = m_data_end;
				}

				auto pos_j = util::clamp<ptrdiff_t>(m_idx.j + j, 0, m_size_j -1);
				auto pos_k = util::clamp<ptrdiff_t>(m_idx.k + k, 0, m_size_k -1);
				auto pos_l = util::clamp<ptrdiff_t>(m_idx.l + l, 0, m_size_l -1);
				return ptr[
					pos_i*m_stride_i
					+ pos_j*m_stride_j
					+ pos_k*m_size_l
					+ pos_l];
			}
			case Edge::Pad:
			{
				auto ptr = m_data;
				ptrdiff_t pos_i = m_idx.i + i;
				if(pos_i < 0)
				{
					if((ptrdiff_t)(pos_i +m_i_offset) < 0)
						return m_pad;
					else
						ptr = m_data_start;
				}
				else if(pos_i >= m_task_size_i)
				{
					if(pos_i +m_i_offset >= m_size_i)
						return m_pad;
					else
						ptr = m_data_end;
				}

				ptrdiff_t pos_j = m_idx.j + j;
				ptrdiff_t pos_k = m_idx.k + k;
				ptrdiff_t pos_l = m_idx.l + l;
				if(pos_j < 0 || pos_j >= m_size_j
						|| pos_k < 0 || pos_k >= m_size_k
						|| pos_l < 0 || pos_l >= m_size_l)
					return m_pad;
				return ptr[
					pos_i*m_stride_i
					+ pos_j*m_stride_j
					+ pos_k*m_size_l
					+ pos_l];
			}
			default:
				std::cerr << "[SkePU][Region4D][operator()] "
					"Edge mode not supported.\n";
				std::abort();
		}
	}

private:
	friend auto
	util::set_index<Region4D, Index4D>(Region4D &, Index4D const &) noexcept
	-> void;

	auto
	set_index(Index4D const & idx) noexcept
	-> void
	{
		m_idx = idx;
	}
};


template<typename T>
struct region_type;

template<typename T>
struct region_type<Region1D<T>> { using type = T; };

template<typename T>
struct region_type<Region2D<T>> { using type = T; };

template<typename T>
struct region_type<Region3D<T>> { using type = T; };

template<typename T>
struct region_type<Region4D<T>> { using type = T; };

} // namespace skepu

#endif // SKEPU_CLUSTER_REGION_HPP
