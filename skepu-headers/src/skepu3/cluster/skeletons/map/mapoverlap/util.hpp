#pragma once
#ifndef SKEPU_CLUSTER_MAPOVERLAP_UTIL_HPP
#define SKEPU_CLUSTER_MAPOVERLAP_UTIL_HPP 1

#include <starpu_mpi.h>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>
#include <skepu3/cluster/skeletons/skeleton_base.hpp>
#include "modes.hpp"
#include "region.hpp"

namespace skepu {
namespace util {

template<typename UserFunc>
struct MapOverlapBaseType
{
	typedef typename region_type<
			typename parameter_type<
					(UserFunc::indexed ? 1 : 0),
					decltype(&UserFunc::CPU)>::type>::type
		type;
};

template<typename T>
class MapOverlapBase
{
protected:
	Overlap m_overlap_mode = skepu::Overlap::RowWise;
	Edge m_edge = Edge::Duplicate;
	T m_pad {};

public:
	void setOverlapMode(Overlap mode)
	{
		m_overlap_mode = mode;
	}

	void setEdgeMode(Edge mode)
	{
		m_edge = mode;
	}

	void setPad(T pad)
	{
		m_pad = pad;
	}
}; // class MapOverlapBase

template<typename T>
class border_region
{
	T * m_data;
	starpu_data_handle_t m_handle;

public:
	border_region() = delete;
	border_region(border_region const &) = delete;

	border_region(border_region && other) noexcept
	: m_handle(other.m_handle), m_data(other.m_data)
	{
		other.m_handle = 0;
		other.m_data = 0;
	}

	border_region(size_t count, size_t rank) noexcept
	: m_handle(0), m_data(0)
	{
		if(!count)
			count = 1;

		/* StarPU limitations with resource allocation
		 * Ideally we would want to not allocate memory here, but rather let StarPU
		 * deal with the resource allocation. However, StarPU imposes limitations
		 * when using parallel tasks (codelet.max_parallelism is set). So we have to
		 * allocate memory on every node for the time being. This also means that
		 * deletion of the buffer is synchronous since we cannot free the memory
		 * before any of the dependent tasks are finished.
		 */
		m_data = new T[count];
		starpu_vector_data_register(
			&m_handle,
			STARPU_MAIN_RAM,
			(uintptr_t)m_data,
			(uint32_t)count,
			sizeof(T));
		starpu_mpi_data_register(m_handle, cluster::mpi_tag(), rank);
	}

	~border_region() noexcept
	{
		/* We would like to use starpu_data_unregister_submit here, but we can't.
		 * More info on why in the comment in the c-tor.
		 */
		if(m_handle)
			starpu_data_unregister_no_coherency(m_handle);
		m_handle = 0;
		if(m_data)
			delete[] m_data;
		m_data = 0;
	}

	auto
	handle() noexcept
	-> starpu_data_handle_t
	{
		return m_handle;
	}
}; // class border_region

} // namespace util
} // namespace skepu

#endif // SKEPU_CLUSTER_MAPOVERLAP_UTIL_HPP
