#pragma once
#ifndef SKEPU_CLUSTER_SKELETON_BASE_HPP
#define SKEPU_CLUSTER_SKELETON_BASE_HPP 1

#include <skepu3/cluster/common.hpp>

namespace skepu {

namespace backend {

class SkeletonBase
{
public:
	virtual ~SkeletonBase() noexcept {};

	auto finishAll() noexcept -> void {};
	auto setExecPlan(ExecPlan) noexcept -> void {};

	auto virtual setBackend(BackendSpec bs) noexcept -> void
	{
		if(m_bs)
			delete m_bs;
		m_bs = new BackendSpec(std::move(bs));
	}

	auto virtual
	resetBackend() noexcept
	-> void
	{
		if(m_bs)
			delete m_bs;
		m_bs = 0;
	}

protected:
	BackendSpec * m_bs;

	SkeletonBase() noexcept
	: m_bs(0)
	{}
}; // class SkeletonBase

} // namespace backend

} // namespace skepu


#endif // SKEPU_CLUSTER_SKELETON_BASE_HPP
