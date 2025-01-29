#pragma once
#ifndef SKEPU_STARPU_SKELETON_REDUCE_MODE_HPP
#define SKEPU_STARPU_SKELETON_REDUCE_MODE_HPP 1

namespace skepu {

enum class ReduceMode
{
	RowWise,
	ColWise,
};

} // namespace skepu

#endif // SKEPU_STARPU_SKELETON_REDUCE_MODE_HPP
