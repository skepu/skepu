#pragma once
#ifndef SKEPU_CLUSTER_MAPOVERLAP_MODES_HPP
#define SKEPU_CLUSTER_MAPOVERLAP_MODES_HPP 1

#include <iostream>

namespace skepu {

enum class Edge
{
	None,
	Cyclic,
	Duplicate,
	Pad,
};

enum class Overlap
{
	RowWise,
	ColWise,
};

} // namespace skepu

#endif // SKEPU_CLUSTER_MAPOVERLAP_MODES_HPP
