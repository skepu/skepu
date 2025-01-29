#ifndef INDEX_INL
#define INDEX_INL

#include <skepu3/cluster/index.hpp>

namespace skepu
{
	inline Index2D operator+(const Index2D & lhs, const Index2D & rhs)
	{
		Index2D res;
		res.row = lhs.row + rhs.row;
		res.col = lhs.col + rhs.col;
		res.i = res.col;
		return res;
	}

	inline bool Index2D
	::transpose()
	{
		std::swap(row, col);
		return true;
	}
}

#endif /* INDEX_INL */
