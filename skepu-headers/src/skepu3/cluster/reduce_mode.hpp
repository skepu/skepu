#ifndef REDUCE_MODE_HPP
#define REDUCE_MODE_HPP
#include <iostream>

namespace skepu
{
	enum class ReduceMode
	{
		RowWise, ColWise, ElWise
	};
	using SweepMode = ReduceMode;

	inline std::ostream &operator<<(std::ostream &o, ReduceMode m)
	{
		switch (m)
		{
		case ReduceMode::RowWise:
			o << "Rowwise"; break;
		case ReduceMode::ColWise:
			o << "Colwise"; break;
		case ReduceMode::ElWise:
			o << "Elwise"; break;
		default:
			o << "<Invalid reduce mode>";
		}
		return o;
	}
}



#endif /* REDUCE_MODE_HPP */
