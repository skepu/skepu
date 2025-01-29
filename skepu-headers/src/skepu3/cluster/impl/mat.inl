#ifndef MAT_INL
#define MAT_INL

#include <skepu3/cluster/mat.hpp>
#include <starpu.h>

namespace skepu
{
	template<typename T>
	Mat<T>
	::Mat(void * buffer)
	{
		data = (T*)STARPU_MATRIX_GET_PTR(buffer);
		rows = STARPU_MATRIX_GET_NY(buffer);
		cols = STARPU_MATRIX_GET_NX(buffer);
		ld = STARPU_MATRIX_GET_LD(buffer);
		size = rows*cols;
	}


	template<typename T>
	inline const T & Mat<T>
	::operator[](size_t index) const
	{
		assert(index < rows * ld);
		return data[index];
	}

	template<typename T>
	inline T & Mat<T>
	::operator[](const size_t index)
	{
		assert(index < rows * ld);
		return data[index];
	}

	template<typename T>
	inline T & Mat<T>
	::operator()(const size_t row, const size_t col)
	{
		assert(row < rows && col < cols);
		return data[row*ld + col];
	}

	template<typename T>
	int Mat<T>
	::offset(const Offset2D && offset)
	{
		rows -= offset.row;
		cols -= offset.col;
		data += offset.row*ld + offset.col;
		return 1;
	}
}

#endif /* MAT_INL */
