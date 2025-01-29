#ifndef VECTOR_ITERATOR_HPP
#define VECTOR_ITERATOR_HPP

#include <skepu3/cluster/matrix_iterator.hpp>

namespace skepu
{
	template<typename T>
	using VectorIterator = MatrixIterator<T>;
}

#endif /* VECTOR_ITERATOR_HPP */
