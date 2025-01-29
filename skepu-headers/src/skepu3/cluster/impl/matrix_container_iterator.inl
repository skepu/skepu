#ifndef MATRIX_CONTAINER_ITERATOR_INL
#define MATRIX_CONTAINER_ITERATOR_INL

#include <skepu3/cluster/matrix_container_iterator.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			template<typename T>
			matrix_container_iterator<T>
			::matrix_container_iterator(const Index2D & start,
			                            const Size2D & size,
			                            starpu_matrix_container<T> & parent)
				: m_start { start },
				  m_size { size },
				  m_parent { parent },
				  m_current { start }
			{}
		}
	}
}

#endif /* MATRIX_CONTAINER_ITERATOR_INL */
