#ifndef MATRIX_CONTAINER_ITERATOR_HPP
#define MATRIX_CONTAINER_ITERATOR_HPP

#include <skepu3/cluster/starpu_matrix_container.hpp>
#include <skepu3/cluster/index.hpp>
#include <skepu3/cluster/container_cut.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			/**
			 * This isn't a full implementation of an "iterator", but
			 * provides some minimal iterator-like
			 * abstractions. Should not be exposed to the user.
			 *
			 * The primary use is to get all relevant blocks together
			 * with offsets and sizes when creating tasks within the
			 * skeletons.
			 */
			template<typename T>
			class matrix_container_iterator
			{
			private:
				Index2D m_start;
				Size2D m_size;
				Index2D m_current;
				starpu_matrix_container<T> & m_parent;

			public:
				matrix_container_iterator(const Index2D & start,
				                          const Size2D & size,
				                          starpu_matrix_container<T> & parent);
				const matrix_container_iterator& operator+=(const Size2D & step);
			};
		}
	}
}

#include <skepu3/cluster/impl/matrix_container_iterator.inl>

#endif /* MATRIX_CONTAINER_ITERATOR_HPP */
