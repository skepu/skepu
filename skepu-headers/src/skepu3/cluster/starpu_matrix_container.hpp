#ifndef STARPU_MATRIX_CONTAINER_HPP
#define STARPU_MATRIX_CONTAINER_HPP

#include <vector>
#include <iostream>

#include <starpu.h>
#include <skepu3/cluster/cut_structure.hpp>
#include <skepu3/cluster/handle_cut.hpp>


namespace skepu
{
	namespace cluster
	{
		/**
		 * This class provides a high level c++ interface to the
		 * starpu matrix data container. This is made to be used as the backend
		 * to the skepu::Vector and skepu::Matrix containers, as well as a
		 * unified target for skeleton execution.
		 */
		template<typename T>
		class starpu_matrix_container {
			bool m_unpartitioned_valid = false;
			T * local_data_ptr;
			starpu_data_handle_t local_data_handle;
			std::vector<starpu_data_handle_t> m_children;
			std::vector<uintptr_t> m_child_data;
			size_t m_n_owned_children = 0;

			helpers::cut_structure m_row_struct;
			helpers::cut_structure m_col_struct;

			void partition();
			void partition(starpu_matrix_container const &);
			size_t block_row_col_to_idx(const size_t & block_row,
			                            const size_t & block_col) const;
			size_t local_elem_idx(starpu_data_handle_t & handle,
			                      const size_t & row,
			                      const size_t & col) const;
		public:
			starpu_matrix_container(const size_t & height, const size_t & width);
			~starpu_matrix_container();
			size_t block_owner(const size_t & block_row,
			                   const size_t & block_col) const;
			size_t elem_owner(const size_t & row, const size_t & col) const;
			T operator()(const size_t & row, const size_t & col);
			T operator[](const size_t & col);
			starpu_data_handle_t & get_block(const size_t & block_row,
			                                const size_t & block_col);
			starpu_data_handle_t const & get_block(const size_t & block_row,
			                                const size_t & block_col) const;
			starpu_data_handle_t& get_block_by_elem(const size_t & row,
			                                        const size_t & col);
			void set(const size_t & row, const size_t & col, const T & value);
			void set(const size_t & i, const T & value);
			size_t width() const;
			size_t height() const;
			size_t size() const;

			starpu_data_handle_t allgather();
			bool invalidate_unpartition();

			// Scheduling
			size_t row_block_height(const size_t & row) const;
			size_t col_block_width(const size_t & col) const;
			helpers::handle_cut largest_cut(const Index2D & idx);
		};
	}
}

#include <skepu3/cluster/impl/starpu_matrix_container.inl>
#endif /* STARPU_MATRIX_CONTAINER_HPP */
