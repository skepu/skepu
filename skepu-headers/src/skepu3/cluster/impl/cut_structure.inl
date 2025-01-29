#ifndef CUT_STRUCTURE_INL
#define CUT_STRUCTURE_INL

#include <skepu3/cluster/cut_structure.hpp>
#include <assert.h>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			/**
			 * Create a cut structure
			 * @param size The number of elements to partition
			 * @param parts How many parts the elements should be split into
			 */
			inline cut_structure
			::cut_structure(const size_t & size, const size_t & parts)
				: m_block_count { parts }, m_total_size { size }
			{
				assert(parts <= size && "Cannot divide in more parts than elements");
				if (m_total_size % parts == 0) {
					m_block_size = m_total_size / m_block_count;
					m_last_block_size = m_block_size;
				} else {
					m_block_size = m_total_size / m_block_count;
					m_last_block_size = m_block_size + m_total_size % m_block_count;
				}
				assert(m_block_size > 0);
				assert(m_last_block_size >= m_block_size);
			}

			/**
			 * @brief Find the index of the first element in a given block
			 *
			 * @param block
			 * @return size_t index of first element in the block
			 */
			inline size_t cut_structure
			::block_start_idx(const size_t & block) const
			{
				assert(block < m_block_count && "block index out of bounds");
				return block*m_block_size;
			}


			/**
			 * @brief Find the first index of the *next* block
			 *
			 * @param block
			 * @return size_t index of the first element in the next block
			 */
			inline size_t cut_structure
			::next_block_start_idx(const size_t & block) const
			{
				assert(block < m_block_count && "block index out of bounds");
				if (block + 1 == m_block_count)
				{
					return m_total_size;
				}
				return block*m_block_size + m_block_size;
			}


			/**
			 * @brief Get the block owning the element at `idx`
			 *
			 * @param idx element index
			 * @return size_t block with element at index present
			 */
			inline size_t cut_structure
			::block(const size_t & idx) const
			{
				assert(idx < m_total_size && "element index out of bounds");
				return std::min<size_t>(idx / m_block_size, block_count() - 1);
			}

			/**
			 * @brief Number of elements inside a block
			 *
			 * @param idx
			 * @return size_t Number of elements inside block containing `idx`
			 */
			inline size_t cut_structure
			::block_size_by_elem(const size_t & idx) const
			{
				assert(idx < size() && "element index out of bounds");
				return block_size(block(idx));
			}


			/**
			 * @brief The total number of elements
			 *
			 * @return size_t
			 */
			inline size_t cut_structure
			::size() const
			{
				assert(m_total_size > 0);
				return m_total_size;
			}


			/**
			 * @brief The total number of blocks
			 *
			 * @return size_t
			 */
			inline size_t cut_structure
			::block_count() const
			{
				return m_block_count;
			}


			/**
			 * @brief Number of elements inside a block
			 *
			 * @param block
			 * @return size_t Number of elements inside `block`
			 */
			inline size_t cut_structure
			::block_size(const size_t & block) const
			{
				assert(block < m_block_count);
				if (block + 1 == m_block_count)
				{
					assert(m_last_block_size > 0);
					return m_last_block_size;
				}
				assert(m_block_size > 0);
				return m_block_size;
			}


			/**
			 * @brief Find the offset from the start of the owning
			 * block to the element at `idx`
			 *
			 * @param idx element index
			 * @return size_t offset from start of block
			 */
			inline size_t cut_structure
			::offset_in_block(const size_t & idx) const
			{
				assert(idx < m_total_size);
				if (block(idx) + 1 == m_block_count){
					return idx - (m_block_count - 1)*m_block_size;
				}
				return idx % m_block_size;
			}


			/**
			 * @brief Find the number of elements from `idx` to the next
			 * block boundary
			 *
			 * @param idx element index
			 * @return size_t "size left in block"
			 */
			inline size_t cut_structure
			::size_in_block(const size_t & idx) const
			{
				assert(idx < m_total_size);
				return block_size(block(idx)) - offset_in_block(idx);
			}
		};
	}
}

#endif /* CUT_STRUCTURE_INL */
