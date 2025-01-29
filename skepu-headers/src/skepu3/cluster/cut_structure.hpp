#ifndef CUT_STRUCTURE_HPP
#define CUT_STRUCTURE_HPP

#include <vector>

// Various helpers for the cluster implementation

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			/**
			 * Data structure abstracting index calculations with even
			 * 1D partitioning.
			 */
			class cut_structure {
			private:
				size_t m_block_size;
				size_t m_last_block_size;
				size_t m_block_count;
				size_t m_total_size;
			public:
				inline cut_structure(const size_t & size, const size_t & parts);
				inline size_t block_start_idx(const size_t & block) const;
				inline size_t next_block_start_idx(const size_t & block) const;
				inline size_t block(const size_t & idx) const;
				inline size_t size() const;
				inline size_t block_count() const;
				inline size_t block_size_by_elem(const size_t & idx) const;
				inline size_t block_size(const size_t & block) const;
				inline size_t offset_in_block(const size_t & idx) const;
				inline size_t size_in_block(const size_t & idx) const;
			};
		}
	}
}
#include <skepu3/cluster/impl/cut_structure.inl>

#endif /* CUT_STRUCTURE_HPP */
