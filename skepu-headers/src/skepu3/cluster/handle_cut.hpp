#ifndef HANDLE_CUT_HPP
#define HANDLE_CUT_HPP

#include <skepu3/cluster/index.hpp>
#include <starpu.h>
#include <tuple>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			/**
			 * @brief A handle_cut represents a part of a handle that
			 * should be scheduled in some way
			 */
			struct handle_cut
			{
				starpu_data_handle_t & handle;
				Offset2D local_offset;
				Size2D local_size;
				inline handle_cut(starpu_data_handle_t & h,
				                  const Offset2D & offset,
				                  const Size2D & size);
			};
		} // helpers
	} // cluster
} // skepu


#include <skepu3/cluster/impl/handle_cut.inl>

#endif /* HANDLE_CUT_HPP */
