#ifndef HANDLE_CUT_INL
#define HANDLE_CUT_INL

#include <skepu3/cluster/starpu_matrix_container.hpp>

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			inline handle_cut
			::handle_cut(starpu_data_handle_t & h,
			             const Offset2D & offset,
			             const Size2D & size)
				: handle { h },
				  local_offset { offset },
				  local_size { size } {}
		}
	}
}

#endif /* HANDLE_CUT_INL */
