#ifndef HELPERS_HPP
#define HELPERS_HPP



#include <skepu3/impl/meta_helpers.hpp>
#include <starpu.h>
#include <tuple>

// Various helpers for the cluster implementation

namespace skepu
{
	namespace cluster
	{
		namespace helpers
		{
			template<typename Ret, typename Tuple>
			Ret unpack_tuple(const Tuple & t);

			template<size_t... Is>
			void set_codelet_read_only_modes(pack_indices<Is...>, starpu_codelet& cl);


			template<typename T, size_t I>
			typename std::tuple_element<I,T>::type* extract_handle(void **&& buffers)
			{
				using ContainerT = typename std::tuple_element<I,T>::type;
				auto r = buffers[I];
				return reinterpret_cast<ContainerT*>(STARPU_MATRIX_GET_PTR(r));
			}


			template<typename Containers, size_t I, size_t offset>
			typename std::tuple_element<I-offset,Containers>::type
			extract_container(void **&& buffers)
			{
				using C = typename std::tuple_element<I-offset,Containers>::type;
				using T = typename C::value_type;

				return C(reinterpret_cast<T*>(
					         STARPU_VECTOR_GET_PTR(buffers[I])),
				         STARPU_VECTOR_GET_NX(buffers[I]));
			}

			template<typename... Args>
			void extract_constants(void* & args, Args&... res_args)
			{
				if (args) {
					starpu_codelet_unpack_args(args, &res_args...);
				}
			}

		}
	}
}

#include <skepu3/cluster/impl/helpers.inl>

#endif /* HELPERS_HPP */
