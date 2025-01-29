#ifndef STARPU_VAR_HPP
#define STARPU_VAR_HPP

#include <starpu.h>

namespace skepu
{
	namespace cluster
	{
		template<typename T>
		class starpu_var {
		private:
			uintptr_t data;
			bool initialized = true;
		public:
			starpu_data_handle_t handle;
			starpu_var() : initialized { false } {};
			starpu_var(const starpu_var&) = delete;
			starpu_var<T>& operator=(const starpu_var&) = delete;
			starpu_var(starpu_var<T>&& other) noexcept
			{
				// This is so ugly, please find a better way and destroy this
				// code
				data = other.data;
				handle = other.handle;
				initialized = true;
				other.data = (uintptr_t)NULL;
				other.initialized = false;
			}

			starpu_var(size_t owner);
			~starpu_var();
			void broadcast();
			T get();
		};
	}
}

#include "skepu3/cluster/impl/starpu_var.inl"
#endif /* STARPU_VAR_HPP */
