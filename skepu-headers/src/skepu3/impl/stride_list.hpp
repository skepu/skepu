#pragma once

namespace skepu
{	
	// For Map skeletons with strides
	
	template<size_t count>
	class StrideList
	{
	public:
		StrideList()
		{
			std::fill(&this->values[0], &this->values[count], 1);
		}
		
		template<typename... S>
		StrideList(S... strides): values{static_cast<int>(strides)...}
		{
			static_assert(sizeof...(S) == count, "Unexpected stride count");
		}

#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		int operator[](size_t i) const
		{
			return this->values[i];
		}

#ifdef SKEPU_CUDA
	__host__ __device__
#endif
		int& operator[](size_t i)
		{
			return this->values[i];
		}
		
	private:
		
		int values[count+1]; // +1 ensures array is never empty
	};
	
}