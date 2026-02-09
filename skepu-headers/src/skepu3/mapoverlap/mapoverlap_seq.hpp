#pragma once

#include "skepu3/mapoverlap_base.hpp"

namespace skepu
{
    namespace impl
    {
        template <template <typename> class PoolType, typename Ret, typename... Args>
        class MapOverlapSeq: public MapOverlapBase<typename region_type_ext<Args...>::type>
        {
        protected:
            static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr bool randomized = has_random<Args...>::value;
			static constexpr size_t randomCount = get_random_count<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;

			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};

			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, randomized, MapFunc>;
			using RegionType = typename pack_element<(indexed ? 1 : 0) + (randomized ? 1 : 0), Args...>::type;
			using T = typename region_type<RegionType>::type;
			static constexpr bool isPool = std::is_same<typename std::decay<RegionType>::type, PoolType<T>>::value;
            size_t expectedInputSize(size_t size, size_t dim) const
			{
				if (isPool) return this->m_overlap[dim] + this->m_strides[dim] * (size - 1);
				else if (this->m_edge == Edge::None) return size + 2 * this->m_overlap[dim];
				else return size;
			}

            bool isInputSizeValid(size_t out_size, size_t in_size, size_t dim) const
			{
                // input must be greater than or equal to the area that will be traversed
                // to obtain the outputs.
                if (this->isPool)
                    return in_size >= this->getSmallestAllowedInputSizePool(out_size, dim);
                
                // input must be sized exactly so that there is no edge handling
                else if (this->m_edge == skepu::Edge::None)
                    return in_size == this->getAllowedInputSizeNone(out_size, dim);

                // it doesn't matter what the input size is
                return true;
			}
        };

			

    } // impl


    // The following 4 functions are what the user calls when constructing
    // a sequential MapOverlap or MapPool instance
    
    // For function pointers
    template<typename Ret, typename... Args>
	auto MapOverlap(Ret(*mapo)(Args...)) -> decltype(MapOverlapWrapper((std::function<Ret(Args...)>)mapo))
	{
		return MapOverlapWrapper((std::function<Ret(Args...)>)mapo);
	}

	// For lambdas and functors
	template<typename T>
	auto MapOverlap(T mapo) -> decltype(MapOverlapWrapper(lambda_cast(mapo)))
	{
		return MapOverlapWrapper(lambda_cast(mapo));
	}

	// For function pointers
	template<typename Ret, typename... Args>
	auto MapPool(Ret(*mapo)(Args...)) -> decltype(MapPoolWrapper((std::function<Ret(Args...)>)mapo))
	{
		return MapPoolWrapper((std::function<Ret(Args...)>)mapo);
	}

	// For lambdas and functors
	template<typename T>
	auto MapPool(T mapo) -> decltype(MapPoolWrapper(lambda_cast(mapo)))
	{
		return MapPoolWrapper(lambda_cast(mapo));
	}

} // skepu