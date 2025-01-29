#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<int, int, typename, typename...>
	class MapPairsImpl;
	
	template<int Varity = 1, int Harity = 1, typename Ret, typename... Args>
	MapPairsImpl<Varity, Harity, Ret, Args...> MapPairsWrapper(std::function<Ret(Args...)> mapPairs)
	{
		return MapPairsImpl<Varity, Harity, Ret, Args...>(mapPairs);
	}
	
	// For function pointers
	template<int Varity = 1, int Harity = 1, typename Ret, typename... Args>
	MapPairsImpl<Varity, Harity, Ret, Args...> MapPairs(Ret(*mapPairs)(Args...))
	{
		return MapPairsWrapper<Varity, Harity>((std::function<Ret(Args...)>)mapPairs);
	}
	
	// For lambdas and functors
	template<int Varity = 1, int Harity = 1, typename T>
	auto MapPairs(T mapPairs) -> decltype(MapPairsWrapper<Varity, Harity>(lambda_cast(mapPairs)))
	{
		return MapPairsWrapper<Varity, Harity>(lambda_cast(mapPairs));
	}
	
	template<int Varity, int Harity, typename Ret, typename... Args>
	class MapPairsImpl: public SeqSkeletonBase
	{
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr bool randomized = has_random<Args...>::value;
		static constexpr size_t randomCount = get_random_count<Args...>::value;
		static constexpr size_t OutArity = out_size<Ret>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
		static constexpr typename make_pack_indices<OutArity + Varity, OutArity>::type Velwise_indices{};
		static constexpr typename make_pack_indices<OutArity + Varity + Harity, OutArity + Varity>::type Helwise_indices{};
		static constexpr typename make_pack_indices<OutArity + Varity + Harity + anyCont, OutArity + Varity + Harity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, OutArity + Varity + Harity + anyCont>::type const_indices{};
		
		using MapPairsFunc = std::function<Ret(Args...)>;
		using F = ConditionalIndexForwarder<indexed, randomized, MapPairsFunc>;
		
		// For iterators
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void apply(pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, size_t Vsize, size_t Hsize, CallArgs&&... args)
		{
			if ( disjunction((get<OI>(std::forward<CallArgs>(args)...).total_rows() < Vsize)...)
				|| disjunction((get<OI>(std::forward<CallArgs>(args)...).total_cols() < Hsize)...))
				SKEPU_ERROR("Non-matching output container sizes");
			
			if (disjunction((get<VEI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
				SKEPU_ERROR("Non-matching vertical container sizes");
			
			if (disjunction((get<HEI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
				SKEPU_ERROR("Non-matching horizontal container sizes");
			
			auto out = std::forward_as_tuple(get<OI>(std::forward<CallArgs>(args)...)...);
			auto VelwiseIterators = std::make_tuple(get<VEI>(std::forward<CallArgs>(args)...).begin()...);
			auto HelwiseIterators = std::make_tuple(get<HEI>(std::forward<CallArgs>(args)...).begin()...);
			
			auto random = this->template prepareRandom<randomCount>(Vsize * Hsize);
			
			for (size_t i = 0; i < Vsize; ++i)
			{
				for (size_t j = 0; j < Hsize; ++j)
				{
					auto index = Index2D { i, j };
					auto res = F::forward(mapPairsFunc, index, random,
						std::get<VEI-OutArity>(VelwiseIterators)(i)...,
						std::get<HEI-Varity-OutArity>(HelwiseIterators)(j)...,
						get<AI>(std::forward<CallArgs>(args)...).hostProxy(typename pack_element<AI-OutArity+(indexed ? 1 : 0), typename proxy_tag<Args>::type...>::type{}, index)...,
						get<CI>(std::forward<CallArgs>(args)...)...);
					SKEPU_VARIADIC_RETURN_INPLACE(this->m_in_place, std::get<OI>(out)(i, j)..., res);
				}
			}
			
		}
		
		
	public:
		
		void setDefaultSize(size_t x, size_t y)
		{
			this->default_size_x = x;
			this->default_size_y = y;
		}
		
		void setInPlace(bool inPlace)
		{
			this->m_in_place = inPlace;
		}
		
		template<typename... CallArgs>
		auto operator()(CallArgs&&... args) -> typename std::add_lvalue_reference<decltype(get<0>(std::forward<CallArgs>(args)...))>::type
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(
				out_indices, Velwise_indices, Helwise_indices, any_indices, const_indices, get<0>(std::forward<CallArgs>(args)...).total_rows(),
				get<0>(std::forward<CallArgs>(args)...).total_cols(),
				std::forward<CallArgs>(args)...
			);
			return get<0>(std::forward<CallArgs>(args)...);
		}
		
//	private:
		MapPairsFunc mapPairsFunc;
		MapPairsImpl(MapPairsFunc mapPairs): mapPairsFunc(mapPairs) {}
		
		size_t default_size_x = 1;
		size_t default_size_y = 1;
		bool m_in_place = false;
		
		friend MapPairsImpl<Varity, Harity, Ret, Args...> MapPairsWrapper<Varity, Harity, Ret, Args...>(MapPairsFunc);
		
	}; // end class MapImpl
	
} // end namespace skepu
