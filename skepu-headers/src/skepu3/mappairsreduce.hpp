#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<int, int, typename, typename, typename...>
	class MapPairsReduceImpl;
	
	template<int Varity = 1, int Harity = 1, typename RetType, typename RedType, typename... Args>
	MapPairsReduceImpl<Varity, Harity, RetType, RedType, Args...> MapPairsReduceWrapper(std::function<RetType(Args...)> mapPairs, std::function<RedType(RedType, RedType)> red)
	{
		return MapPairsReduceImpl<Varity, Harity, RetType, RedType, Args...>(mapPairs, red);
	}
	
	// For function pointers
	template<int Varity = 1, int Harity = 1, typename RetType, typename RedType, typename... Args>
	MapPairsReduceImpl<Varity, Harity, RetType, RedType, Args...> MapPairsReduce(RetType(*mapPairs)(Args...), RedType(*red)(RedType, RedType))
	{
		return MapPairsReduceWrapper<Varity, Harity>((std::function<RetType(Args...)>)mapPairs, (std::function<RedType(RedType, RedType)>)red);
	}
	
	// For lambdas and functors
	template<int Varity = 1, int Harity = 1, typename T1, typename T2>
	auto MapPairsReduce(T1 mapPairs, T2 red) -> decltype(MapPairsReduceWrapper<Varity, Harity>(lambda_cast(mapPairs), lambda_cast(red)))
	{
		return MapPairsReduceWrapper<Varity, Harity>(lambda_cast(mapPairs), lambda_cast(red));
	}
	
	template<int Varity, int Harity, typename RetType, typename RedType, typename... Args>
	class MapPairsReduceImpl: public SeqSkeletonBase
	{
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr bool randomized = has_random<Args...>::value;
		static constexpr size_t randomCount = get_random_count<Args...>::value;
		static constexpr size_t OutArity = out_size<RetType>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
		static constexpr typename make_pack_indices<OutArity + Varity, OutArity>::type Velwise_indices{};
		static constexpr typename make_pack_indices<OutArity + Varity + Harity, OutArity + Varity>::type Helwise_indices{};
		static constexpr typename make_pack_indices<OutArity + Varity + Harity + anyCont, OutArity + Varity + Harity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, OutArity + Varity + Harity + anyCont>::type const_indices{};
		
		using MapPairsFunc = std::function<RetType(Args...)>;
		using RedFunc = std::function<RedType(RedType, RedType)>;
		using F = ConditionalIndexForwarder<indexed, randomized, MapPairsFunc>;
		
		// For iterators
		template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
		void apply(pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, size_t size, CallArgs&&... args)
		{
			size_t Vsize = get<0>(get<VEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_y);
			size_t Hsize = get<0>(get<HEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_x);
			
			if (disjunction((get<VEI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
				SKEPU_ERROR("Non-matching input container sizes");
			
			if (disjunction((get<HEI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
				SKEPU_ERROR("Non-matching input container sizes");
				
			if  ((this->m_mode == ReduceMode::RowWise && disjunction((get<OI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
				|| (this->m_mode == ReduceMode::ColWise && disjunction((get<OI>(std::forward<CallArgs>(args)...).size() < Hsize)...)))
				SKEPU_ERROR("Non-matching output container size");
			
			auto VelwiseIterators = std::make_tuple(get<VEI>(std::forward<CallArgs>(args)...).begin()...);
			auto HelwiseIterators = std::make_tuple(get<HEI>(std::forward<CallArgs>(args)...).begin()...);
			
			auto random = this->template prepareRandom<randomCount>(Vsize * Hsize);
			
			if (this->m_mode == ReduceMode::RowWise)
				for (size_t i = 0; i < Vsize; ++i)
				{
					pack_expand((get<OI>(std::forward<CallArgs>(args)...)(i) = get_or_return<OI>(this->m_start), 0)...);
					for (size_t j = 0; j < Hsize; ++j)
					{
						Index2D index{ i, j };
						RetType temp = F::forward(mapPairsFunc, index, random,
							std::get<VEI-OutArity>(VelwiseIterators)(i)...,
							std::get<HEI-OutArity-Varity>(HelwiseIterators)(j)...,
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
							get<CI>(std::forward<CallArgs>(args)...)...
						);
						pack_expand((get<OI>(std::forward<CallArgs>(args)...)(i) = redFunc(get<OI>(std::forward<CallArgs>(args)...)(i), get_or_return<OI>(temp)), 0)...);
					}
				}
			else if (this->m_mode == ReduceMode::ColWise)
				for (size_t j = 0; j < Hsize; ++j)
				{
					pack_expand((get<OI>(std::forward<CallArgs>(args)...)(j) = get_or_return<OI>(this->m_start), 0)...);
					for (size_t i = 0; i < Vsize; ++i)
					{
						Index2D index{ i, j };
						RetType temp = F::forward(mapPairsFunc, index, random,
							std::get<VEI-OutArity>(VelwiseIterators)(i)...,
							std::get<HEI-OutArity-Varity>(HelwiseIterators)(j)...,
							get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
							get<CI>(std::forward<CallArgs>(args)...)...
						);
						pack_expand((get<OI>(std::forward<CallArgs>(args)...)(j) = redFunc(get<OI>(std::forward<CallArgs>(args)...)(j), get_or_return<OI>(temp)), 0)...);
					}
				}
			
		}
		
		
	public:
		
		void setStartValue(RetType val)
		{
			this->m_start = val;
		}
		
		void setReduceMode(ReduceMode mode)
		{
			this->m_mode = mode;
		}
		
		void setDefaultSize(size_t y, size_t x)
		{
			this->default_size_x = x;
			this->default_size_y = y;
		}
		
		template<typename... CallArgs>
		auto operator()(CallArgs&&... args) -> typename std::add_lvalue_reference<decltype(get<0>(std::forward<CallArgs>(args)...))>::type
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(out_indices, Velwise_indices, Helwise_indices, any_indices, const_indices, get<0>(std::forward<CallArgs>(args)...).size(), std::forward<CallArgs>(args)...);
			return get<0>(std::forward<CallArgs>(args)...);
		}
		
	private:
		MapPairsFunc mapPairsFunc;
		RedFunc redFunc;
		MapPairsReduceImpl(MapPairsFunc mapPairs, RedFunc red): mapPairsFunc(mapPairs), redFunc(red) {}
		
		ReduceMode m_mode = ReduceMode::RowWise;
		RetType m_start{};
		size_t default_size_x = 1;
		size_t default_size_y = 1;
		
		friend MapPairsReduceImpl<Varity, Harity, RetType, RedType, Args...> MapPairsReduceWrapper<Varity, Harity, RetType, RedType, Args...>(MapPairsFunc, RedFunc);
		
	}; // end class MapImpl
	
} // end namespace skepu
