#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{
	template<int, int, typename, typename...>
	class MapImpl;

	template<int GivenArity, typename Ret, typename... Args>
	MapImpl<resolve_map_arity<GivenArity, Args...>::value, GivenArity, Ret, Args...>
	MapWrapper(std::function<Ret(Args...)> map)
	{
		return MapImpl<resolve_map_arity<GivenArity, Args...>::value, GivenArity, Ret, Args...>(map);
	}

	// For function pointers
	template<int GivenArity = SKEPU_UNSET_ARITY, typename Ret, typename... Args>
	auto Map(Ret(*map)(Args...))
		-> decltype(MapWrapper<GivenArity>((std::function<Ret(Args...)>)map))
	{
		return MapWrapper<GivenArity>((std::function<Ret(Args...)>)map);
	}

	// For lambdas and functors
	template<int GivenArity = SKEPU_UNSET_ARITY, typename T>
	auto Map(T map)
		-> decltype(MapWrapper<GivenArity>(lambda_cast(map)))
	{
		return MapWrapper<GivenArity>(lambda_cast(map));
	}

	template<int InArity, int GivenArity, typename Ret, typename... Args>
	class MapImpl: public SeqSkeletonBase
	{
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr bool randomized = has_random<Args...>::value;
		static constexpr size_t randomCount = get_random_count<Args...>::value;
		static constexpr size_t OutArity = out_size<Ret>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;

		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
		static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
		static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};

		using MapFunc = std::function<Ret(Args...)>;
		using F = ConditionalIndexForwarder<indexed, randomized, MapFunc>;

		// For iterators
		template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
		void apply(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("Native C++ Map: size = " << size);

			auto out = std::make_tuple(get<OI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[OI])...);
			auto in  = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[EI])...);

			if (disjunction((std::get<OI>(out).size() < size)...))
				SKEPU_ERROR("Non-matching output container sizes");

			if (disjunction((std::get<EI-OutArity>(in).size() < size)...))
				SKEPU_ERROR("Non-matching input container sizes");

			SKEPU_TRACE_START_EVENT(trace_handle);

			auto random = this->template prepareRandom<randomCount>(size);
			for (size_t i = 0; i < size; ++i)
			{
				auto index = std::get<0>(out).getIndex();
				auto res = F::forward(mapFunc, index, random,
					*std::get<EI-OutArity>(in)++...,
					get<AI>(std::forward<CallArgs>(args)...).hostProxy(typename pack_element<AI-OutArity + (indexed ? 1 : 0) + (randomized ? 1 : 0), typename proxy_tag<Args>::type...>::type{}, index)...,
					get<CI>(std::forward<CallArgs>(args)...)...
				);
				SKEPU_VARIADIC_RETURN(*std::get<OI>(out)++..., res);
			}
		}


	public:

		template<typename... CallArgs>
		auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(
				get<0>(std::forward<CallArgs>(args)...).size() / this->m_strides[0],
				out_indices, elwise_indices, any_indices, const_indices,
				std::forward<CallArgs>(args)...
			);
			return get<0>(std::forward<CallArgs>(args)...);
		}

		template<typename Iterator, typename... CallArgs, REQUIRES_VALUE(is_skepu_iterator<Iterator, Ret>)>
		Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs-1, "Number of arguments not matching Map function");
			this->apply(
				(res_end - res) / this->m_strides[0],
				out_indices, elwise_indices, any_indices, const_indices,
				res, std::forward<CallArgs>(args)...
			);
			return res;
		}

		template<typename... S>
		void setStride(S... strides)
		{
			this->m_strides = StrideList<OutArity + InArity>(strides...);
		}

//	private:
		MapFunc mapFunc;
		MapImpl(MapFunc map): mapFunc(map) {}

		StrideList<OutArity + InArity> m_strides{};

	//	friend MapImpl<InArity, GivenArity, Ret, Args...> MapWrapper<GivenArity, Ret, Args...>(MapFunc);

	}; // end class MapImpl

} // end namespace skepu
