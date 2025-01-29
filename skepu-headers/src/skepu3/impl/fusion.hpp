
#pragma once

#include <utility>
#include <tuple>
#include <cstddef>
#include <type_traits>

#if (__cplusplus >= 201703L)

namespace skepu
{	
	
  // https://stackoverflow.com/a/32166787

	template <typename T>
	auto explode(T&& t, char)
	{
		return std::forward_as_tuple(std::forward<T>(t));
	}

	template <typename T, std::size_t I = std::tuple_size<std::decay_t<T>>{}>
	auto explode(T&& t, int);

	template <typename T, std::size_t... Is>
	auto explode(T&& t, std::index_sequence<Is...>)
	{
		return std::tuple_cat(explode(std::get<Is>(std::forward<T>(t)), 0)...);
	}

	template <typename T, std::size_t I>
	auto explode(T&& t, int)
	{
		return explode(std::forward<T>(t), std::make_index_sequence<I>{});
	}

	template <typename T, std::size_t... Is>
	auto decay_tuple(T&& t, std::index_sequence<Is...>)
	{
		return std::make_tuple(std::get<Is>(std::forward<T>(t))...);
	}

	template <typename T>
	auto decay_tuple(T&& t)
	{
		return decay_tuple(std::forward<T>(t), std::make_index_sequence<std::tuple_size<std::decay_t<T>>{}>{});
	}

	template <typename T, std::size_t... Is>
	auto merge_tuple(T&& t, std::index_sequence<Is...>)
	{
		return decay_tuple(std::tuple_cat(explode(std::get<Is>(std::forward<T>(t)), 0)...));
	}

	template <typename T>
	auto merge_tuple(T&& t)
	{
		return merge_tuple(std::forward<T>(t), std::make_index_sequence<std::tuple_size<std::decay_t<T>>{}>{});
	}
  
  
  
  
  
  
  
  // Map >> Map
  template<
    int InArityLHS, int GivenArityLHS, typename RetLHS, typename... ArgsLHS,
    int InArityRHS, int GivenArityRHS, typename RetRHS, typename... ArgsRHS
  >
  MapImpl<InArityLHS, GivenArityLHS, RetRHS, ArgsLHS...>
  operator >>(
    MapImpl<InArityLHS, GivenArityLHS, RetLHS, ArgsLHS...> &&lhs,
    MapImpl<InArityRHS, GivenArityRHS, RetRHS, ArgsRHS...> &&rhs
  )
  {
		auto new_uf = [=]
    (ArgsLHS... args) -> RetRHS
    {
      RetLHS temp = lhs.mapFunc(std::forward<ArgsLHS>(args)...);
      if constexpr (skepu_variadic_return::is_tuple<RetRHS>::value)
        return std::apply(rhs.mapFunc, temp);
      else
        return rhs.mapFunc(temp);
    };
    
    return Map(new_uf);
  }
  
  
	
	
  // Map >> MapReduce
  template<
    int InArityLHS, int GivenArityLHS, typename RetLHS,             typename... ArgsLHS,
    int InArityRHS, int GivenArityRHS, typename RetRHS, typename T, typename... ArgsRHS
  >
  MapReduceImpl<InArityLHS, GivenArityLHS, RetRHS, T, ArgsLHS...>
  operator >>(
    MapImpl<InArityLHS, GivenArityLHS, RetLHS, ArgsLHS...> &&lhs,
    MapReduceImpl<InArityRHS, GivenArityRHS, RetRHS, T, ArgsRHS...> &&rhs
  )
  {
    auto new_uf = [=]
    (ArgsLHS... args) -> RetRHS
    {
      RetLHS temp = lhs.mapFunc(std::forward<ArgsLHS>(args)...);
      return rhs.mapFunc(temp);
    };
    
    return MapReduce(new_uf, rhs.redFunc);
  }
  
  
  
  // Map >> Reduce
  template<
    int InArityLHS, int GivenArityLHS, typename RetLHS, typename... ArgsLHS,
    typename T
  >
  MapReduceImpl<InArityLHS, GivenArityLHS, RetLHS, T, ArgsLHS...>
  operator >>(
    MapImpl<InArityLHS, GivenArityLHS, RetLHS, ArgsLHS...> &&lhs,
    impl::Reduce1D<T> &&rhs
  )
  {
    return MapReduce(lhs.mapFunc, rhs.redFunc);
  }
  
  
  // MapPairs >> Reduce
  template<
    int Varity, int Harity, typename Ret, typename Red, typename... Args
  >
  MapPairsReduceImpl<Varity, Harity, Ret, Red, Args...>
  operator >>(
    MapPairsImpl<Varity, Harity, Ret, Args...> &&lhs,
    impl::Reduce1D<Red> &&rhs
  )
  {
    return MapPairsReduce(lhs.mapPairsFunc, rhs.redFunc);
  }
	
	
	
	
  
  
  // MapPairs >> Map
  template<
    int Varity, int Harity, typename RetLHS, typename... ArgsLHS,
    int InArityRHS, int GivenArityRHS, typename RetRHS, typename... ArgsRHS
  >
  MapPairsImpl<Varity, Harity, RetRHS, ArgsLHS...>
  operator >>(
    MapPairsImpl<Varity, Harity, RetLHS, ArgsLHS...> &&lhs,
    MapImpl<InArityRHS, GivenArityRHS, RetRHS, ArgsRHS...> &&rhs
  )
  {
    auto new_uf = [=]
    (ArgsLHS... args) -> RetRHS
    {
      RetLHS temp = lhs.mapPairsFunc(std::forward<ArgsLHS>(args)...);
      if constexpr (skepu_variadic_return::is_tuple<RetRHS>::value)
        return std::apply(rhs.mapFunc, temp);
      else
        return rhs.mapFunc(temp);
    };
    
    return MapPairs(new_uf);
  }
  
  
  /*
  // MapOverlap >> MapOverlap
  template<
    typename RetLHS, typename... ArgsLHS,
    typename RetRHS, typename... ArgsRHS
  >
  MapOverlapImpl<RetRHS, ArgsLHS...>
  operator >>(
    impl::MapOverlap1D<RetLHS, ArgsLHS...> &&lhs,
    impl::MapOverlap1D<RetRHS, T, ArgsRHS...> &&rhs
  )
  {
    auto new_uf = [=]
    (ArgsLHS... args) -> RetRHS
    {
      RetRHS temp = lhs.mapFunc(std::forward<ArgsLHS>(args)...);
      return rhs.mapFunc(temp);
    };
    
    return MapReduce(new_uf, rhs.redFunc);
  }
  */
  
  
  
  
  
  // Map || Map
  template<
    int InArityLHS, int GivenArityLHS, typename RetLHS, typename... ArgsLHS,
    int InArityRHS, int GivenArityRHS, typename RetRHS, typename... ArgsRHS
  >
  MapImpl<
    InArityLHS + InArityRHS,
    GivenArityLHS, // irrelevant
    decltype(merge_tuple(skepu::multiple<RetRHS, RetLHS>{})),
    ArgsLHS..., ArgsRHS...
  >
  operator ||(
    MapImpl<InArityLHS, GivenArityLHS, RetLHS, ArgsLHS...> &&lhs,
    MapImpl<InArityRHS, GivenArityRHS, RetRHS, ArgsRHS...> &&rhs
  )
  {
    auto new_uf = [=]
    (ArgsLHS... argsLHS, ArgsRHS... argsRHS)
    -> decltype(merge_tuple(skepu::multiple<RetRHS, RetLHS>{}))
    {
      skepu::multiple<RetLHS, RetRHS> temp;
      std::get<0>(temp) = lhs.mapFunc(std::forward<ArgsLHS>(argsLHS)...);
      std::get<1>(temp) = rhs.mapFunc(std::forward<ArgsRHS>(argsRHS)...);
      return merge_tuple(temp);
    };
    
    return Map(new_uf);
  }
  
  
  
  // Reduce || Reduce
  template<typename TLHS, typename TRHS>
  impl::Reduce1D<skepu::multiple<TRHS, TLHS>>
  operator ||(
    impl::Reduce1D<TLHS> &&lhs,
    impl::Reduce1D<TRHS> &&rhs
  )
  {
    auto new_uf = [=]
    (skepu::multiple<TLHS, TRHS> a, skepu::multiple<TLHS, TRHS> b)
    -> decltype(merge_tuple(skepu::multiple<TRHS, TLHS>{}))
    {
      skepu::multiple<TLHS, TRHS> temp;
      std::get<0>(temp) = lhs.redFunc(std::get<0>(a), std::get<0>(b));
      std::get<1>(temp) = rhs.redFunc(std::get<1>(a), std::get<1>(b));
      return merge_tuple(temp);
    };
    
    return Reduce(new_uf);
  }
  
  
  
  // MapReduce || MapReduce
  template<
    int InArityLHS, int GivenArityLHS, typename RetLHS, typename RedLHS, typename... ArgsLHS,
    int InArityRHS, int GivenArityRHS, typename RetRHS, typename RedRHS, typename... ArgsRHS
  >
  MapReduceImpl<InArityLHS + InArityRHS, GivenArityLHS, skepu::multiple<RetRHS, RetLHS>, RedLHS, ArgsLHS..., ArgsRHS...>
  operator ||(
    MapReduceImpl<InArityLHS, GivenArityLHS, RetLHS, RedLHS, ArgsLHS...> &&lhs,
    MapReduceImpl<InArityRHS, GivenArityRHS, RetRHS, RedRHS, ArgsRHS...> &&rhs
  )
  {
    static_assert(std::is_same<decltype(rhs.redFunc), decltype(lhs.redFunc)>::value, "Non-matching reduction functions in MapReduce || MapReduce fusion");
    auto new_uf = [=]
    (ArgsLHS... argsLHS, ArgsRHS... argsRHS) -> skepu::multiple<RetRHS, RetLHS>
    {
      skepu::multiple<RetLHS, RetRHS> temp;
      std::get<0>(temp) = lhs.mapFunc(std::forward<ArgsLHS>(argsLHS)...);
      std::get<1>(temp) = rhs.mapFunc(std::forward<ArgsRHS>(argsRHS)...);
      return temp;
    };
    
    return MapReduce(new_uf, lhs.redFunc);
  }
  
  
  
  // MapPairs || MapPairs
  template<
    int VarityLHS, int HarityLHS, typename RetLHS, typename... ArgsLHS,
    int VarityRHS, int HarityRHS, typename RetRHS, typename... ArgsRHS
  >
  MapPairsImpl<
		VarityLHS + VarityRHS,
		HarityLHS + HarityRHS,
		skepu::multiple<RetRHS, RetLHS>,
		ArgsLHS..., ArgsRHS...
	>
  operator ||(
    MapPairsImpl<VarityLHS, HarityLHS, RetLHS, ArgsLHS...> &&lhs,
    MapPairsImpl<VarityRHS, HarityRHS, RetRHS, ArgsRHS...> &&rhs
  )
  {
    auto new_uf = [=]
    (ArgsLHS... argsLHS, ArgsRHS... argsRHS) -> skepu::multiple<RetRHS, RetLHS>
    {
      skepu::multiple<RetLHS, RetRHS> temp;
      std::get<0>(temp) = lhs.mapPairsFunc(std::forward<ArgsLHS>(argsLHS)...);
      std::get<1>(temp) = rhs.mapPairsFunc(std::forward<ArgsRHS>(argsRHS)...);
      return temp;
    };
    
    return MapPairs<VarityLHS + VarityRHS, HarityLHS + HarityRHS>(new_uf);
  }
	
	
	
	// Map ^ N
  template<
    int InArity, int GivenArity, typename Ret, typename... Args
  >
  MapImpl<InArity, GivenArity, Ret, Args...>
  operator ^(
    MapImpl<InArity, GivenArity, Ret, Args...> &&skel, size_t N
  )
  {
    auto new_uf = [=]
    (Args... args) -> Ret
    {
			Ret temp = skel.mapFunc(std::forward<Args>(args)...);
			for (size_t i = 1; i < N; ++i)
				if constexpr (skepu_variadic_return::is_tuple<Ret>::value)
	        temp = std::apply(skel.mapFunc, temp);
	      else
	        temp = skel.mapFunc(temp);
			return temp;
    };
    
    return Map(new_uf);
  }
  
	
}


#endif
