/*! \file mapoverlap.h
 *  \brief Contains a class declaration for the MapOverlap skeleton.
 */

#ifndef MAPOVERLAP_PAR_H
#define MAPOVERLAP_PAR_H

#include "skepu3/impl/region.hpp"
#include "skepu3/mapoverlap_base.hpp"

namespace skepu
{

	namespace backend
	{
        template <typename MapOverlapFunc, SkeletonType skeletonType>
        class MapOverlapPar: public impl::MapOverlapBase<typename region_type<typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type>::type>
        {
        protected:
            using Ret = typename MapOverlapFunc::Ret;
            static constexpr bool isPool = MapOverlapFunc::isPool;
            using T = typename region_type<typename parameter_type<(MapOverlapFunc::indexed ? 1 : 0) + (MapOverlapFunc::usesPRNG ? 1 : 0), decltype(&MapOverlapFunc::CPU)>::type>::type;
            using F = ConditionalIndexForwarder<MapOverlapFunc::indexed, MapOverlapFunc::usesPRNG, decltype(&MapOverlapFunc::CPU)>;
            size_t expectedInputSize(size_t size, size_t dim) const
			{
				if (isPool) return this->m_overlap[dim] + this->m_strides[dim] * (size - 1);
				else if (this->m_edge == Edge::None) return size + 2 * this->m_overlap[dim];
				else return size;
			}

        public:
            using ResultArg = std::tuple<T>;
            using ElwiseArgs = std::tuple<T>;
            using ContainerArgs = typename MapOverlapFunc::ContainerArgs;
            using UniformArgs = typename MapOverlapFunc::UniformArgs;
            static constexpr bool prefers_matrix = false;

            static constexpr size_t InArity = 1;
            static constexpr size_t OutArity = MapOverlapFunc::outArity;
			static constexpr size_t numArgs = MapOverlapFunc::totalArity - (MapOverlapFunc::indexed ? 1 : 0) - (MapOverlapFunc::usesPRNG ? 1 : 0) + OutArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

            static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
            static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
            static constexpr typename make_pack_indices<InArity + anyArity + OutArity, InArity + OutArity>::type any_indices{};
            static constexpr typename make_pack_indices<numArgs, InArity + anyArity + OutArity>::type const_indices{};

        }; // MapOverlapPar

    } // backend

#ifdef SKEPU_MERCURIUM

	template<typename Ret, typename Arg1, typename... Args>
	class MapOverlapImpl: public SeqSkeletonBase
	{
	protected:
		using T =
			typename std::remove_const<typename std::remove_pointer<Arg1>::type>::type;
		using MapFunc1D = std::function<Ret(int, size_t, Arg1, Args...)>;
		using MapFunc2D = std::function<Ret(int, int, size_t, Arg1, Args...)>;

	public:
		void setOverlapMode(Overlap mode);
		void setEdgeMode(Edge mode);
		void setPad(T pad);

		MapOverlapImpl(MapFunc1D map);
		MapOverlapImpl(MapFunc2D map);

		void setOverlap(size_t o);
		void setOverlap(size_t y, size_t x);
		size_t getOverlap() const;
		std::pair<size_t, size_t> getOverlap() const;

		template<
			template<class> class Container,
			size_t... AI,
			size_t... CI,
			typename... CallArgs>
		Container<Ret> &helper(
			Container<Ret> &res,
			Container<T> &arg,
			pack_indices<AI...>,
			pack_indices<CI...>,
			CallArgs&&... args);

		template<template<class> class Container, typename... CallArgs>
		Container<Ret> &operator()(
			Container<Ret> &res, Container<T>& arg, CallArgs&&... args);

		template<size_t... AI, size_t... CI, typename... CallArgs>
		void apply_colwise(
			skepu::Matrix<Ret>& res,
			skepu::Matrix<T>& arg,
			pack_indices<AI...>,
			pack_indices<CI...>,
			CallArgs&&... args);

		template<size_t... AI, size_t... CI, typename... CallArgs>
		void apply_rowwise(
			skepu::Matrix<Ret>& res,
			skepu::Matrix<T>& arg,
			pack_indices<AI...>,
			pack_indices<CI...>,
			CallArgs&&... args);

		template<typename... CallArgs>
		Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args);

		template<size_t... AI, size_t... CI, typename... CallArgs>
		void apply_helper(
			Matrix<Ret> &res,
			Matrix<T> &arg,
			pack_indices<AI...>,
			pack_indices<CI...>,
			CallArgs&&... args);

		template<typename... CallArgs>
		Matrix<Ret> &operator()(Matrix<Ret> &res, Matrix<T>& arg, CallArgs&&... args);

	};

	template<typename Ret, typename Arg1, typename ... ArgRest>
	auto inline
	MapOverlap(Ret (*)(int, size_t, Arg1, ArgRest...))
	-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

	template<typename Ret, typename Arg1, typename ... ArgRest>
	auto inline
	MapOverlap(std::function<Ret(int, size_t, Arg1, ArgRest...)>)
	-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

	template<typename Ret, typename Arg1, typename ... ArgRest>
	auto inline
	MapOverlap(Ret (*)(int, int, size_t, Arg1, ArgRest...))
	-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

	template<typename Ret, typename Arg1, typename ... ArgRest>
	auto inline
	MapOverlap(std::function<Ret(int, int, size_t, Arg1, ArgRest...)>)
	-> MapOverlapImpl<Ret, Arg1, ArgRest...>;

	template<typename T>
	auto inline
	MapOverlap(T op)
	-> decltype(MapOverlap(lambda_cast(op)));

#endif // SKEPU_MERCURIUM

} // skepu

#endif // MAPOVERLAP_PAR_H