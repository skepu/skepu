/*! \file map.h
 *  \brief Contains a class declaration for the MapPairs skeleton.
 */

#ifndef MAPPAIRS_H
#define MAPPAIRS_H

namespace skepu
{
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
 		 */
		/*!
		 *  \class MapPairs
		 *
		 *  \brief A class representing the MapPairs skeleton.
		 *
		 *  This class defines the MapPairs skeleton, a calculation pattern where a user function is applied to each element of the cartesian product of input vectors.
		 *  Result is a Matrix.
		 */
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		class MapPairs : public SkeletonBase
		{
			// ==========================    Type definitions   ==========================

			using T = typename MapPairsFunc::Ret;
			using F = ConditionalIndexForwarder<MapPairsFunc::indexed, MapPairsFunc::usesPRNG, decltype(&MapPairsFunc::CPU)>;

			// ==========================     Class members     ==========================

			static constexpr size_t outArity = MapPairsFunc::outArity;
			static constexpr size_t numArgs = MapPairsFunc::totalArity - (MapPairsFunc::indexed ? 1 : 0) - (MapPairsFunc::usesPRNG ? 1 : 0) + outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapPairsFunc::ContainerArgs>::value;

			// ==========================    Instance members   ==========================

			CUDAKernel m_cuda_kernel;

			size_t default_size_x;
			size_t default_size_y;
			bool m_in_place = false;

		public:

			static constexpr auto skeletonType = SkeletonType::MapPairs;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = typename MapPairsFunc::ElwiseArgs;
			using ContainerArgs = typename MapPairsFunc::ContainerArgs;
			using UniformArgs = typename MapPairsFunc::UniformArgs;

			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<outArity + Varity, outArity>::type Velwise_indices{};
			static constexpr typename make_pack_indices<outArity + Varity + Harity, outArity + Varity>::type Helwise_indices{};
			static constexpr typename make_pack_indices<outArity + Varity + Harity + anyArity, outArity + Varity + Harity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, outArity + Varity + Harity + anyArity>::type const_indices{};

			// =========================      Constructors      ==========================

			MapPairs(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}

			// =======================  Persistent parameters   ==========================

			void setDefaultSize(size_t x, size_t y = 0)
			{
				this->default_size_x = x;
				this->default_size_y = y;
			}

			void setInPlace(bool inPlace)
			{
				this->m_in_place = inPlace;
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

			// =======================      Call operators      ==========================

			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> typename std::add_lvalue_reference<decltype(get<0>(std::forward<CallArgs>(args)...))>::type
			{
				this->backendDispatch(
					out_indices, Velwise_indices, Helwise_indices, any_indices, const_indices,
					get<0>(std::forward<CallArgs>(args)...).total_rows(),
					get<0>(std::forward<CallArgs>(args)...).total_cols(),
					std::forward<CallArgs>(args)...
				);
				return get<0>(std::forward<CallArgs>(args)...);
			}

		private:

			// ==========================    Implementation     ==========================

			template<size_t... OI, size_t... VEI, size_t... HEI,  size_t... AI, size_t... CI, typename... CallArgs>
			void CPU(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


#ifdef SKEPU_OPENMP

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename ...CallArgs>
			void OMP(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_OPENMP


#ifdef SKEPU_CUDA

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void CUDA(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsSingleThread_CU(size_t deviceID, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsMultiStream_CU(size_t deviceID, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsSingleThreadMultiGPU_CU(size_t numDevices, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsMultiStreamMultiGPU_CU(size_t numDevices, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_CUDA


#ifdef SKEPU_OPENCL

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void CL(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapNumDevices_CL(size_t startIdx, size_t numDevices, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_OPENCL

#ifdef SKEPU_HYBRID

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void Hybrid(size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...>, pack_indices<HEI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_HYBRID

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void backendDispatch(pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t Vsize, size_t Hsize, CallArgs&&... args)
			{
			//	assert(this->m_execPlan != nullptr && this->m_execPlan->isCalibrated());

				if (  disjunction((get<OI>(std::forward<CallArgs>(args)...).total_rows() < Vsize)...)
					|| disjunction((get<OI>(std::forward<CallArgs>(args)...).total_cols() < Hsize)...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction((get<VEI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
					SKEPU_ERROR("Non-matching vertical container sizes");

				if (disjunction((get<HEI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
					SKEPU_ERROR("Non-matching horizontal container sizes");

				this->selectBackend(Vsize + Hsize);

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->Hybrid(Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...)...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CUDA(0, Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...)...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(0, Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...)...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...)...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				default:
					this->CPU(Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...)...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
				}
			}

		}; // class MapPairs

	} // namespace backend

#ifdef SKEPU_MERCURIUM
template<typename Ret, typename... Args>
struct MapPairsImpl : public SeqSkeletonBase
{
	MapPairsImpl(Ret(*)(Args...));

	void setDefaultSize(size_t x, size_t y = 0);

	template<template<class> class Container, typename... CallArgs>
	Container<Ret> &operator()(Container<Ret> &res, CallArgs&&... args);

	template<
		typename Iterator,
		typename... CallArgs,
		REQUIRES_VALUE(is_skepu_iterator<Iterator, Ret>)>
	Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args);

	template<template<class> class Container = Vector, typename... CallArgs>
	Container<Ret> operator()(CallArgs&&... args);
};

template<int Varity, int Harity, typename Ret, typename... Args>
auto inline
MapPairs(Ret(*)(Args...))
-> MapPairsImpl<Ret, Args...>;
#endif // SKEPU_MERCURIUM

} // namespace skepu


#include "impl/mappairs/mappairs_cpu.inl"
#include "impl/mappairs/mappairs_omp.inl"
#include "impl/mappairs/mappairs_cl.inl"
#include "impl/mappairs/mappairs_cu.inl"
#include "impl/mappairs/mappairs_hy.inl"

#endif // MAPPAIRS_H
