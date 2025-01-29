/*! \file mappairsreduce.h
 *  \brief Contains a class declaration for the MapPairsReduce skeleton.
 */

#ifndef MAPPAIRSREDUCE_H
#define MAPPAIRSREDUCE_H

namespace skepu
{
	namespace backend
	{
		/*!
		*  \ingroup skeletons
		*/
		/*!
		*  \class MapPairsReduce
		*
		*  \brief A class representing the MapPairsReduce skeleton.
		*/
		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		class MapPairsReduce : public SkeletonBase
		{
		public:
			MapPairsReduce(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}

			static constexpr auto skeletonType = SkeletonType::MapPairsReduce;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = typename MapPairsFunc::ElwiseArgs;
			using ContainerArgs = typename MapPairsFunc::ContainerArgs;
			using UniformArgs = typename MapPairsFunc::UniformArgs;
			static constexpr bool prefers_matrix = MapPairsFunc::prefersMatrix;

		private:
			CUDAKernel m_cuda_kernel;
			size_t default_size_i;
			size_t default_size_j;

			static constexpr size_t outArity = MapPairsFunc::outArity;
			static constexpr size_t numArgs = MapPairsFunc::totalArity - (MapPairsFunc::indexed ? 1 : 0) - (MapPairsFunc::usesPRNG ? 1 : 0) + outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapPairsFunc::ContainerArgs>::value;

			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<outArity + Varity, outArity>::type Velwise_indices{};
			static constexpr typename make_pack_indices<outArity + Varity + Harity, outArity + Varity>::type Helwise_indices{};
			static constexpr typename make_pack_indices<outArity + Varity + Harity + anyArity, outArity + Varity + Harity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, outArity + Varity + Harity + anyArity>::type const_indices{};

			using defaultDim = typename std::conditional<MapPairsFunc::indexed, index_dimension<typename MapPairsFunc::IndexType>, std::integral_constant<int, 1>>::type;
			using First = typename parameter_type<(MapPairsFunc::indexed ? 1 : 0) + (MapPairsFunc::usesPRNG ? 1 : 0), decltype(&MapPairsFunc::CPU)>::type;

			using F = ConditionalIndexForwarder<MapPairsFunc::indexed, MapPairsFunc::usesPRNG, decltype(&MapPairsFunc::CPU)>;
			using Temp = typename MapPairsFunc::Ret;
			using Ret = typename ReduceFunc::Ret;

			Ret m_start{};
			ReduceMode m_mode = ReduceMode::RowWise;


		public:
			void setStartValue(Ret val)
			{
				this->m_start = val;
			}

			void setDefaultSize(size_t i, size_t j = 0)
			{
				this->default_size_i = i;
				this->default_size_j = j;
			}

			void setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> typename std::add_lvalue_reference<decltype(get<0>(std::forward<CallArgs>(args)...))>::type
			{
				backendDispatch(
					out_indices, Velwise_indices, Helwise_indices, any_indices, const_indices,
					get<0>(std::forward<CallArgs>(args)...).size(),
					std::forward<CallArgs>(args)...
				);

				return get<0>(std::forward<CallArgs>(args)...);
			}

		private:
			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void backendDispatch(pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t size, CallArgs&&... args)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());

				size_t Vsize = get<0>(get<VEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_i);
				size_t Hsize = get<0>(get<HEI>(std::forward<CallArgs>(args)...).size()..., this->default_size_j);

				if (disjunction((get<VEI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
					SKEPU_ERROR("Non-matching input container sizes");

				if (disjunction((get<HEI>(std::forward<CallArgs>(args)...).size() < Hsize)...))
					SKEPU_ERROR("Non-matching input container sizes");

				if  ((this->m_mode == ReduceMode::RowWise && disjunction((get<OI>(std::forward<CallArgs>(args)...).size() < Vsize)...))
					|| (this->m_mode == ReduceMode::ColWise && disjunction((get<OI>(std::forward<CallArgs>(args)...).size() < Hsize)...)))
					SKEPU_ERROR("Non-matching output container size");

				this->selectBackend(Vsize + Hsize);

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->Hybrid(Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI> (std::forward<CallArgs>(args)...).begin()...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI> (std::forward<CallArgs>(args)...)...,
						get<CI> (std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CUDA(0, Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI> (std::forward<CallArgs>(args)...).begin()...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI> (std::forward<CallArgs>(args)...)...,
						get<CI> (std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(0, Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI> (std::forward<CallArgs>(args)...).begin()...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI> (std::forward<CallArgs>(args)...)...,
						get<CI> (std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI> (std::forward<CallArgs>(args)...).begin()...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI> (std::forward<CallArgs>(args)...)...,
						get<CI> (std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				default:
					this->CPU(Vsize, Hsize, oi, vei, hei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).begin()...,
						get<VEI>(std::forward<CallArgs>(args)...).begin()...,
						get<HEI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
				}
			}


			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void CPU(size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#ifdef SKEPU_OPENMP

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void OMP(size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_OPENMP

#ifdef SKEPU_CUDA

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void CUDA(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsReduceSingleThread_CU(size_t deviceID, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsReduceMultiStream_CU(size_t deviceID, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsReduceSingleThreadMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsReduceMultiStreamMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


#endif // SKEPU_CUDA

#ifdef SKEPU_OPENCL

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void CL(size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...> vei, pack_indices<VEI...>, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapPairsReduceNumDevices_CL(size_t numDevices, size_t startIdx, size_t Vsize, size_t Hsize, pack_indices<OI...>, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_OPENCL


#ifdef SKEPU_HYBRID

			template<size_t... OI, size_t... VEI, size_t... HEI, size_t... AI, size_t... CI, typename ...CallArgs>
			void Hybrid(size_t Vsize, size_t Hsize, pack_indices<OI...> oi, pack_indices<VEI...> vei, pack_indices<HEI...> hei, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_HYBRID

}; // class MapPairsReduce

	} // end namespace backend

#ifdef SKEPU_MERCURIUM

// TODO

#endif // SKEPU_MERCURIUM

} // end namespace skepu


#include "impl/mappairsreduce/mappairsreduce_cpu.inl"
#include "impl/mappairsreduce/mappairsreduce_omp.inl"
#include "impl/mappairsreduce/mappairsreduce_cl.inl"
#include "impl/mappairsreduce/mappairsreduce_cu.inl"
#include "impl/mappairsreduce/mappairsreduce_hy.inl"

#endif
