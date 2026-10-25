/*! \file map.h
 *  \brief Contains a class declaration for the Map skeleton.
 */

#ifndef MAP_H
#define MAP_H

namespace skepu
{
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
 		 */
		/*!
		 *  \class Map
		 *
		 *  \brief A class representing the Map skeleton.
		 *
		 *  This class defines the Map skeleton, a calculation pattern where a user function is applied to each element of an input
		 *  range. Once the Map object is instantiated, it is meant to be used as a function and therefore overloading
		 *  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
		 *  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
		 *  takes whole containers (vectors, matrices). The container variants are merely wrappers for the functions which takes iterators as parameters.
		 */
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		class Map : public SkeletonBase
		{
			// ==========================    Type definitions   ==========================

			using T = typename MapFunc::Ret;
			using F = ConditionalIndexForwarder<MapFunc::indexed, MapFunc::usesPRNG, decltype(&MapFunc::CPU)>;

			// ==========================     Class members     ==========================

			static constexpr size_t outArity = MapFunc::outArity;
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0) - (MapFunc::usesPRNG ? 1 : 0) + outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;

			// ==========================    Instance members   ==========================

			CUDAKernel m_cuda_kernel;
			StrideList<outArity + arity> m_strides{};
			SparsePolicy m_sparse_policy = SparsePolicy::SamePattern;
			double m_implicit_zero = 0;

		public:
			SKEPU_CHAIN_SKELETON_SETTERS(SkeletonBase)

			static constexpr auto skeletonType = SkeletonType::Map;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = typename MapFunc::ElwiseArgs;
			using ContainerArgs = typename MapFunc::ContainerArgs;
			using UniformArgs = typename MapFunc::UniformArgs;
			static constexpr bool prefers_matrix = MapFunc::prefersMatrix;

			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<arity + outArity, outArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity + outArity, arity + outArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity + outArity>::type const_indices{};

			Map(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				this->template prepareOpenCL<CLKernel>(any_uf_requires_double<MapFunc>::value);
#endif
			}

			// Where the instance and its user functions are defined (for the tracer)
			void traceDefinition(tracing::Definition &definition) const override { definition.template describe<MapFunc>(); }


			void traceParameters(tracing::Parameters &parameters) const override
			{
				// one stride for each container argument, the outputs first; only if one is not 1 (it skips elements)
				int strides[outArity + arity];
				bool strided = false;
				for (size_t i = 0; i < outArity + arity; ++i)
					strided |= (strides[i] = this->m_strides[i]) != 1;
				if (strided)
					parameters.add("strides", strides, outArity + arity);
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

			template<typename... S>
			auto &setStride(S... strides)
			{
				this->m_strides = StrideList<outArity + arity>(strides...);
				return *this;
			}

			// How sparse inputs of different patterns are combined: the output has an entry wherever any (Union) or all (Intersection)
			// of the inputs have one, and a missing entry has the implicit zero as its value
			auto &setSparsePolicy(SparsePolicy policy)
			{
				this->m_sparse_policy = policy;
				return *this;
			}

			auto &setImplicitZero(double zero)
			{
				this->m_implicit_zero = zero;
				return *this;
			}

			// =======================      Call operators      ==========================

			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
				auto sparse_guard = sparse_prepare<outArity, arity>(this->m_sparse_policy, this->m_implicit_zero, args...);
				this->backendDispatch(
					get<0>(std::forward<CallArgs>(args)...).size() / this->m_strides[0],
					out_indices, elwise_indices, any_indices, const_indices,
					std::forward<CallArgs>(args)...
				);
				return get<0>(std::forward<CallArgs>(args)...);
			}


			template<typename Iterator, typename... CallArgs, REQUIRES(is_skepu_iterator<Iterator, T>::value)>
			Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
			{
				static_assert(sizeof...(CallArgs) == numArgs-1, "Number of arguments not matching Map function");
				this->backendDispatch(
					(res_end - res)  / this->m_strides[0],
					out_indices, elwise_indices, any_indices, const_indices,
					res, std::forward<CallArgs>(args)...
				);
				return res;
			}

		private:

			// ==========================    Implementation     ==========================

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void CPU(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


#ifdef SKEPU_OPENMP

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			void OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_OPENMP


#ifdef SKEPU_CUDA

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void CUDA(size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			void         mapSingleThread_CU(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			void          mapMultiStream_CU(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapSingleThreadMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			void  mapMultiStreamMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_CUDA


#ifdef SKEPU_OPENCL

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void CL(size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapNumDevices_CL(size_t startIdx, size_t numDevices, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_OPENCL

#ifdef SKEPU_HYBRID

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void Hybrid(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif // SKEPU_HYBRID

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void backendDispatch(size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
			{
			//	assert(this->m_execPlan != nullptr && this->m_execPlan->isCalibrated());

				if (disjunction((get<OI>(std::forward<CallArgs>(args)...).size() / abs(this->m_strides[OI]) < size)...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction((get<EI>(std::forward<CallArgs>(args)...).size() / abs(this->m_strides[EI]) < size)...))
					SKEPU_ERROR("Non-matching input container sizes");

				this->selectBackend(size);

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->Hybrid(size, oi, ei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).begin()...,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CUDA(0, size, oi, ei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).begin()...,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(0, size, oi, ei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).begin()...,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(size, oi, ei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[OI])...,
						get<EI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[EI])...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				default:
					this->CPU(size, oi, ei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[OI])...,
						get<EI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[EI])...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
				}
			}

		}; // class Map

	} // namespace backend

} // namespace skepu


#include "impl/map/map_cpu.inl"
#include "impl/map/map_omp.inl"
#include "impl/map/map_cl.inl"
#include "impl/map/map_cu.inl"
#include "impl/map/map_hy.inl"

#endif // MAP_H
