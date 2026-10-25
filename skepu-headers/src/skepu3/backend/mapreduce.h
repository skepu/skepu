/*! \file mapreduce.h
 *  \brief Contains a class declaration for the MapReduce skeleton.
 */

#ifndef MAPREDUCE_H
#define MAPREDUCE_H

namespace skepu
{
	namespace backend
	{
		/*!
		*  \ingroup skeletons
		*/
		/*!
		*  \class MapReduce
		*
		*  \brief A class representing the MapReduce skeleton.
		*
		*  This class defines the MapReduce skeleton which is a combination of the Map and Reduce operations. It produces
		*  the same result as if one would first Map one or more vectors (matrices) to a result vector (matrix), then do a reduction on that result.
		*  It is provided since it combines the mapping and reduction in the same computation kernel and therefore avoids some
		*  synchronization, which speeds up the calculation. Once instantiated, it is meant to be used as a function and therefore overloading
		*  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
		*  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
		*  takes whole containers (vectors, matrices).
		*/
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		class MapReduce : public SkeletonBase
		{
		public:
			SKEPU_CHAIN_SKELETON_SETTERS(SkeletonBase)
			MapReduce(std::string label, CUDAKernel mapreduce, CUDAReduceKernel reduce)
			: SkeletonBase{label}, m_cuda_kernel(mapreduce), m_cuda_reduce_kernel(reduce)
			{
#ifdef SKEPU_OPENCL
				this->template prepareOpenCL<CLKernel>(any_uf_requires_double<MapFunc, ReduceFunc>::value);
#endif
			}

			static constexpr auto skeletonType = SkeletonType::MapReduce;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = typename MapFunc::ElwiseArgs;
			using ContainerArgs = typename MapFunc::ContainerArgs;
			using UniformArgs = typename MapFunc::UniformArgs;
			static constexpr bool prefers_matrix = MapFunc::prefersMatrix;

		private:
			CUDAKernel m_cuda_kernel;
			CUDAReduceKernel m_cuda_reduce_kernel;
			size_t default_size_i = 0;
			size_t default_size_j = 0;
			size_t default_size_k = 0;
			size_t default_size_l = 0;
			StrideList<arity> m_strides{};

			static constexpr size_t outArity = MapFunc::outArity;
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0)  - (MapFunc::usesPRNG ? 1 : 0);
			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;
			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<arity, 0>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity, arity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity>::type const_indices{};
			
			using TempIndexType = typename std::conditional<MapFunc::indexed, typename MapFunc::IndexType, skepu::Index1D>::type;
			using defaultDim = index_dimension<TempIndexType>;

			using F = ConditionalIndexForwarder<MapFunc::indexed, MapFunc::usesPRNG, decltype(&MapFunc::CPU)>;
			using Ret = typename MapFunc::Ret;
			using RedType = typename ReduceFunc::Ret;

			Ret m_start{};
			bool m_apply_start = true;
			SparsePolicy m_sparse_policy = SparsePolicy::SamePattern;
			double m_implicit_zero = 0;
			ReduceMode m_reduce_mode = ReduceMode::RowWise;
			size_t m_default_origin = 0;

		public:
			auto &setStartValue(Ret val)
			{
				this->m_start = val;
				return *this;
			}

			// With false, the result of the skeleton is the reduction of the mapped elements alone: the start value is not
			// an operand. (A process of a distributed reduction that is not the one that has the start value.)
			auto &setApplyStartValue(bool apply)
			{
				this->m_apply_start = apply;
				return *this;
			}

			// The number, in the sequence of the indices of the default size, of the first index (of a part of an index space that
			// is distributed over processes)
			auto &setDefaultOrigin(size_t origin)
			{
				this->m_default_origin = origin;
				return *this;
			}

			// How sparse inputs of different patterns are combined (see Map)
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

			// For a sparse input and an output Vector: the direction of the reduction (only the rows, so far)
			auto &setReduceMode(ReduceMode mode)
			{
				this->m_reduce_mode = mode;
				return *this;
			}

			auto &setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
			{
				this->default_size_i = i;
				this->default_size_j = j;
				this->default_size_k = k;
				this->default_size_l = l;
				return *this;
			}

			template<typename... S>
			auto &setStride(S... strides)
			{
				this->m_strides = StrideList<arity>(strides...);
				return *this;
			}

			// Where the instance and its user functions are defined (for the tracer)
			void traceDefinition(tracing::Definition &definition) const override { definition.template describe<MapFunc, ReduceFunc>(); }


			void traceParameters(tracing::Parameters &parameters) const override
			{
				int strides[arity + 1];
				bool strided = false;
				for (size_t i = 0; i < arity; ++i)
					strided |= (strides[i] = this->m_strides[i]) != 1;
				if (strided)
					parameters.add("strides", strides, arity);
				parameters.addNumber("start_value", this->m_start);
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

			// With sparse input(s): the mapped entries of each row are reduced, to an element of the output Vector for each row
			template<typename R, typename S, typename... CallArgs, REQUIRES(arity > 0 && sizeof...(CallArgs) + 2 == numArgs + 1)>
			Vector<R> &operator()(Vector<R> &out, SparseMatrix<S> &arg1, CallArgs&&... args)
			{
				static_assert(outArity == 1, "MapReduce of the rows of sparse matrices: the map function must have one return value");
				static_assert(!MapFunc::usesPRNG, "MapReduce of the rows of sparse matrices: the map function cannot use random numbers");
				if (this->m_reduce_mode != ReduceMode::RowWise)
					SKEPU_ERROR("MapReduce of a sparse matrix: only the reduction of the rows is supported (use the transpose for columns)");
				auto sparse_guard = sparse_prepare<0, arity>(this->m_sparse_policy, this->m_implicit_zero, arg1, args...);
				return this->sparseRows(elwise_indices, any_indices, const_indices, out, arg1, std::forward<CallArgs>(args)...);
			}

			template<size_t... EI, size_t... AI, size_t... CI, typename R, typename... CallArgs>
			Vector<R> &sparseRows(pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Vector<R> &out, CallArgs&&... args)
			{
				const SparsePatternPtr &pattern = get<0>(std::forward<CallArgs>(args)...).pattern();
				const size_t rows = pattern->rows();
				if (out.size() != rows)
					SKEPU_ERROR("MapReduce: the output Vector has " << out.size() << " elements, and the sparse matrix " << rows << " rows");
				DEBUG_TEXT_LEVEL1("MapReduce of sparse rows: rows = " << rows << " nnz = " << pattern->nnz());

				this->selectBackend(pattern->nnz());
				pack_expand((get<EI>(std::forward<CallArgs>(args)...).getParent().updateHost(), 0)...);
				pack_expand((get<AI>(std::forward<CallArgs>(args)...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				out.invalidateDeviceData();

				SKEPU_TRACE_START_EVENT(trace_handle);
				auto random = this->template prepareRandom<MapFunc::randomCount>(rows);
				auto elwise = std::make_tuple(get<EI>(std::forward<CallArgs>(args)...).begin()...);
				const SparseIndex *offsets = pattern->row_offsets().data(), *columns = pattern->col_indices().data();
				R *result = out.getAddress();

				auto row = [&](size_t r)
				{
					Ret res = this->m_start;
					for (SparseIndex k = offsets[r]; k < offsets[r + 1]; ++k)
					{
						const Index2D index{r, (size_t)columns[k]};
						Ret temp = F::forward(MapFunc::CPU, index, random,
							std::get<EI>(elwise)(k)...,
							get<AI>(std::forward<CallArgs>(args)...).hostProxy(std::get<AI-arity>(typename MapFunc::ProxyTags{}), index)...,
							get<CI>(std::forward<CallArgs>(args)...)...
						);
						res = ReduceFunc::CPU(res, temp);
					}
					result[r] = res;
				};

#ifdef SKEPU_OPENMP
				if (this->m_selected_spec->activateBackend() == Backend::Type::OpenMP)
				{
#pragma omp parallel for schedule(runtime)
					for (size_t r = 0; r < rows; ++r)
						row(r);
				}
				else
#endif
				{
					for (size_t r = 0; r < rows; ++r)
						row(r);
				}

				SKEPU_TRACE_CALL(trace_handle, "MapReduce", this, {pattern->nnz()},
					{out.getObjectID()},
					{get<EI>(std::forward<CallArgs>(args)...).getParent().getObjectID()...},
					{get<AI>(std::forward<CallArgs>(args)...).getObjectID()...},
					tracing::scalar_labels(get<CI>(std::forward<CallArgs>(args)...)...),
					this->template randomID<MapFunc::randomCount>()
				);
				return out;
			}

			// For first elwise argument as container
			template<typename First, template<class> class Container, typename... CallArgs, REQUIRES(arity > 0 && is_skepu_container<Container<First>>::value)>
			scalar_wrapper_t<Ret> operator()(Container<First> &arg1, CallArgs&&... args)
			{
				auto sparse_guard = sparse_prepare<0, arity>(this->m_sparse_policy, this->m_implicit_zero, arg1, args...);
				return backendDispatch(
					out_indices, elwise_indices, any_indices, const_indices,
					arg1.size() / this->m_strides[0], arg1,
					std::forward<CallArgs>(args)...
				);
			}

			// For first elwise argument as iterator
			template<typename Iterator, typename... CallArgs, REQUIRES(sizeof...(CallArgs) + 1 == numArgs)>
			scalar_wrapper_t<Ret> operator()(Iterator arg1, Iterator arg1_end, CallArgs&&... args)
			{
				return backendDispatch(
					out_indices, elwise_indices, any_indices, const_indices,
					(arg1_end - arg1) / this->m_strides[0], arg1,
					std::forward<CallArgs>(args)...
				);
			}

			// For no elwise arguments
			template<typename... CallArgs, REQUIRES(sizeof...(CallArgs) == numArgs)>
			scalar_wrapper_t<Ret> operator()(CallArgs&&... args)
			{
				size_t size = this->default_size_i;
				if (defaultDim::value >= 2) size *= this->default_size_j;
				if (defaultDim::value >= 3) size *= this->default_size_k;
				if (defaultDim::value >= 4) size *= this->default_size_l;

				if (size == 0)
					SKEPU_WARNING("MapReduce<0> default size not set or set to 0.")

				return this->backendDispatch(
					out_indices, elwise_indices, any_indices, const_indices,
					size,
					std::forward<CallArgs>(args)...
				);
			}

		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> backendDispatch(pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t size, CallArgs&&... args)
			{
			//	assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());

				Ret res = this->m_start;

				if (disjunction((get<EI>(std::forward<CallArgs>(args)...).size() < size)...))
					SKEPU_ERROR("Non-matching container sizes");

				this->selectBackend(size);

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					return this->Hybrid(size, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					return this->CUDA(0, size, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					return this->CL(0, size, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					return this->OMP(size, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[EI])...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
#endif
				default:
					return this->CPU(size, oi, ei, ai, ci, res,
						get<EI>(std::forward<CallArgs>(args)...).stridedBegin(size, this->m_strides[EI])...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
				}
			}


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> CPU(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);


#ifdef SKEPU_OPENMP

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			scalar_wrapper_t<Ret> OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

#endif // SKEPU_OPENMP

#ifdef SKEPU_CUDA

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> CUDA(size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			scalar_wrapper_t<Ret> mapReduceSingleThread_CU(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			scalar_wrapper_t<Ret> mapReduceMultiStream_CU(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> mapReduceSingleThreadMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs>
			scalar_wrapper_t<Ret> mapReduceMultiStreamMultiGPU_CU(size_t useNumGPU, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

#endif // SKEPU_CUDA

#ifdef SKEPU_OPENCL

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> CL(size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> mapReduceNumDevices_CL(size_t numDevices, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> mapReduceSingle_CL(size_t deviceID, size_t startIdx, size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

#endif // SKEPU_OPENCL


#ifdef SKEPU_HYBRID

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			scalar_wrapper_t<Ret> Hybrid(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);

#endif // SKEPU_HYBRID

		}; // class MapReduce

	} // end namespace backend

} // end namespace skepu


#include "impl/mapreduce/mapreduce_cpu.inl"
#include "impl/mapreduce/mapreduce_omp.inl"
#include "impl/mapreduce/mapreduce_cl.inl"
#include "impl/mapreduce/mapreduce_cu.inl"
#include "impl/mapreduce/mapreduce_hy.inl"

#endif
