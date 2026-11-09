#ifndef MAPPOOL_1D_H
#define MAPPOOL_1D_H

#include "skepu3/impl/region.hpp"

namespace skepu
{
	namespace backend
	{
		// single device; no Hybrid
		template<typename MapFunc, typename CUDAKernel, typename CLKernel>
		class MapPool1D : public SkeletonBase
		{
			using F = ConditionalIndexForwarder<MapFunc::indexed, MapFunc::usesPRNG, decltype(&MapFunc::CPU)>;
			using PoolType = typename parameter_type<(MapFunc::indexed ? 1 : 0) + (MapFunc::usesPRNG ? 1 : 0), decltype(&MapFunc::CPU)>::type;
			using T = typename region_type<PoolType>::type;

			static constexpr size_t outArity = MapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0) - (MapFunc::usesPRNG ? 1 : 0) + outArity;

			CUDAKernel m_cuda_kernel;
			int m_pool_size = 1;
			int m_stride = 1;

		public:
			SKEPU_CHAIN_SKELETON_SETTERS(SkeletonBase)

			static constexpr auto skeletonType = SkeletonType::MapOverlap1D;

			static constexpr typename make_pack_indices<outArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<1 + outArity, outArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<1 + anyArity + outArity, 1 + outArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, 1 + anyArity + outArity>::type const_indices{};

			MapPool1D(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				this->template prepareOpenCL<CLKernel>(any_uf_requires_double<MapFunc>::value);
#endif
			}

			void traceDefinition(tracing::Definition &definition) const override { definition.template describe<MapFunc>(); }

			void traceParameters(tracing::Parameters &parameters) const override
			{
				parameters.add("pool", true);
				parameters.add("pool_size", &this->m_pool_size, 1);
				parameters.add("stride", &this->m_stride, 1);
			}

			auto &setPoolSize(int p)
			{
				if (p < 0)
					SKEPU_ERROR("Pool size cannot be less than 0");
				this->m_pool_size = p;
				return *this;
			}

			int getPoolSize() const { return this->m_pool_size; }

			auto &setStride(int s)
			{
				if (s < 0)
					SKEPU_ERROR("Stride cannot be less than 0");
				this->m_stride = s;
				return *this;
			}

			int getStride() const { return this->m_stride; }

			auto &setPoolSizeAndStride(int p)
			{
				this->setPoolSize(p);
				this->setStride(p);
				return *this;
			}

			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching MapPool function");
				const size_t size = get<0>(std::forward<CallArgs>(args)...).size();
				if (size > 0 && get<outArity>(std::forward<CallArgs>(args)...).size() < inputSize(size))
					SKEPU_ERROR("MapPool1D: the input has " << get<outArity>(std::forward<CallArgs>(args)...).size() << " elements, the output of "
						<< size << " elements needs at least " << inputSize(size));
				this->backendDispatch(size, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				return get<0>(std::forward<CallArgs>(args)...);
			}

		private:
			size_t inputSize(size_t size) const
			{
				return size == 0 ? 0 : this->m_pool_size + (size_t)this->m_stride * (size - 1);
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void CPU(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#ifdef SKEPU_OPENMP
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void OMP(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
#endif

#ifdef SKEPU_OPENCL
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void CL(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
#endif

#ifdef SKEPU_CUDA
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void CUDA(size_t size, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
#endif

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void backendDispatch(size_t size, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
			{
				this->selectBackend(size);

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
					SKEPU_ERROR("MapPool1D is not implemented for Hybrid execution");
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CUDA(size, oi, ei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).begin()...,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(size, oi, ei, ai, ci,
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
						get<OI>(std::forward<CallArgs>(args)...).begin()...,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
#endif
				default:
					this->CPU(size, oi, ei, ai, ci,
						get<OI>(std::forward<CallArgs>(args)...).begin()...,
						get<EI>(std::forward<CallArgs>(args)...).begin()...,
						get<AI>(std::forward<CallArgs>(args)...)...,
						get<CI>(std::forward<CallArgs>(args)...)...
					);
					break;
				}
			}
		};

	} // namespace backend
} // namespace skepu

#include "skepu3/backend/impl/mapoverlap/mappool_1d/mappool_1d_cpu.inl"
#include "skepu3/backend/impl/mapoverlap/mappool_1d/mappool_1d_omp.inl"
#include "skepu3/backend/impl/mapoverlap/mappool_1d/mappool_1d_cl.inl"
#include "skepu3/backend/impl/mapoverlap/mappool_1d/mappool_1d_cu.inl"

#endif // MAPPOOL_1D_H
