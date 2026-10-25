#ifndef MAPOVERLAP_3D_H
#define MAPOVERLAP_3D_H

#include "skepu3/backend/mapoverlap/mapoverlap_par.h"

namespace skepu
{

	namespace backend
	{

        template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapOverlap3D: public MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap3D>, public SkeletonBase
		{
		protected:
			using Base = MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap3D>;
			using Base::InArity;
			using Base::OutArity;
            using typename Base::RegionType;
			using typename Base::T;
			using typename Base::F;

		public:
			SKEPU_CHAIN_SKELETON_SETTERS(SkeletonBase)
			SKEPU_CHAIN(setEdgeMode, Base)
			SKEPU_CHAIN(setPad, Base)
			SKEPU_CHAIN(setUpdateMode, Base)
			MapOverlap3D(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				this->template prepareOpenCL<CLKernel>(any_uf_requires_double<MapOverlapFunc>::value);
#endif
			}

			auto &setOverlap(int oi, int oj, int ok)
			{
				if (oi < 0 || oj < 0 || ok < 0)
					SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
				this->m_overlap[2] = ok;
				return *this;
			}

            auto &setOverlap(int o)
			{
                this->setOverlap(o, o, o);
				return *this;
			}

			std::tuple<int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1], this->m_overlap[2]);
			}

			auto &setStride(int si, int sj, int sk)
			{
				if (si < 0 || sj < 0 || sk < 0)
					SKEPU_ERROR("Stride cannot be less than 0");
				this->m_strides[0] = si;
				this->m_strides[1] = sj;
				this->m_strides[2] = sk;
				return *this;
			}

            auto &setStride(int s)
			{
                this->setStride(s, s, s);
				return *this;
			}

            std::tuple<int, int, int> getStride() const
			{
				return std::make_tuple(this->m_strides[0], this->m_strides[1], this->m_strides[2]);
			}

			// Where the instance and its user functions are defined (for the tracer)
			void traceDefinition(tracing::Definition &definition) const override { definition.template describe<MapOverlapFunc>(); }


			void traceParameters(tracing::Parameters &parameters) const override
			{
				this->describeOverlap(parameters, 3);
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

		private:
			CUDAKernel m_cuda_kernel;


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#ifdef SKEPU_OPENMP

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif

#ifdef SKEPU_OPENCL

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_OpenCL(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CL(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CL(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif

#ifdef SKEPU_CUDA

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CU(size_t numDevices, Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CUDA(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);

#endif

#ifdef SKEPU_HYBRID

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_Hybrid(Parity p, pack_indices<OI...> oi, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args);

#endif

			template<typename First, typename... Rest>
			void checkOutputTensor3Sizes(size_t expected_size_i, size_t expected_size_j, size_t expected_size_k,
                                         size_t i, First&& first, Rest&&... rest)
			{
				if (first.size_i() != expected_size_i)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output tensor3 size i"
					<< "\nexpected output tensor3 size i: " << colorRed(expected_size_i)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size i: " << colorRed(first.size_i()) << ")");
				
				if (first.size_j() != expected_size_j)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output tensor3 size j"
					<< "\nexpected output tensor3 size j: " << colorRed(expected_size_j)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size j: " << colorRed(first.size_j()) << ")");
				
				if (first.size_k() != expected_size_k)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output tensor3 size k"
					<< "\nexpected output tensor3 size k: " << colorRed(expected_size_k)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size k: " << colorRed(first.size_k()) << ")");

				checkOutputTensor3Sizes(expected_size_i, expected_size_j, expected_size_k, i+1, rest...);
			}

			void checkOutputTensor3Sizes(size_t expected_size_i, size_t expected_size_j, size_t expected_size_k, size_t i){}

			template<size_t... OI, typename... CallArgs>
			void checkTensor3Sizes(pack_indices<OI...>, CallArgs&&... args)
			{
				auto& first_output = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t out_size_i = first_output.size_i();
				size_t out_size_j = first_output.size_j();
				size_t out_size_k = first_output.size_k();
				size_t in_size_i = input.size_i();
				size_t in_size_j = input.size_j();
				size_t in_size_k = input.size_k();
				
				// Mirror mode: the edge mode None with an input of the size of the output; only the interior of the output is computed (the other elements keep their values)
				this->m_mirror_mode = !this->isPool && this->m_edge == skepu::Edge::None && out_size_i == in_size_i && out_size_j == in_size_j && out_size_k == in_size_k;
				if (this->m_mirror_mode)
					return;
				
				if (!this->isInputSizeValid(out_size_i, in_size_i, 0) && this->m_overlap[0] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output tensor3 size i mismatch"
					<< "\nfirst output tensor3 (label: " << first_output.getLabel() << ", size i: " << out_size_i << ")"
					<< "\nexpected input tensor3 size i: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_i, 0)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_i) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_i, 0)))
					<< "\ninput tensor3 (label: " << input.getLabel() << ", size i: " << colorRed(in_size_i) << ")");
				
                if (!this->isInputSizeValid(out_size_j, in_size_j, 1) && this->m_overlap[1] > 0)   
                    SKEPU_ERROR(generateCallMetadata()
                    << "\ninput/output tensor3 size j mismatch"
                    << "\nfirst output tensor3 (label: " << first_output.getLabel() << ", size j: " << out_size_j << ")"
                    << "\nexpected input tensor3 size j: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_j, 1)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_j) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_j, 1)))
                    << "\ninput tensor3 (label: " << input.getLabel() << ", size j: " << colorRed(in_size_j) << ")");
                
				if (!this->isInputSizeValid(out_size_k, in_size_k, 2) && this->m_overlap[2] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output tensor3 size k mismatch"
					<< "\nfirst output tensor3 (label: " << first_output.getLabel() << ", size k: " << out_size_k << ")"
					<< "\nexpected input tensor3 size k: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_k, 2)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_k) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_k, 2)))
					<< "\ninput tensor3 (label: " << input.getLabel() << ", size k: " << colorRed(in_size_k) << ")");
				
				checkOutputTensor3Sizes(out_size_i, out_size_j, out_size_k, 1, get<OI>(std::forward<CallArgs>(args)...)...);
			}

			std::string generateCallMetadata()
			{
				return "MapOverlap3D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode()) +
                    ", Overlap: (" + std::to_string(this->m_overlap[0]) + ", " + std::to_string(this->m_overlap[1]) +
                    ", " + std::to_string(this->m_overlap[2]) + ")";
			}

		public:

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				checkTensor3Sizes(this->out_indices, std::forward<CallArgs>(args)...);

				{
					auto &first_output = get<0>(std::forward<CallArgs>(args)...);
					if (this->mirrorHasNoInterior(first_output.size_i(), first_output.size_j(), first_output.size_k())) // (mirror mode with nothing to compute: only the random numbers are used)
					{
						this->template prepareRandom<MapOverlapFunc::randomCount>(first_output.size());
						return first_output;
					}
				}

				this->selectBackend(get<0>(std::forward<CallArgs>(args)...).size());

				switch (this->checkMirrorBackend(this->m_selected_spec->activateBackend(), "MapOverlap 3D"))
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->helper_Hybrid(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->helper_CUDA(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->helper_OpenCL(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->helper_OpenMP(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->helper_CPU(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
				}

				return get<0>(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				this->checkUpdateMode(MapOverlapFunc::randomCount, this->out_indices, this->elwise_indices, this->any_indices, std::forward<CallArgs>(args)...);
				
				if (this->m_updateMode == UpdateMode::Normal)
				{
					this->backendDispatch(Parity::None, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->backendDispatch(Parity::Odd, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->backendDispatch(Parity::Even, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}
		}; // MapOverlap3D

        template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapPool3D: public MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
		private:
		using typename MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>::T;

		public:
			using SkepuChainBase = MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>;
			SKEPU_CHAIN_SKELETON_SETTERS(SkepuChainBase)
			SKEPU_CHAIN(setStride, SkepuChainBase)
			void setOverlap(int, int, int) = delete;
			void setOverlap(int) = delete;
			std::tuple<int, int, int> getOverlap() const = delete;

			void setEdgeMode(Edge) = delete;
			Edge getEdgeMode() const = delete;

			void setPad(T) = delete;
			T getPad() const = delete;
			
			void setUpdateMode(UpdateMode) = delete;
			UpdateMode getUpdateMode() const = delete;

			auto &setPoolSize(int pi, int pj, int pk)
			{
				if (pi < 0 || pj < 0 || pk < 0)
                    SKEPU_ERROR("Pool size cannot be less than 0");
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
				this->m_overlap[2] = pk;
				return *this;
			}

			auto &setPoolSize(int p)
			{
				this->setPoolSize(p, p, p);
				return *this;
			}

			std::tuple<int, int, int> getPoolSize() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1], this->m_overlap[2]);
			}

			auto &setPoolSizeAndStride(int pi, int pj, int pk)
			{
				this->setPoolSize(pi, pj, pk);
				this->setStride(pi, pj, pk);
				return *this;
			}

			auto &setPoolSizeAndStride(int p)
			{
				this->setPoolSize(p);
				this->setStride(p);
				return *this;
			}

			MapPool3D(std::string label, CUDAKernel kernel): MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>(label, kernel) {}
		};

    } // backend

} // skepu

#include "skepu3/backend/impl/mapoverlap/mapoverlap_3d/mapoverlap_3d_cpu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_3d/mapoverlap_3d_omp.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_3d/mapoverlap_3d_cl.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_3d/mapoverlap_3d_cu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_3d/mapoverlap_3d_hy.inl"

#endif // MAPOVERLAP_3D_H