#ifndef MAPOVERLAP_4D_H
#define MAPOVERLAP_4D_H

#include "skepu3/backend/mapoverlap/mapoverlap_par.h"

namespace skepu
{

	namespace backend
	{

        template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapOverlap4D: public MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap4D>, public SkeletonBase
		{
		protected:
			using Base = MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap4D>;
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
			MapOverlap4D(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}

			auto &setOverlap(int oi, int oj, int ok, int ol)
			{
				if (oi < 0 || oj < 0 || ok < 0 || ol < 0)
					SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
				this->m_overlap[2] = ok;
				this->m_overlap[3] = ol;
				return *this;
			}

			auto &setOverlap(int o)
			{
				this->setOverlap(o, o, o, o);
				return *this;
			}

			std::tuple<int, int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3]);
			}

			auto &setStride(int si, int sj, int sk, int sl)
			{
				if (si < 0 || sj < 0 || sk < 0 || sl < 0)
					SKEPU_ERROR("Stride cannot be less than 0");
				this->m_strides[0] = si;
				this->m_strides[1] = sj;
				this->m_strides[2] = sk;
				this->m_strides[3] = sl;
				return *this;
			}

			auto &setStride(int s)
			{
				this->setStride(s, s, s, s);
				return *this;
			}

			std::tuple<int, int, int, int> getStride() const
			{
				return std::make_tuple(this->m_strides[0], this->m_strides[1], this->m_strides[2], this->m_strides[3]);
			}

			void traceParameters(tracing::Parameters &parameters) const override
			{
				this->describeOverlap(parameters, 4);
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
			void checkOutputTensor4Sizes(size_t expected_size_i, size_t expected_size_j, size_t expected_size_k, size_t expected_size_l,
										 size_t i, First&& first, Rest&&... rest)
			{
				if (first.size_i() != expected_size_i)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output tensor4 size i"
					<< "\nexpected output tensor4 size i: " << colorRed(expected_size_i)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size i: " << colorRed(first.size_i()) << ")");
				
				if (first.size_j() != expected_size_j)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output tensor4 size j"
					<< "\nexpected output tensor4 size j: " << colorRed(expected_size_j)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size j: " << colorRed(first.size_j()) << ")");
				
				if (first.size_k() != expected_size_k)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output tensor4 size k"
					<< "\nexpected output tensor4 size k: " << colorRed(expected_size_k)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size k: " << colorRed(first.size_k()) << ")");
				
				if (first.size_l() != expected_size_l)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output tensor4 size l"
					<< "\nexpected output tensor4 size l: " << colorRed(expected_size_l)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size l: " << colorRed(first.size_l()) << ")");

				checkOutputTensor4Sizes(expected_size_i, expected_size_j, expected_size_k, expected_size_l, i+1, rest...);
			}

			void checkOutputTensor4Sizes(size_t expected_size_i, size_t expected_size_j, size_t expected_size_k, size_t expected_size_l, size_t i){}

			template<size_t... OI, typename... CallArgs>
			void checkTensor4Sizes(pack_indices<OI...>, CallArgs&&... args)
			{
				auto& first_output = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t out_size_i = first_output.size_i();
				size_t out_size_j = first_output.size_j();
				size_t out_size_k = first_output.size_k();
				size_t out_size_l = first_output.size_l();
				size_t in_size_i = input.size_i();
				size_t in_size_j = input.size_j();
				size_t in_size_k = input.size_k();
				size_t in_size_l = input.size_l();

				// Mirror mode: the edge mode None with an input of the size of the output; only the interior of the output is computed (the other elements keep their values)
				this->m_mirror_mode = !this->isPool && this->m_edge == skepu::Edge::None && out_size_i == in_size_i && out_size_j == in_size_j && out_size_k == in_size_k && out_size_l == in_size_l;
				if (this->m_mirror_mode)
					return;

                if (!this->isInputSizeValid(out_size_i, in_size_i, 0) && this->m_overlap[0] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output tensor4 size i mismatch"
					<< "\nfirst output tensor4 (label: " << first_output.getLabel() << ", size i: " << out_size_i << ")"
					<< "\nexpected input tensor4 size i: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_i, 0)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_i) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_i, 0)))
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size i: " << colorRed(in_size_i) << ")");
				
				if (!this->isInputSizeValid(out_size_j, in_size_j, 1) && this->m_overlap[1] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output tensor4 size j mismatch"
					<< "\nfirst output tensor4 (label: " << first_output.getLabel() << ", size j: " << out_size_j << ")"
					<< "\nexpected input tensor4 size j: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_j, 0)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_j) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_j, 0)))
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size j: " << colorRed(in_size_j) << ")");
                
                if (!this->isInputSizeValid(out_size_k, in_size_k, 2) && this->m_overlap[2] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output tensor4 size k mismatch"
					<< "\nfirst output tensor4 (label: " << first_output.getLabel() << ", size k: " << out_size_k << ")"
					<< "\nexpected input tensor4 size k: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_k, 0)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_k) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_k, 0)))
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size k: " << colorRed(in_size_k) << ")");

                if (!this->isInputSizeValid(out_size_l, in_size_l, 3) && this->m_overlap[3] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output tensor4 size l mismatch"
					<< "\nfirst output tensor4 (label: " << first_output.getLabel() << ", size l: " << out_size_l << ")"
					<< "\nexpected input tensor4 size l: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_l, 0)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_l) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_l, 0)))
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size l: " << colorRed(in_size_l) << ")");
				
				checkOutputTensor4Sizes(out_size_i, out_size_j, out_size_k, out_size_l, 1, get<OI>(std::forward<CallArgs>(args)...)...);
			}

			std::string generateCallMetadata()
			{
				return "MapOverlap4D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode()) +
                    ", Overlap: (" + std::to_string(this->m_overlap[0]) + ", " + std::to_string(this->m_overlap[1]) +
                    ", " + std::to_string(this->m_overlap[2]) + ", " + std::to_string(this->m_overlap[3]) + ")";
			}

		public:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
				size_t size_l = get<0>(std::forward<CallArgs>(args)...).size_l();

				checkTensor4Sizes(this->out_indices, std::forward<CallArgs>(args)...);

				if (this->mirrorHasNoInterior(size_i, size_j, size_k, size_l)) // (mirror mode with nothing to compute: only the random numbers are used)
				{
					this->template prepareRandom<MapOverlapFunc::randomCount>(size_i * size_j * size_k * size_l);
					return get<0>(std::forward<CallArgs>(args)...);
				}

				// Remove later
				auto &res = get<0>(std::forward<CallArgs>(args)...);
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
				// End remove

				this->selectBackend(get<0>(std::forward<CallArgs>(args)...).size());

				switch (this->checkMirrorBackend(this->m_selected_spec->activateBackend(), "MapOverlap 4D"))
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
				this->checkUpdateModeRandom(MapOverlapFunc::randomCount);
				
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
		}; // MapOverlap4D

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapPool4D: public MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
		private:
		using typename MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>::T;

		public:
			using SkepuChainBase = MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>;
			SKEPU_CHAIN_SKELETON_SETTERS(SkepuChainBase)
			SKEPU_CHAIN(setStride, SkepuChainBase)
			void setOverlap(int, int, int, int) = delete;
			void setOverlap(int) = delete;
			std::tuple<int, int, int, int> getOverlap() const = delete;

			void setEdgeMode(Edge) = delete;
			Edge getEdgeMode() const = delete;

			void setPad(T) = delete;
			T getPad() const = delete;
			
			void setUpdateMode(UpdateMode) = delete;
			UpdateMode getUpdateMode() const = delete;

			auto &setPoolSize(int pi, int pj, int pk, int pl)
			{
				if (pi < 0 || pj < 0 || pk < 0 || pl < 0)
                    SKEPU_ERROR("Pool size cannot be less than 0");
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
				this->m_overlap[2] = pk;
				this->m_overlap[3] = pl;
				return *this;
			}

			auto &setPoolSize(int p)
			{
				this->setPoolSize(p, p, p, p);
				return *this;
			}

			std::tuple<int, int, int, int> getPoolSize() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3]);
			}

			auto &setPoolSizeAndStride(int pi, int pj, int pk, int pl)
			{
				this->setPoolSize(pi, pj, pk, pl);
				this->setStride(pi, pj, pk, pl);
				return *this;
			}

			auto &setPoolSizeAndStride(int p)
			{
				this->setPoolSize(p);
				this->setStride(p);
				return *this;
			}

			MapPool4D(std::string label, CUDAKernel kernel): MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>(label, kernel) {}
		};

    } // backend

} // skepu

#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_cpu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_omp.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_cl.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_cu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_hy.inl"

#endif // MAPOVERLAP_4D_H