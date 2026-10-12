#ifndef MAPOVERLAP_2D_H
#define MAPOVERLAP_2D_H

#include "skepu3/backend/mapoverlap/mapoverlap_par.h"

namespace skepu
{

	namespace backend
	{

        template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapOverlap2D: public MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap2D>, public SkeletonBase
		{
		protected:
			using Base = MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap2D>;
			using Base::InArity;
			using Base::OutArity;
			using typename Base::RegionType;
			using typename Base::T;
			using typename Base::F;

		public:
			MapOverlap2D(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}

			SKEPU_CHAIN_SKELETON_SETTERS(SkeletonBase)
			SKEPU_CHAIN(setEdgeMode, Base)
			SKEPU_CHAIN(setPad, Base)
			SKEPU_CHAIN(setUpdateMode, Base)

			auto &setOverlap(int oi, int oj)
			{
				if (oi < 0 || oj < 0)
					SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
				return *this;
			}

			auto &setOverlap(int o)
			{
				return this->setOverlap(o, o);
			}

			std::tuple<int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1]);
			}

			auto &setStride(int si, int sj)
			{
				if (si < 0 || sj < 0)
					SKEPU_ERROR("Stride cannot be less than 0");
				this->m_strides[0] = si;
				this->m_strides[1] = sj;
				return *this;
			}

			auto &setStride(int s)
			{
				this->setStride(s, s);
				return *this;
			}

			std::tuple<int, int> getStride() const
			{
				return std::make_tuple(this->m_strides[0], this->m_strides[1]);
			}

			// Where the instance and its user functions are defined (for the tracer)
			void traceDefinition(tracing::Definition &definition) const override { definition.template describe<MapOverlapFunc>(); }


			void traceParameters(tracing::Parameters &parameters) const override
			{
				this->describeOverlap(parameters, 2);
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

		private:
			CUDAKernel m_cuda_kernel;


		private:
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
			void mapOverlapSingleThread_CU(size_t deviceID, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultipleThread_CU(size_t numDevices, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_CUDA(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif

#ifdef SKEPU_HYBRID

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void helper_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif

			template<typename First, typename... Rest>
			void checkOutputMatrixSizes(size_t expected_size_i, size_t expected_size_j, size_t i, First&& first, Rest&&... rest)
			{
				if (first.size_i() != expected_size_i)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid number of output matrix rows"
					<< "\nexpected output matrix rows: " << colorRed(expected_size_i)
					<< "\noutput matrix (label: " << first.getLabel() << ", index: " << i << ", rows: " << colorRed(first.size_i()) << ")");
				
				if (first.size_j() != expected_size_j)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid number of output matrix cols"
					<< "\nexpected output matrix cols: " << colorRed(expected_size_j)
					<< "\noutput matrix (label: " << first.getLabel() << ", index: " << i << ", cols: " << colorRed(first.size_j()) << ")");

				checkOutputMatrixSizes(expected_size_i, expected_size_j, i+1, rest...);
			}

			void checkOutputMatrixSizes(size_t expected_size_i, size_t expected_size_j, size_t i){}

			template<size_t... OI, typename... CallArgs>
			void checkMatrixSizes(pack_indices<OI...>, CallArgs&&... args)
			{
				auto& first_output = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t out_size_i = first_output.size_i();
				size_t out_size_j = first_output.size_j();
				size_t in_size_i = input.size_i();
				size_t in_size_j = input.size_j();

				// Mirror mode: the input has the size of the output and there is no edge
				// handling, so only the interior of the output is computed. With an edge
				// mode (Pad, Cyclic, Duplicate) an input of the output's size is the normal case
				// and the whole output is computed.
				if (this->m_edge == skepu::Edge::None
				    && out_size_i == in_size_i
				    && out_size_j == in_size_j)
				{
				    this->m_mirror_mode = true;
					return;
				}
				this->m_mirror_mode = false;
				
				if (!this->isInputSizeValid(out_size_i, in_size_i, 0) && this->m_overlap[0] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output matrix row count mismatch"
					<< "\nfirst output matrix (label: " << first_output.getLabel() << ", rows: " << out_size_i << ")"
					<< "\nexpected input matrix rows: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_i, 0)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_i) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_i, 0)))
					<< "\ninput matrix (label: " << input.getLabel() << ", rows: " << colorRed(in_size_i) << ")");
				
				if (!this->isInputSizeValid(out_size_j, in_size_j, 1) && this->m_overlap[1] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output matrix col count mismatch"
					<< "\nfirst output matrix (label: " << first_output.getLabel() << ", cols: " << out_size_j << ")"
					<< "\nexpected input matrix cols: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_j, 1)) :
                        this->m_edge != skepu::Edge::None ? colorRed(std::to_string(out_size_j) + " ± 2n") :
                                      colorRed(this->getAllowedInputSizeNone(out_size_j, 1)))
					<< "\ninput matrix (label: " << input.getLabel() << ", cols: " << colorRed(in_size_j) << ")");
				
				checkOutputMatrixSizes(out_size_i, out_size_j, 1, get<OI>(std::forward<CallArgs>(args)...)...);
			}

			std::string generateCallMetadata()
			{
				return "MapOverlap2D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

		public:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				size_t firstOutputRows = get<0>(std::forward<CallArgs>(args)...).total_rows();
				size_t firstOutputCols = get<0>(std::forward<CallArgs>(args)...).total_cols();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap2D: size = " << firstOutputRows << " x " << firstOutputCols);


				checkMatrixSizes(this->out_indices, std::forward<CallArgs>(args)...);

				if (this->mirrorHasNoInterior(firstOutputRows, firstOutputCols)) // (mirror mode with nothing to compute: only the random numbers are used)
				{
					this->template prepareRandom<MapOverlapFunc::randomCount>(firstOutputRows * firstOutputCols);
					return get<0>(std::forward<CallArgs>(args)...);
				}

				// Remove later
				auto &res = get<0>(std::forward<CallArgs>(args)...);
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
				// End remove

				this->selectBackend(get<0>(std::forward<CallArgs>(args)...).size());

				switch (this->checkMirrorBackend(this->m_selected_spec->activateBackend(), "MapOverlap 2D"))
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
		}; // MapOverlap2D

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapPool2D: public MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
		private:
		using typename MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>::T;

		public:
			using SkepuChainBase = MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>;
			SKEPU_CHAIN_SKELETON_SETTERS(SkepuChainBase)
			SKEPU_CHAIN(setStride, SkepuChainBase)
			void setOverlap(int, int) = delete;
			void setOverlap(int) = delete;
			std::tuple<int, int> getOverlap() const = delete;

			void setEdgeMode(Edge) = delete;
			Edge getEdgeMode() const = delete;

			void setPad(T) = delete;
			T getPad() const = delete;
			
			void setUpdateMode(UpdateMode) = delete;
			UpdateMode getUpdateMode() const = delete;

			auto &setPoolSize(int pi, int pj)
			{
				if (pi < 0 || pj < 0)
                    SKEPU_ERROR("Pool size cannot be less than 0");
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
				return *this;
			}

			auto &setPoolSize(int p)
			{
				this->setPoolSize(p, p);
				return *this;
			}

			std::tuple<int, int> getPoolSize() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1]);
			}

			auto &setPoolSizeAndStride(int pi, int pj)
			{
				this->setPoolSize(pi, pj);
				this->setStride(pi, pj);
				return *this;
			}

			auto &setPoolSizeAndStride(int p)
			{
				this->setPoolSize(p);
				this->setStride(p);
				return *this;
			}

			MapPool2D(std::string label, CUDAKernel kernel): MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>(label, kernel) {}
		};

    } // backend

} // skepu

#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_cpu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_omp.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_cl.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_cu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_hy.inl"

#endif // MAPOVERLAP_2D_H
