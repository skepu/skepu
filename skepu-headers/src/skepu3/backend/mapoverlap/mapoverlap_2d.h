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
		private:
			using Base = MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap2D>;
			using Base::InArity;
			using Base::OutArity;
			using typename Base::T;
			using typename Base::F;

		public:
			MapOverlap2D(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}

			void setOverlap(int o)
			{
				if (o < 0)
					SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = o;
				this->m_overlap[1] = o;
			}

			void setOverlap(int y, int x)
			{
				if (y < 0 || x < 0)
					SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = y;
				this->m_overlap[1] = x;
			}

			std::tuple<int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1]);
			}

			void setStride(int si, int sj)
			{
				if (si < 0 || sj < 0)
					SKEPU_ERROR("Stride cannot be less than 0");
				this->m_strides[0] = si;
				this->m_strides[1] = sj;
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
			void checkOutputMatrixSizes(size_t expectedSizeRow, size_t expectedSizeCol, size_t i, std::string const& callMetadata, First&& first, Rest&&... rest)
			{
				if (first.total_rows() != expectedSizeRow)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid number of output matrix rows"
					<< "\nexpected output matrix rows: " << colorRed(expectedSizeRow)
					<< "\noutput matrix (label: " << first.getLabel() << ", index: " << i << ", rows: " << colorRed(first.total_rows()) << ")");
				
				if (first.total_cols() != expectedSizeCol)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid number of output matrix cols"
					<< "\nexpected output matrix cols: " << colorRed(expectedSizeCol)
					<< "\noutput matrix (label: " << first.getLabel() << ", index: " << i << ", cols: " << colorRed(first.total_cols()) << ")");

				checkOutputMatrixSizes(expectedSizeRow, expectedSizeCol, i+1, callMetadata, rest...);
			}

			void checkOutputMatrixSizes(size_t expectedSizeRow, size_t expectedSizeCol, size_t i, std::string const& callMetadata){}

			template<size_t... OI, size_t... EI, typename... CallArgs>
			void checkMatrixSizes(size_t expectedInputRows, size_t expectedInputCols,std::string const& callMetadata,
								  pack_indices<OI...>, pack_indices<EI...>, CallArgs&&... args)
			{
				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t firstOutputRows = firstOutput.total_rows();
				size_t firstOutputCols = firstOutput.total_cols();
				size_t inputRows = input.total_rows();
				size_t inputCols = input.total_cols();
				
				if (inputRows != expectedInputRows)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output matrix row count mismatch"
					<< "\nfirst output matrix (label: " << firstOutput.getLabel() << ", rows: " << firstOutputRows << ")"
					<< "\nexpected input matrix rows: " << colorRed(expectedInputRows)
					<< "\ninput matrix (label: " << input.getLabel() << ", rows: " << colorRed(inputRows) << ")");
				
				if (inputCols != expectedInputCols)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output matrix col count mismatch"
					<< "\nfirst output matrix (label: " << firstOutput.getLabel() << ", cols: " << firstOutputCols << ")"
					<< "\nexpected input matrix cols: " << colorRed(expectedInputCols)
					<< "\ninput matrix (label: " << input.getLabel() << ", cols: " << colorRed(inputCols) << ")");
				
				checkOutputMatrixSizes(firstOutputRows, firstOutputCols, 0, callMetadata, get<OI>(std::forward<CallArgs>(args)...)...);
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


				checkMatrixSizes(this->expectedInputSize(firstOutputRows, 0), this->expectedInputSize(firstOutputCols, 1),
								 generateCallMetadata(), this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				// Remove later
				auto &res = get<0>(std::forward<CallArgs>(args)...);
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
				// End remove

				this->selectBackend(get<0>(std::forward<CallArgs>(args)...).size());

				switch (this->m_selected_spec->activateBackend())
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

    } // backend

} // skepu

#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_cpu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_omp.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_cl.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_cu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_2d/mapoverlap_2d_hy.inl"

#endif // MAPOVERLAP_2D_H