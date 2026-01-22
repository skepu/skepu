#ifndef MAPOVERLAP_1D_H
#define MAPOVERLAP_1D_H

#include "skepu3/backend/mapoverlap/mapoverlap_par.h"

namespace skepu
{

	namespace backend
	{

        /*!
		 *  \ingroup skeletons
		 */
		/*!
		 *  \class MapOverlap
		 *
		 *  \brief A class representing the MapOverlap skeleton.
		 *
		 *  This class defines the MapOverlap1D skeleton which is similar to a Map, but each element of the result (vector/matrix) is a function
		 *  of \em several adjacent elements of one or more input (vectors/matrices) that reside at a certain constant maximum distance from each other.
		 *  This class can be used to apply (1) overlap to a vector and (2) separable-overlap to a matrix (row-wise, column-wise). For
		 *  non-separable matrix overlap which considers diagonal neighbours as well besides row- and column-wise neighbours, please see \p src/MapOverlap2D.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		class MapOverlap1D: public MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap1D>, public SkeletonBase
		{
		private:
			using Base = MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap1D>;
			using Base::InArity;
			using Base::OutArity;
			using typename Base::T;
			using typename Base::F;
		public:

			MapOverlap1D(std::string label, CUDAKernel kernel, C2 k2, C3 k3, C4 k4)
			: SkeletonBase{label}, m_cuda_kernel(kernel), m_cuda_rowwise_kernel(k2), m_cuda_colwise_kernel(k3), m_cuda_colwise_multi_kernel(k4)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}

			void setOverlapMode(Overlap mode)
			{
				this->m_overlapPolicy = mode;
			}

			Overlap getOverlapMode() const
			{
				return this->m_overlapPolicy;
			}

			void setOverlap(size_t o)
			{
				this->m_overlap[0] = o;
			}

			int getOverlap() const
			{
				return this->m_overlap[0];
			}

			void setStride(int si)
			{
				if (si < 0)
					SKEPU_ERROR("Stride cannot be less than 0");
				this->m_strides[0] = si;
			}

			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}

		private:
			CUDAKernel m_cuda_kernel;
			C2 m_cuda_rowwise_kernel;
			C3 m_cuda_colwise_kernel;
			C4 m_cuda_colwise_multi_kernel;

			Overlap m_overlapPolicy = Overlap::RowWise;

		public:


		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_CPU(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


#ifdef SKEPU_OPENMP
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_OpenMP(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif

#ifdef SKEPU_CUDA
		public:


		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU(size_t deviceID, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapNumDevices_CU(size_t numDevices, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_CUDA(size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU_Col(size_t deviceID, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultiThread_CU_Col(size_t numDevices, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_CUDA(size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingleThread_CU_Row(size_t deviceID, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapMultiThread_CU_Row(size_t numDevices, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_CUDA(size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template <typename T>
			size_t getThreadNumber_CU(size_t width, size_t &numThreads, size_t deviceID);

			template <typename T>
			bool sharedMemAvailable_CU(size_t &numThreads, size_t deviceID);

#endif

#ifdef SKEPU_OPENCL
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_OpenCL(size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_OpenCL(size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_OpenCL(size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL(size_t deviceID, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapNumDevices_CL(size_t numDevices, size_t startIdx, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_Row(size_t deviceID, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_RowMulti(size_t numDevices, size_t numrows, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_Col(size_t deviceID, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void mapOverlapSingle_CL_ColMulti(size_t numDevices, size_t numcols, Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);


			template<typename T>
			int getThreadNumber_CL(size_t width, size_t numThreads, size_t deviceID);

			template<typename T>
			bool sharedMemAvailable_CL(size_t &numThreads, size_t deviceID);

#endif



#ifdef SKEPU_HYBRID
		private:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void vector_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void rowwise_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void colwise_Hybrid(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);

#endif

			std::string generateCallMetadata()
			{
				return "MapOverlap1D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			template<typename First, typename... Rest>
			void checkOutputVectorSizes(size_t const& expectedSize, size_t const& i, First&& first, Rest&&... rest) // TODO: Add index to output
			{
				if (first.size() != expectedSize)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output vector size (label: " << first.getLabel() << ", index: " << i << ", size: " << colorRed(first.size()) << ")"
					<< "\nexpected size: " << colorRed(expectedSize) << ")");

				checkOutputVectorSizes(expectedSize, i+1, rest...);
			}

			void checkOutputVectorSizes(size_t const& expectedSize, size_t const& i){}

			template<size_t... OI, size_t... EI, typename... CallArgs>
			void checkVectorSizes(pack_indices<OI...>, pack_indices<EI...>, CallArgs&&... args)
			{
				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t firstOutputSize = firstOutput.size();
				size_t inputSize = input.size();

				
				if (inputSize != this->expectedInputSize(firstOutputSize, 0))
					SKEPU_ERROR(this->generateCallMetadata()
					<< "\ninput/output vector size mismatch"
					<< "\noutput vector (label: " << firstOutput.getLabel() << ", size: " << firstOutputSize << ")"
					<< "\nexpected input vector size: " << colorRed(this->expectedInputSize(firstOutputSize, 0))
					<< "\ninput vector (label: " << input.getLabel() << ", size: " << colorRed(inputSize) << ")");
				
				this->checkOutputVectorSizes(firstOutputSize, 0, get<OI>(std::forward<CallArgs>(args)...)...);
			}

		public:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs,
				REQUIRES(is_skepu_vector<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{

				this->checkVectorSizes(this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				size_t size = get<OutArity>(std::forward<CallArgs>(args)...).size();

				this->selectBackend(size);

				switch (this->m_selected_spec->activateBackend())
				{
				case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
					this->vector_Hybrid(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->vector_CUDA(0, p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->vector_OpenCL(0, p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->vector_OpenMP(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
#endif
				default:
					this->vector_CPU(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
					break;
				}

				return get<0>(std::forward<CallArgs>(args)...);
			}

			template<typename First, typename... Rest>
			void checkOutputMatrixSizes(size_t const& expectedSizeRow, size_t const& expectedSizeCol, size_t const& i, std::string const& callMetadata, First&& first, Rest&&... rest)
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

			void checkOutputMatrixSizes(size_t const& expectedSizeRow, size_t const& expectedSizeCol, size_t const& i, std::string const& callMetadata){}

			template<size_t... OI, size_t... EI, typename... CallArgs>
			void checkMatrixSizes(size_t const& expectedInputRows, size_t const& expectedInputCols, std::string const& callMetadata,
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

			std::string generateRowwiseCallMetadata()
			{
				return "MapOverlap1D RowWise call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			std::string generateColwiseCallMetadata()
			{
				return "MapOverlap1D ColWise call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs,
				REQUIRES(is_skepu_matrix<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
				size_t inputRows = arg.total_rows();
				size_t inputCols = arg.total_cols();

				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				size_t firstOutputRows = firstOutput.total_rows();
				size_t firstOutputCols = firstOutput.total_cols();

				if (this->m_overlapPolicy == Overlap::ColWise)
					checkMatrixSizes(this->expectedInputSize(firstOutputRows, 0), firstOutputCols, generateColwiseCallMetadata(),
									 this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);
				else // RowWise
					checkMatrixSizes(firstOutputRows, this->expectedInputSize(firstOutputCols, 0), generateRowwiseCallMetadata(),
									 this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();

				this->selectBackend(get<OutArity>(std::forward<CallArgs>(args)...).size());

				switch (this->m_overlapPolicy)
				{
					case Overlap::ColWise:
						switch (this->m_selected_spec->activateBackend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->colwise_Hybrid(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->colwise_CUDA(size_j, p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->colwise_OpenCL(size_j, p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->colwise_OpenMP(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->colwise_CPU(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;

					case Overlap::RowWise:
						switch (this->m_selected_spec->activateBackend())
						{
						case Backend::Type::Hybrid:
#ifdef SKEPU_HYBRID
							this->rowwise_Hybrid(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
							this->rowwise_CUDA(size_i, p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
							this->rowwise_OpenCL(size_i, p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
							this->rowwise_OpenMP(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
#endif
						default:
							this->rowwise_CPU(p, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
							break;
						}
						break;

					default:
						SKEPU_ERROR("MapOverlap: Invalid overlap mode");
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
		}; // MapOverlap1D

    } // backend

} // skepu

#include "skepu3/backend/impl/mapoverlap/mapoverlap_1d/mapoverlap_1d_cpu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_1d/mapoverlap_1d_omp.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_1d/mapoverlap_1d_cl.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_1d/mapoverlap_1d_cu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_1d/mapoverlap_1d_hy.inl"

#endif // MAPOVERLAP_1D_H