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
		private:
			using Base = MapOverlapPar<MapOverlapFunc, SkeletonType::MapOverlap4D>;
			using Base::InArity;
			using Base::OutArity;
			using typename Base::T;
			using typename Base::F;

		public:
			MapOverlap4D(std::string label, CUDAKernel kernel): SkeletonBase(label), m_cuda_kernel(kernel)
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
				this->m_overlap[2] = o;
				this->m_overlap[3] = o;
			}

			void setOverlap(int oi, int oj, int ok, int ol)
			{
				if (oi < 0 || oj < 0 || ok < 0 || ol < 0)
					SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
				this->m_overlap[2] = ok;
				this->m_overlap[3] = ol;
			}

			void setPoolSize(int pi, int pj, int pk, int pl)
			{
				if (pi < 0 || pj < 0 || pk < 0 || pl < 0)
					SKEPU_ERROR("Pool size cannot be less than 0");
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
				this->m_overlap[2] = pk;
				this->m_overlap[3] = pl;
			}

			std::tuple<int, int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3]);
			}

			void setStride(int si, int sj, int sk, int sl)
			{
				if (si < 0 || sj < 0 || sk < 0 || sl < 0)
					SKEPU_ERROR("Stride cannot be less than 0");
				this->m_strides[0] = si;
				this->m_strides[1] = sj;
				this->m_strides[2] = sk;
				this->m_strides[3] = sl;
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
										 size_t i, std::string const& callMetadata, First&& first, Rest&&... rest)
			{
				if (first.size_i() != expected_size_i)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid output tensor4 size i"
					<< "\nexpected output tensor4 size i: " << colorRed(expected_size_i)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size i: " << colorRed(first.size_i()) << ")");
				
				if (first.size_j() != expected_size_j)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid output tensor4 size j"
					<< "\nexpected output tensor4 size j: " << colorRed(expected_size_j)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size j: " << colorRed(first.size_j()) << ")");
				
				if (first.size_k() != expected_size_k)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid output tensor4 size k"
					<< "\nexpected output tensor4 size k: " << colorRed(expected_size_k)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size k: " << colorRed(first.size_k()) << ")");
				
				if (first.size_l() != expected_size_l)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid output tensor4 size l"
					<< "\nexpected output tensor4 size l: " << colorRed(expected_size_l)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size l: " << colorRed(first.size_l()) << ")");

				checkOutputTensor4Sizes(expected_size_i, expected_size_j, expected_size_k, expected_size_l, i+1, callMetadata, rest...);
			}

			void checkOutputTensor4Sizes(size_t expected_size_i, size_t expected_size_j, size_t expected_size_k, size_t expected_size_l, size_t i, std::string const& callMetadata){}

			template<size_t... OI, size_t... EI, typename... CallArgs>
			void checkTensor4Sizes(size_t expected_input_size_i, size_t expected_input_size_j, size_t expected_input_size_k, size_t expected_input_size_l,
								   std::string const& callMetadata, pack_indices<OI...>, pack_indices<EI...>, CallArgs&&... args)
			{
				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t first_output_size_i = firstOutput.size_i();
				size_t first_output_size_j = firstOutput.size_j();
				size_t first_output_size_k = firstOutput.size_k();
				size_t first_output_size_l = firstOutput.size_l();
				size_t input_size_i = input.size_i();
				size_t input_size_j = input.size_j();
				size_t input_size_k = input.size_k();
				size_t input_size_l = input.size_l();
				
				if (input_size_i != expected_input_size_i)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output tensor4 size i mismatch"
					<< "\nfirst output tensor4 (label: " << firstOutput.getLabel() << ", size i: " << first_output_size_i << ")"
					<< "\nexpected input tensor4 size i: " << colorRed(expected_input_size_i)
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size i: " << colorRed(input_size_i) << ")");
				
				if (input_size_j != expected_input_size_j)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output tensor4 size j mismatch"
					<< "\nfirst output tensor4 (label: " << firstOutput.getLabel() << ", size j: " << first_output_size_j << ")"
					<< "\nexpected input tensor4 size j: " << colorRed(expected_input_size_j)
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size j: " << colorRed(input_size_j) << ")");

				if (input_size_k != expected_input_size_k)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output tensor4 size k mismatch"
					<< "\nfirst output tensor4 (label: " << firstOutput.getLabel() << ", size k: " << first_output_size_k << ")"
					<< "\nexpected input tensor4 size k: " << colorRed(expected_input_size_k)
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size k: " << colorRed(input_size_k) << ")");
				
				if (input_size_l != expected_input_size_l)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output tensor4 size l mismatch"
					<< "\nfirst output tensor4 (label: " << firstOutput.getLabel() << ", size l: " << first_output_size_l << ")"
					<< "\nexpected input tensor4 size l: " << colorRed(expected_input_size_l)
					<< "\ninput tensor4 (label: " << input.getLabel() << ", size k: " << colorRed(input_size_l) << ")");
				
				checkOutputTensor4Sizes(first_output_size_i, first_output_size_j, first_output_size_k, first_output_size_l, 1, callMetadata, get<OI>(std::forward<CallArgs>(args)...)...);
			}

			std::string generateCallMetadata()
			{
				return "MapOverlap4D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

		public:
			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			auto backendDispatch(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
				size_t size_l = get<0>(std::forward<CallArgs>(args)...).size_l();

				checkTensor4Sizes(this->expectedInputSize(size_i, 0), this->expectedInputSize(size_j, 1), this->expectedInputSize(size_k, 2), this->expectedInputSize(size_l, 3),
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
		}; // MapOverlap4D

    } // backend

} // skepu

#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_cpu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_omp.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_cl.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_cu.inl"
#include "skepu3/backend/impl/mapoverlap/mapoverlap_4d/mapoverlap_4d_hy.inl"

#endif // MAPOVERLAP_4D_H