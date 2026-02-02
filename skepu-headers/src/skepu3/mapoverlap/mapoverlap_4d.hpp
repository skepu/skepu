#pragma once

#include "skepu3/impl/region.hpp"
#include "skepu3/mapoverlap/mapoverlap_seq.hpp"

namespace skepu
{

    namespace impl
    {
        template<typename, typename...>
		class MapOverlap4D;

		template<typename, typename...>
		class MapPool4D;
    }

    template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 4)>
	impl::MapOverlap4D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap4D<Ret, Args...>(mapo);
	}

    template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 4)>
	impl::MapPool4D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool4D<Ret, Args...>(mapo);
	}

    namespace impl
    {
        template<typename Ret, typename... Args>
		class MapOverlap4D: public MapOverlapSeq<Pool4D, Ret, Args...>, public SeqSkeletonBase
		{
        private:
			using Base = MapOverlapSeq<Pool4D, Ret, Args...>;
			using Base::randomCount; // cannot use this->randomCount as template argument
			using Base::OutArity;
			using typename Base::MapFunc;
			using typename Base::F;
			using typename Base::RegionType;
			using typename Base::T;

		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}

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

		private:
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

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
				size_t size_l = get<0>(std::forward<CallArgs>(args)...).size_l();

				const std::string Name = this->isPool ? "MapPool4D" : "MapOverlap4D";
				DEBUG_TEXT_LEVEL1("Native C++ " << Name << ": size = " << size_i << " x " << size_j << " x " << size_k << " x " << size_l);
				DEBUG_TEXT_LEVEL1("Native C++ " << Name << ": kernel = " << this->m_overlap[0] << " x " << this->m_overlap[1] << " x " << this->m_overlap[2] << " x " << this->m_overlap[3]);
				DEBUG_TEXT_LEVEL1("Native C++ " << Name << ": strides = " << this->m_strides[0] << " x " << this->m_strides[1] << " x " << this->m_strides[2] << " x " << this->m_strides[3]);

				checkTensor4Sizes(this->expectedInputSize(size_i, 0), this->expectedInputSize(size_j, 1), this->expectedInputSize(size_k, 2), this->expectedInputSize(size_l, 3),
				generateCallMetadata(), this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				SKEPU_TRACE_START_EVENT(trace_handle);
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3], this->m_edge, this->m_pad};

				auto random = this->template prepareRandom<randomCount>(size_i * size_j * size_k * size_l);

				for (size_t i = 0; i < size_i; i++)
					for (size_t j = 0; j < size_j; j++)
						for (size_t k = 0; k < size_k; k++)
							for (size_t l = 0; l < size_l; l++)
								if (p == Parity::None || index_parity(p, i, j, k, l))
								{
									region.idx = Index4D{
										i * this->m_strides[0],
										j * this->m_strides[1],
										k * this->m_strides[2],
										l * this->m_strides[3]
									};
									auto res = F::forward(
										this->mapFunc,
										Index4D{i,j,k,l}, random, region,
										get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
										get<CI>(std::forward<CallArgs>(args)...)...
									);
									SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j, k, l)..., res);
								}
			}

		public:

			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				if (this->m_updateMode == UpdateMode::Normal)
				{
					this->apply(Parity::None, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->apply(Parity::Odd, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->apply(Parity::Even, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}

//		protected:
			MapFunc mapFunc;
			MapOverlap4D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::Duplicate;
			}

			friend MapOverlap4D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};


        template<typename Ret, typename... Args>
		class MapPool4D: public MapOverlap4D<Ret, Args...>
		{
			using MapFunc = std::function<Ret(Args...)>;

		public:
			void setOverlap(size_t, size_t, size_t, size_t) = delete;
			void setEdgeMode(Edge) = delete;
			void setUpdateMode(UpdateMode) = delete;

			void setPoolSize(size_t pi, size_t pj, size_t pk, size_t pl)
			{
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
				this->m_overlap[2] = pk;
				this->m_overlap[3] = pl;
			}

			MapPool4D(MapFunc map): MapOverlap4D<Ret, Args...>(map) {}
		};

    } // impl

} // skepu