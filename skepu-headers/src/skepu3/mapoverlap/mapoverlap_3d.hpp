#pragma once

#include "skepu3/impl/region.hpp"
#include "skepu3/mapoverlap/mapoverlap_seq.hpp"

namespace skepu
{
    namespace impl
    {
        template<typename, typename...>
		class MapOverlap3D;

        template<typename, typename...>
		class MapPool3D;
    }

    template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 3)>
	impl::MapOverlap3D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap3D<Ret, Args...>(mapo);
	}

    template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 3)>
	impl::MapPool3D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool3D<Ret, Args...>(mapo);
	}
    
    namespace impl
    {

        template<typename Ret, typename... Args>
		class MapOverlap3D: public MapOverlapSeq<Pool3D, Ret, Args...>, public SeqSkeletonBase
		{
        private:
            using Base = MapOverlapSeq<Pool3D, Ret, Args...>;
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
			}

			void setOverlap(int oi, int oj, int ok)
			{
                if (oi < 0 || oj < 0 || ok < 0)
                    SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
				this->m_overlap[2] = ok;
			}

			std::tuple<int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1], this->m_overlap[2]);
			}

			void setStride(int si, int sj, int sk)
			{
                if (si < 0 || sj < 0 || sk < 0)
                    SKEPU_ERROR("Stride cannot be less than 0");
                this->m_strides[0] = si;
                this->m_strides[1] = sj;
                this->m_strides[2] = sk;
			}

		private:

			/*template<size_t dim>
			size_t expectedInputSize(size_t size) const
			{
				size_t side = this->m_overlap[dim]; // With Pool
				if (!isPool) side = 2*this->m_overlap[dim] + 1; // With Region

				size_t inputSize = side + this->m_strides[dim] * (size - 1);

				if (isPool && this->m_edge != Edge::None)
					inputSize -= (side - 1) * 2;
				else if (!isPool && this->m_edge != Edge::None)
					inputSize -= 2 * this->m_overlap[dim];

				return inputSize;
			}*/

			template<typename First, typename... Rest>
			void checkOutputTensor3Sizes(size_t expected_size_i, size_t expected_size_j, size_t expected_size_k, size_t i, std::string const& callMetadata, First&& first, Rest&&... rest)
			{
				if (first.size_i() != expected_size_i)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid output tensor3 size i"
					<< "\nexpected output tensor3 size i: " << colorRed(expected_size_i)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size i: " << colorRed(first.size_i()) << ")");
				
				if (first.size_j() != expected_size_j)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid output tensor3 size j"
					<< "\nexpected output tensor3 size j: " << colorRed(expected_size_j)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size j: " << colorRed(first.size_j()) << ")");
				
				if (first.size_k() != expected_size_k)
					SKEPU_ERROR(callMetadata
					<< "\ninvalid output tensor3 size k"
					<< "\nexpected output tensor3 size k: " << colorRed(expected_size_k)
					<< "\noutput tensor (label: " << first.getLabel() << ", index: " << i << ", size k: " << colorRed(first.size_k()) << ")");

				checkOutputTensor3Sizes(expected_size_i, expected_size_j, expected_size_k, i+1, callMetadata, rest...);
			}

			void checkOutputTensor3Sizes(size_t expected_size_i, size_t expected_size_j, size_t expected_size_k, size_t i, std::string const& callMetadata){}

			template<size_t... OI, size_t... EI, typename... CallArgs>
			void checkTensor3Sizes(size_t expected_input_size_i, size_t expected_input_size_j, size_t expected_input_size_k, std::string const& callMetadata,
								   pack_indices<OI...>, pack_indices<EI...>, CallArgs&&... args)
			{
				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t first_output_size_i = firstOutput.size_i();
				size_t first_output_size_j = firstOutput.size_j();
				size_t first_output_size_k = firstOutput.size_k();
				size_t input_size_i = input.size_i();
				size_t input_size_j = input.size_j();
				size_t input_size_k = input.size_k();
				
				if (input_size_i != expected_input_size_i)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output tensor3 size i mismatch"
					<< "\nfirst output tensor3 (label: " << firstOutput.getLabel() << ", size i: " << first_output_size_i << ")"
					<< "\nexpected input tensor3 size i: " << colorRed(expected_input_size_i)
					<< "\ninput tensor3 (label: " << input.getLabel() << ", size i: " << colorRed(input_size_i) << ")");
				
				if (input_size_j != expected_input_size_j)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output tensor3 size j mismatch"
					<< "\nfirst output tensor3 (label: " << firstOutput.getLabel() << ", size j: " << first_output_size_j << ")"
					<< "\nexpected input tensor3 size j: " << colorRed(expected_input_size_j)
					<< "\ninput tensor3 (label: " << input.getLabel() << ", size j: " << colorRed(input_size_j) << ")");

				if (input_size_k != expected_input_size_k)
					SKEPU_ERROR(callMetadata
					<< "\ninput/output tensor3 size k mismatch"
					<< "\nfirst output tensor3 (label: " << firstOutput.getLabel() << ", size k: " << first_output_size_k << ")"
					<< "\nexpected input tensor3 size k: " << colorRed(expected_input_size_k)
					<< "\ninput tensor3 (label: " << input.getLabel() << ", size k: " << colorRed(input_size_k) << ")");
				
				checkOutputTensor3Sizes(first_output_size_i, first_output_size_j, first_output_size_k, 1, callMetadata, get<OI>(std::forward<CallArgs>(args)...)...);
			}

			std::string generateCallMetadata()
			{
				return "MapOverlap3D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap3D: size = " << size_i << " x " << size_j << " x" << size_k);

				checkTensor3Sizes(this->expectedInputSize(size_i, 0), this->expectedInputSize(size_j, 1), this->expectedInputSize(size_k, 2),
				generateCallMetadata(), this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t offset[3];
				for (size_t d = 0; d < 3; ++d)
				{
					if (!this->isPool && this->m_edge != Edge::None)
					{
						offset[d] = 0;
					}
					else if (!this->isPool && this->m_edge == Edge::None)
					{
						offset[d] = this->m_overlap[d];
					}
					else if (this->isPool && this->m_edge != Edge::None)
					{
						offset[d] = -this->m_overlap[d] / 2;
					}
					else if (this->isPool && this->m_edge == Edge::None)
					{
						offset[d] = 0;
					}
				}

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_edge, this->m_pad};

				auto random = this->template prepareRandom<randomCount>(size_i * size_j * size_k);

				for (size_t i = 0; i < size_i; i++)
					for (size_t j = 0; j < size_j; j++)
						for (size_t k = 0; k < size_k; k++)
							if (p == Parity::None || index_parity(p, i, j, k))
							{
								region.idx = Index3D{
									(i + offset[0]) * this->m_strides[0],
									(j + offset[1]) * this->m_strides[1],
									(k + offset[2]) * this->m_strides[2]
								};
								auto res = F::forward(this->mapFunc, region.idx, random, region,
									get<AI>(std::forward<CallArgs>(args)...).hostProxy()...,
									get<CI>(std::forward<CallArgs>(args)...)...
								);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j, k)..., res);
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
			MapOverlap3D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::Duplicate;
			}

			friend MapOverlap3D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};


        template<typename Ret, typename... Args>
		class MapPool3D: public MapOverlap3D<Ret, Args...>
		{
			using MapFunc = std::function<Ret(Args...)>;

		public:
			void setOverlap(size_t, size_t, size_t) = delete;
		//	void setEdgeMode(Edge) = delete;
			void setUpdateMode(UpdateMode) = delete;

			void setPoolSize(int pi, int pj, int pk)
			{
                if (pi < 0 || pj < 0 || pk < 0)
                    SKEPU_ERROR("Pool size cannot be less than 0");
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
				this->m_overlap[2] = pk;
			}

			MapPool3D(MapFunc map): MapOverlap3D<Ret, Args...>(map) {}
		};
    } // impl

} // skepu