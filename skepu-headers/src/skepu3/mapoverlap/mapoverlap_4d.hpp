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
        protected:
			using Base = MapOverlapSeq<Pool4D, Ret, Args...>;
			using Base::randomCount; // cannot use this->randomCount as template argument
			using Base::OutArity;
			using typename Base::MapFunc;
			using typename Base::F;
			using typename Base::RegionType;
			using typename Base::T;

		public:
			SKEPU_CHAIN_SKELETON_SETTERS(SeqSkeletonBase)
			SKEPU_CHAIN(setEdgeMode, Base)
			SKEPU_CHAIN(setPad, Base)
			SKEPU_CHAIN(setUpdateMode, Base)

			auto &setOverlap(int o)
			{
                if (o < 0)
                    SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = o;
				this->m_overlap[1] = o;
				this->m_overlap[2] = o;
				this->m_overlap[3] = o;
				return *this;
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

		private:
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

                if (!this->isInputSizeValid(out_size_i, in_size_i, 0) && this->m_overlap[0] > 0)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output tensor4 size i mismatch"
					<< "\nfirst output tensor4 (label: " << first_output.getLabel() << ", size i: " << out_size_i << ")"
					<< "\nexpected tensor4 input size i: "
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

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t out_size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t out_size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t out_size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
				size_t out_size_l = get<0>(std::forward<CallArgs>(args)...).size_l();

				const std::string Name = this->isPool ? "MapPool4D" : "MapOverlap4D";
				DEBUG_TEXT_LEVEL1("Native C++ " << Name << ": size = " << out_size_i << " x " << out_size_j << " x " << out_size_k << " x " << out_size_l);
				DEBUG_TEXT_LEVEL1("Native C++ " << Name << ": kernel = " << this->m_overlap[0] << " x " << this->m_overlap[1] << " x " << this->m_overlap[2] << " x " << this->m_overlap[3]);
				DEBUG_TEXT_LEVEL1("Native C++ " << Name << ": strides = " << this->m_strides[0] << " x " << this->m_strides[1] << " x " << this->m_strides[2] << " x " << this->m_strides[3]);

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				// Mirror mode: the edge mode None with an input of the size of the output; only the interior of the output is computed
				// (the other elements keep their values) and the stream of random numbers is used by the elements that are computed
				this->m_mirror_mode = !this->isPool && this->m_edge == Edge::None && out_size_i == arg.size_i() && out_size_j == arg.size_j() && out_size_k == arg.size_k() && out_size_l == arg.size_l();
				if (!this->m_mirror_mode)
					checkTensor4Sizes(this->out_indices, std::forward<CallArgs>(args)...);
				else if (this->mirrorHasNoInterior(out_size_i, out_size_j, out_size_k, out_size_l)) // (nothing to compute: only the random numbers are used)
				{
					this->template prepareRandom<randomCount>(out_size_i * out_size_j * out_size_k * out_size_l);
					return;
				}

				SKEPU_TRACE_START_EVENT(trace_handle);

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3], this->m_edge, this->m_pad};

				auto random = this->template prepareRandom<randomCount>(out_size_i * out_size_j * out_size_k * out_size_l);

			long offset[4] = {0, 0, 0, 0};
			if (!this->isPool && !this->m_mirror_mode)
			{
				offset[0] = region_offset(out_size_i, arg.size_i(), this->m_overlap[0], this->m_edge);
				offset[1] = region_offset(out_size_j, arg.size_j(), this->m_overlap[1], this->m_edge);
				offset[2] = region_offset(out_size_k, arg.size_k(), this->m_overlap[2], this->m_edge);
				offset[3] = region_offset(out_size_l, arg.size_l(), this->m_overlap[3], this->m_edge);
			}

				const size_t first_i = this->m_mirror_mode ? this->m_overlap[0] : 0, first_j = this->m_mirror_mode ? this->m_overlap[1] : 0;
				const size_t first_k = this->m_mirror_mode ? this->m_overlap[2] : 0, first_l = this->m_mirror_mode ? this->m_overlap[3] : 0;
				for (size_t i = first_i; i + first_i < out_size_i; i++)
					for (size_t j = first_j; j + first_j < out_size_j; j++)
						for (size_t k = first_k; k + first_k < out_size_k; k++)
							for (size_t l = first_l; l + first_l < out_size_l; l++)
								if (p == Parity::None || index_parity(p, i, j, k, l))
								{
									region.idx = Index4D{
										(size_t)((long)(i * this->m_strides[0]) + offset[0]),
										(size_t)((long)(j * this->m_strides[1]) + offset[1]),
										(size_t)((long)(k * this->m_strides[2]) + offset[2]),
										(size_t)((long)(l * this->m_strides[3]) + offset[3])
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
				this->checkUpdateModeRandom(randomCount);
				
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
			using typename MapOverlap4D<Ret, Args...>::T;

		public:
			using SkepuChainBase = MapOverlap4D<Ret, Args...>;
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

			// For LSP CodeComplete (the return type is not deduced from the body: copying a MapPool tests if it converts to a
			// std::function, which must not instantiate the body)
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args) -> decltype(MapOverlap4D<Ret, Args...>::operator()(std::forward<CallArgs>(args)...))
			{
				return MapOverlap4D<Ret, Args...>::operator()(std::forward<CallArgs>(args)...);
			}

			MapPool4D(MapFunc map): MapOverlap4D<Ret, Args...>(map) {}
		};

    } // impl

} // skepu