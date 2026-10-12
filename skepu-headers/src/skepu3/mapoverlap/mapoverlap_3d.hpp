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
        protected:
            using Base = MapOverlapSeq<Pool3D, Ret, Args...>;
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

		private:

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

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t out_size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t out_size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t out_size_k = get<0>(std::forward<CallArgs>(args)...).size_k();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap3D: size = " << out_size_i << " x " << out_size_j << " x" << out_size_k);

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				// Mirror mode: the edge mode None with an input of the size of the output; only the interior of the output is computed
				// (the other elements keep their values) and the stream of random numbers is used by the elements that are computed
				this->m_mirror_mode = !this->isPool && this->m_edge == Edge::None && out_size_i == arg.size_i() && out_size_j == arg.size_j() && out_size_k == arg.size_k();
				if (!this->m_mirror_mode)
					checkTensor3Sizes(this->out_indices, std::forward<CallArgs>(args)...);
				else if (this->mirrorHasNoInterior(out_size_i, out_size_j, out_size_k)) // (nothing to compute: only the random numbers are used)
				{
					this->template prepareRandom<randomCount>(out_size_i * out_size_j * out_size_k);
					return;
				}

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_edge, this->m_pad};

				auto random = this->template prepareRandom<randomCount>(out_size_i * out_size_j * out_size_k);

                Index3D offset{0, 0, 0};
                if (!this->isPool && !this->m_mirror_mode)
                {
                    if (this->m_edge == Edge::None)
                    {
                        offset.i = this->m_overlap[0];
                        offset.j = this->m_overlap[1];
                        offset.k = this->m_overlap[2];
                    }
                    else
                    {
                        offset.i = (arg.size_i() - out_size_i) / 2;
                        offset.j = (arg.size_j() - out_size_j) / 2;
                        offset.k = (arg.size_k() - out_size_k) / 2;
                    }
                }

				const size_t first_i = this->m_mirror_mode ? this->m_overlap[0] : 0, first_j = this->m_mirror_mode ? this->m_overlap[1] : 0, first_k = this->m_mirror_mode ? this->m_overlap[2] : 0;
				for (size_t i = first_i; i + first_i < out_size_i; i++)
					for (size_t j = first_j; j + first_j < out_size_j; j++)
						for (size_t k = first_k; k + first_k < out_size_k; k++)
							if (p == Parity::None || index_parity(p, i, j, k))
							{
								region.idx = Index3D{
									(i + offset.i) * this->m_strides[0],
									(j + offset.j) * this->m_strides[1],
									(k + offset.k) * this->m_strides[2]
								};
								auto res = F::forward(this->mapFunc, Index3D{i,j,k}, random, region,
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
            using typename MapOverlap3D<Ret, Args...>::T;

		public:
			using SkepuChainBase = MapOverlap3D<Ret, Args...>;
			SKEPU_CHAIN_SKELETON_SETTERS(SkepuChainBase)
			SKEPU_CHAIN(setStride, SkepuChainBase)
			void setOverlap(int, int, int) = delete;
            void setOverlap(int) = delete;
			std::tuple<int, int, int> getOverlap() const = delete;

			void setEdgeMode(Edge) = delete;
            void getEdgeMode() const = delete;

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

			// For LSP CodeComplete
			template<typename... CallArgs>
			auto operator()(CallArgs&&... args)
			{
				return MapOverlap3D<Ret, Args...>::operator()(std::forward<CallArgs>(args)...);
			}

			MapPool3D(MapFunc map): MapOverlap3D<Ret, Args...>(map) {}
		};
    } // impl

} // skepu