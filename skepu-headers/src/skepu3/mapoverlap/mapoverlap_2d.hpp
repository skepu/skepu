#pragma once

#include "skepu3/impl/region.hpp"
#include "skepu3/mapoverlap/mapoverlap_seq.hpp"

namespace skepu
{
    
    namespace impl
    {
        template<typename, typename...>
        class MapOverlap2D;

        template<typename, typename...>
		class MapPool2D;
    }

    template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 2)>
	impl::MapOverlap2D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap2D<Ret, Args...>(mapo);
	}

    template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 2)>
	impl::MapPool2D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool2D<Ret, Args...>(mapo);
	}

    namespace impl
    {
        template<typename Ret, typename... Args>
		class MapOverlap2D: public MapOverlapSeq<Pool2D, Ret, Args...>, public SeqSkeletonBase
		{
        protected:
            using Base = MapOverlapSeq<Pool2D, Ret, Args...>;
			using Base::randomCount; // cannot use this->randomCount as template argument
			using Base::OutArity;
			using typename Base::MapFunc;
			using typename Base::F;
			using typename Base::RegionType;
			using typename Base::T;


		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}

			void setOverlap(int oi, int oj)
			{
                if (oi < 0 || oj < 0)
                    SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
			}

			void setOverlap(int o)
			{
				this->setOverlap(o, o);
			}

			std::tuple<int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1]);
			}

			void setStride(int si, int sj)
			{
                if (si < 0 || sj < 0)
                    SKEPU_ERROR("Stride cannot be less than 0")
                this->m_strides[0] = si;
                this->m_strides[1] = sj;
			}

			void setStride(int s)
			{
				this->setStride(s, s);
			}

			std::tuple<int, int> getStride() const
			{
				return std::make_tuple(this->m_strides[0], this->m_strides[1]);
			}
			

		private:

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
				
				if (!this->isInputSizeValid(out_size_i, in_size_i, 0))
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output matrix row count mismatch"
					<< "\nfirst output matrix (label: " << first_output.getLabel() << ", rows: " << out_size_i << ")"
					<< "\nexpected input matrix rows: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_i, 0)) :
                                      colorRed(this->getAllowedInputSizeNone(out_size_i, 0)))
					<< "\ninput matrix (label: " << input.getLabel() << ", rows: " << colorRed(in_size_i) << ")");
				
				if (!this->isInputSizeValid(out_size_j, in_size_j, 1))
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output matrix col count mismatch"
					<< "\nfirst output matrix (label: " << first_output.getLabel() << ", cols: " << out_size_j << ")"
					<< "\nexpected input matrix cols: "
                    << (this->isPool ? ">= " + colorRed(this->getSmallestAllowedInputSizePool(out_size_j, 1)) :
                                      colorRed(this->getAllowedInputSizeNone(out_size_j, 1)))
					<< "\ninput matrix (label: " << input.getLabel() << ", cols: " << colorRed(in_size_j) << ")");
				
				checkOutputMatrixSizes(out_size_i, out_size_j, 0, get<OI>(std::forward<CallArgs>(args)...)...);
			}

			std::string generateCallMetadata()
			{
				return "MapOverlap2D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t out_size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t out_size_j = get<0>(std::forward<CallArgs>(args)...).size_j();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap2D: size = " << out_size_i << " x " << out_size_j);

				checkMatrixSizes(this->out_indices, std::forward<CallArgs>(args)...);

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_edge, this->m_pad};

				auto random = this->template prepareRandom<randomCount>(out_size_i * out_size_j);

				Index2D offset{0, 0};
                if (!this->isPool)
                {
                    if (this->m_edge == Edge::None)
                    {
                        offset.row = this->m_overlap[0];
                        offset.col = this->m_overlap[1];
                    }
                    else
                    {
                        offset.row = (arg.size_i() - out_size_i) / 2;
                        offset.col = (arg.size_j() - out_size_j) / 2;
                    }
                }

				for (size_t i = 0; i < out_size_i; ++i)
					for (size_t j = 0; j < out_size_j; ++j)
						if (p == Parity::None || index_parity(p, i, j))
						{
							region.idx = Index2D{(i + offset.row) * this->m_strides[0], (j + offset.col) * this->m_strides[1]};
							auto res = F::forward(this->mapFunc, Index2D{i,j}, random, region, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, j)..., res);
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
			MapOverlap2D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::Duplicate;
			}

			friend MapOverlap2D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};

        template<typename Ret, typename... Args>
		class MapPool2D: public MapOverlap2D<Ret, Args...>
		{
			using MapFunc = std::function<Ret(Args...)>;

			using typename MapOverlap2D<Ret, Args...>::T;

		public:
			void setOverlap(int, int) = delete;
			void setOverlap(int) = delete;
			std::tuple<int, int> getOverlap() const = delete;

			void setEdgeMode(Edge) = delete;
			Edge getEdgeMode() const = delete;

			void setPad(T) = delete;
			T getPad() const = delete;
			
			void setUpdateMode(UpdateMode) = delete;
			UpdateMode getUpdateMode() const = delete;

			void setPoolSize(int pi, int pj)
			{
				if (pi < 0 || pj < 0)
                    SKEPU_ERROR("Pool size cannot be less than 0");
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
			}

			void setPoolSize(int p)
			{
				this->setPoolSize(p, p);
			}

			std::tuple<int, int> getPoolSize() const
			{
				return std::make_tuple(this->m_overlap[0], this->m_overlap[1]);
			}

			void setPoolSizeAndStride(int pi, int pj)
			{
				this->setPoolSize(pi, pj);
				this->setStride(pi, pj);
			}

			void setPoolSizeAndStride(int p)
			{
				this->setPoolSize(p);
				this->setStride(p);
			}

			MapPool2D(MapFunc map): MapOverlap2D<Ret, Args...>(map) {}
		};

    } // impl

} // skepu