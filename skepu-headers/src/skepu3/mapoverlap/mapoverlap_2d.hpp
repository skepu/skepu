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
        private:
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

			void setOverlap(int o)
			{
                if (o < 0)
                    SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = o;
				this->m_overlap[1] = o;
			}

			void setOverlap(int i, int j)
			{
                if (i < 0 || j < 0)
                    SKEPU_ERROR("Overlap cannot be less than 0");
				this->m_overlap[0] = i;
				this->m_overlap[1] = j;
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
				//this->m_strides = StrideList<2>(si, sj);
			}
			
            // TODO: Discuss with August.
			void setStride(size_t si, size_t sj, size_t sk)
			{
				this->m_strides = StrideList<3>(si, sj, sk);
			}

		private:

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap2D: size = " << size_i << " x " << size_j);

				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != this->expectedInputSize(size_i, 0)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != this->expectedInputSize(size_j, 1)) ...))
					SKEPU_ERROR("Non-matching input container sizes");

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_edge, this->m_pad};

		/*		Index2D start{0, 0}, end{size_i, size_j};
				if (isPool)
				{
					end = Index2D{size_i, size_j};
				}
				else if (this->m_edge == Edge::None)
				{
					start = Index2D{(size_t)this->m_overlap[0], (size_t)this->m_overlap[1]};
					end = Index2D{size_i - this->m_overlap[0], size_j - this->m_overlap[1]};
				}*/

				auto random = this->template prepareRandom<randomCount>(size_i * size_j);

				Index2D offset = {0, 0};
				if (this->m_edge == skepu::Edge::None)
				{
					offset.row = this->m_overlap[0];
					offset.col = this->m_overlap[1];
				}

				for (size_t i = 0; i < size_i; ++i)
					for (size_t j = 0; j < size_j; ++j)
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

		public:
			void setOverlap(size_t, size_t) = delete;
			void setEdgeMode(Edge) = delete;
			void setUpdateMode(UpdateMode) = delete;

			void setPoolSize(size_t pi, size_t pj)
			{
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
			}

			MapPool2D(MapFunc map): MapOverlap2D<Ret, Args...>(map) {}
		};

    } // impl

} // skepu