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
				//this->m_strides = StrideList<4>(si, sj, sk, sl);
			}

		private:
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

				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() < size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() < size_j) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_k() < size_k) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_l() < size_l)...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != this->expectedInputSize(size_i, 0)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != this->expectedInputSize(size_j, 1)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_k() != this->expectedInputSize(size_k, 2)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_l() != this->expectedInputSize(size_l, 3))...))
					SKEPU_ERROR("Non-matching input container sizes");

				SKEPU_TRACE_START_EVENT(trace_handle);
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_overlap[2], this->m_overlap[3], this->m_edge, this->m_pad};

		/*		Index4D start{0, 0, 0, 0}, end{size_i, size_j, size_k, size_l};
				if (this->m_edge == Edge::None)
				{
					start = Index4D{(size_t)this->m_overlap_i, (size_t)this->m_overlap_j, (size_t)this->m_overlap_k, (size_t)this->m_overlap_l};
					end = Index4D{size_i - this->m_overlap_i, size_j - this->m_overlap_j, size_k - this->m_overlap_k, size_l - this->m_overlap_l};
				}

				size_t final_size = (end.i - start.i) * (end.j - start.j) * (end.k - start.k) * (end.l - start.l);*/
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