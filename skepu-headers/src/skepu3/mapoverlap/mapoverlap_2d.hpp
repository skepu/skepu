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

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t firstOutputRows = get<0>(std::forward<CallArgs>(args)...).total_rows();
				size_t firstOutputCols = get<0>(std::forward<CallArgs>(args)...).total_cols();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap2D: size = " << firstOutputRows << " x " << firstOutputCols);


				checkMatrixSizes(this->expectedInputSize(firstOutputRows, 0), this->expectedInputSize(firstOutputCols, 1),
								 generateCallMetadata(), this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				RegionType region{arg, this->m_overlap[0], this->m_overlap[1], this->m_edge, this->m_pad};

		/*		Index2D start{0, 0}, end{firstOutputRows, firstOutputCols};
				if (isPool)
				{
					end = Index2D{firstOutputRows, firstOutputCols};
				}
				else if (this->m_edge == Edge::None)
				{
					start = Index2D{(size_t)this->m_overlap[0], (size_t)this->m_overlap[1]};
					end = Index2D{firstOutputRows - this->m_overlap[0], firstOutputCols - this->m_overlap[1]};
				}*/

				auto random = this->template prepareRandom<randomCount>(firstOutputRows * firstOutputCols);

				Index2D offset = {0, 0};
				if (this->m_edge == skepu::Edge::None)
				{
					offset.row = this->m_overlap[0];
					offset.col = this->m_overlap[1];
				}

				for (size_t i = 0; i < firstOutputRows; ++i)
					for (size_t j = 0; j < firstOutputCols; ++j)
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