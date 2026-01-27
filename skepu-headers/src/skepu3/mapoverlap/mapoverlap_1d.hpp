#pragma once

#include "skepu3/impl/region.hpp"
#include "skepu3/mapoverlap/mapoverlap_seq.hpp"

namespace skepu
{

    namespace impl
    {
        template<typename, typename...>
		class MapOverlap1D;

		template<typename, typename...>
		class MapPool1D;
    }

    template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 1)>
	impl::MapOverlap1D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap1D<Ret, Args...>(mapo);
	}

	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 1)>
	impl::MapPool1D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool1D<Ret, Args...>(mapo);
	}

    namespace impl
    {
        template<typename Ret, typename... Args>
		class MapOverlap1D: public MapOverlapSeq<Pool2D, Ret, Args...>, public SeqSkeletonBase
		{
		private:
			using Base = MapOverlapSeq<Pool2D, Ret, Args...>;
			using Base::randomCount; // cannot use this->randomCount as template argument
			using Base::OutArity;
			using typename Base::MapFunc;
			using typename Base::F;
			using typename Base::RegionType;
			using typename Base::T;
			Overlap m_overlapPolicy = skepu::Overlap::RowWise;
		public:

			void setOverlapMode(Overlap mode)
			{
				this->m_overlapPolicy = mode;
			}

			Overlap getOverlapMode() const
			{
				return this->m_overlapPolicy;
			}

			void setOverlap(int o)
			{
                if (o < 0)
                    SKEPU_ERROR("Overlap cannot be less than 0");
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

			std::string generateCallMetadata()
			{
				return "MapOverlap1D call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			template<typename First, typename... Rest>
			void checkOutputVectorSizes(size_t expectedSize, size_t i, First&& first, Rest&&... rest)
			{
				if (first.size() != expectedSize)
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninvalid output vector size (label: " << first.getLabel() << ", index: " << i << ", size: " << colorRed(first.size()) << ")"
					<< "\nexpected size: " << colorRed(expectedSize) << ")");

				checkOutputVectorSizes(expectedSize, i+1, rest...);
			}

			void checkOutputVectorSizes(size_t expectedSize, size_t i){}

			template<size_t... OI, size_t... EI, typename... CallArgs>
			void checkVectorSizes(pack_indices<OI...>, pack_indices<EI...>, CallArgs&&... args)
			{
				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				auto& input = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t firstOutputSize = firstOutput.size();
				size_t inputSize = input.size();
				
				if (inputSize != this->expectedInputSize(firstOutputSize, 0))
					SKEPU_ERROR(generateCallMetadata()
					<< "\ninput/output vector size mismatch"
					<< "\noutput vector (label: " << firstOutput.getLabel() << ", size: " << firstOutputSize << ")"
					<< "\nexpected input vector size: " << colorRed(this->expectedInputSize(firstOutputSize, 0))
					<< "\ninput vector (label: " << input.getLabel() << ", size: " << colorRed(inputSize) << ")");
				
				checkOutputVectorSizes(firstOutputSize, 0, get<OI>(std::forward<CallArgs>(args)...)...);
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
				
				const int overlap = this->m_overlap[0];
				const size_t size = arg.size();
				
				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap1D: input size = " << size);
				
				checkVectorSizes(this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);


				size_t edge_prng_size = (this->m_edge != Edge::None) ? overlap : 0;
				auto random_pre = this->template prepareRandom<randomCount>(edge_prng_size);
				size_t random_size = size - 2 * overlap;
				if (p != Parity::None) random_size /= 2;
				if (p == Parity::Odd && size % 2 == 1) random_size += 1;
				auto random = this->template prepareRandom<randomCount>(random_size);
				auto random_post = this->template prepareRandom<randomCount>(edge_prng_size);

				if (this->m_edge != Edge::None)
				{
					T start[3*overlap], end[3*overlap];

					for (size_t i = 0; i < overlap; ++i)
					{
						switch (this->m_edge)
						{
						case Edge::Cyclic:
							start[i] = arg[size + i  - overlap];
							end[3*overlap-1 - i] = arg[overlap-i-1];
							break;
						case Edge::Duplicate:
							start[i] = arg[0];
							end[3*overlap-1 - i] = arg[size-1];
							break;
						case Edge::Pad:
							start[i] = this->m_pad;
							end[3*overlap-1 - i] = this->m_pad;
							break;
						default:
							break;
						}
					}

					for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
						start[i] = arg[j];

					for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
						end[i] = arg[j + size - 2*overlap];

					for (size_t i = 0; i < overlap; ++i)
					{
						if (p == Parity::None || index_parity(p, i))
						{
							auto res = F::forward(this->mapFunc, Index1D{i}, random_pre, RegionType{overlap, 1, &start[i + overlap]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
						}
					}
					
					for (size_t i = overlap; i < size - overlap; ++i)
					{
						if (p == Parity::None || index_parity(p, i))
						{
							auto res = F::forward(this->mapFunc, Index1D{i}, random, RegionType{overlap, 1, &arg[i]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
						}
					}

					for (size_t i = size - overlap; i < size; ++i)
					{
						if (p == Parity::None || index_parity(p, i))
						{
							auto res = F::forward(this->mapFunc, Index1D{i}, random_post, RegionType{overlap, 1, &end[i + 2 * overlap - size]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
						}
					}
				}

				else // Edge::None
				{
					size_t out_size = get<0>(std::forward<CallArgs>(args)...).size();
					for (size_t i = 0; i < out_size; ++i)
					{
						if (p == Parity::None || index_parity(p, i))
						{
							auto res = F::forward(this->mapFunc, Index1D{i}, random, RegionType{overlap, 1, &arg[i + overlap]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
						}
					}
				}
			}


			template<typename... CallArgs,
				REQUIRES(is_skepu_vector<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
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
			void checkMatrixSizes(size_t expectedInputRows, size_t expectedInputCols, std::string const& callMetadata,
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

			std::string generateColwiseCallMetadata()
			{
				return "MapOverlap1D ColWise call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply_colwise(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
				size_t inputRows = arg.total_rows();
				size_t inputCols = arg.total_cols();

				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				size_t firstOutputRows = firstOutput.total_rows();
				size_t firstOutputCols = firstOutput.total_cols();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap1D ColWise: input size = " << inputRows << " x " << inputCols);

				checkMatrixSizes(this->expectedInputSize(firstOutputRows, 0), firstOutputCols,
								 generateColwiseCallMetadata(), this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				const int overlap = this->m_overlap[0];
				size_t size = arg.size();
				T start[3*overlap], end[3*overlap];

				size_t rowWidth = arg.total_cols();
				size_t colWidth = arg.total_rows();
				size_t stride = rowWidth;

				const T *inputBegin = arg.getAddress();
				const T *inputEnd = inputBegin + size;

				for(size_t col = 0; col < arg.total_cols(); ++col)
				{
					size_t edge_prng_size = (this->m_edge != Edge::None) ? overlap : 0;
					auto random_pre = this->template prepareRandom<randomCount>(edge_prng_size);
					size_t random_size = colWidth - 2 * overlap;;
					if (p != Parity::None) random_size /= 2;
					if (p == Parity::Odd && colWidth % 2 == 1) random_size += 1;
					auto random = this->template prepareRandom<randomCount>(random_size);
					auto random_post = this->template prepareRandom<randomCount>(edge_prng_size);

					inputEnd = inputBegin + (rowWidth * (colWidth-1));

					

					if (this->m_edge != Edge::None)
					{
						for (size_t i = 0; i < overlap; ++i)
						{
							switch (this->m_edge)
							{
							case Edge::Cyclic:
								start[i] = inputEnd[(i+1-overlap)*stride];
								end[3*overlap-1 - i] = inputBegin[(overlap-i-1)*stride];
								break;
							case Edge::Duplicate:
								start[i] = inputBegin[0];
								end[3*overlap-1 - i] = inputEnd[0]; // hmmm...
								break;
							case Edge::Pad:
								start[i] = this->m_pad;
								end[3*overlap-1 - i] = this->m_pad;
								break;
							default:
								break;
							}
						}

						for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
							start[i] = inputBegin[j*stride];

						for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
							end[i] = inputEnd[(j - 2*overlap + 1)*stride];

						for (size_t i = 0; i < overlap; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
								auto res = F::forward(this->mapFunc, Index1D{i}, random_pre, RegionType{overlap, 1, &start[i + overlap]},
								get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
							}
						}

						for (size_t i = overlap; i < colWidth - overlap; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
								auto res = F::forward(this->mapFunc, Index1D{i}, random, RegionType{overlap, stride, &inputBegin[i*stride]},
									get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
							}
						}

						for (size_t i = colWidth - overlap; i < colWidth; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
								auto res = F::forward(this->mapFunc, Index1D{i}, random_post, RegionType{overlap, 1, &end[i + 2 * overlap - colWidth]},
									get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
							}
						}
					}

					else // Edge::None
					{
						size_t out_rows = get<0>(std::forward<CallArgs>(args)...).total_rows();
						for (size_t i = 0; i < out_rows; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
								auto res = F::forward(this->mapFunc, Index1D{i}, random, RegionType{overlap, stride, &inputBegin[(i + overlap)*stride]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
							}
						}
					}

					inputBegin += 1;
				}
			}

			std::string generateRowwiseCallMetadata()
			{
				return "MapOverlap1D RowWise call, label: " + this->getLabel() + ", Edge mode: " + to_string(this->getEdgeMode());
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply_rowwise(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);
				size_t inputRows = arg.total_rows();
				size_t inputCols = arg.total_cols();

				auto& firstOutput = get<0>(std::forward<CallArgs>(args)...);
				size_t firstOutputRows = firstOutput.total_rows();
				size_t firstOutputCols = firstOutput.total_cols();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap1D RowWise: input size = " << inputRows << " x " << inputCols);

				checkMatrixSizes(firstOutputRows, this->expectedInputSize(firstOutputCols, 0),
								 generateRowwiseCallMetadata(), this->out_indices, this->elwise_indices, std::forward<CallArgs>(args)...);

				int overlap = this->m_overlap[0];
				T start[3*overlap], end[3*overlap];

				size_t rowWidth = arg.total_cols();
				size_t stride = 1;

				const T *inputBegin = arg.getAddress();
				const T *inputEnd;

				for (size_t row = 0; row < arg.total_rows(); ++row)
				{
					size_t edge_prng_size = (this->m_edge != Edge::None) ? overlap : 0;
					auto random_pre = this->template prepareRandom<randomCount>(edge_prng_size);
					size_t random_size = rowWidth - 2 * overlap;;
					if (p != Parity::None) random_size /= 2;
					if (p == Parity::Odd && rowWidth % 2 == 1) random_size += 1;
					auto random = this->template prepareRandom<randomCount>(random_size);
					auto random_post = this->template prepareRandom<randomCount>(edge_prng_size);

					inputEnd = inputBegin + rowWidth;

					

					if (this->m_edge != Edge::None)
					{
						for (size_t i = 0; i < overlap; ++i)
						{
							switch (this->m_edge)
							{
							case Edge::Cyclic:
								start[i] = inputEnd[i  - overlap];
								end[3*overlap-1 - i] = inputBegin[overlap-i-1];
								break;
							case Edge::Duplicate:
								start[i] = inputBegin[0];
								end[3*overlap-1 - i] = inputEnd[-1];
								break;
							case Edge::Pad:
								start[i] = this->m_pad;
								end[3*overlap-1 - i] = this->m_pad;
								break;
							default:
								break;
							}
						}

						for (size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
							start[i] = inputBegin[j];

						for (size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
							end[i] = inputEnd[j - 2*overlap];

						for (size_t i = 0; i < overlap; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
								auto res = F::forward(this->mapFunc, Index1D{i}, random_pre, RegionType{overlap, stride, &start[i + overlap]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
							}
						}

						for (size_t i = overlap; i < rowWidth - overlap; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
								auto res = F::forward(this->mapFunc, Index1D{i}, random, RegionType{overlap, stride, &inputBegin[i]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
							}
						}

						for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
							 	auto res = F::forward(this->mapFunc, Index1D{i}, random_post, RegionType{overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
							}
						}
					}

					else // Edge::None
					{
						size_t out_cols = get<0>(std::forward<CallArgs>(args)...).total_cols();
						for (size_t i = 0; i < out_cols; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
								auto res = F::forward(this->mapFunc, Index1D{i}, random, RegionType{overlap, stride, &inputBegin[i + overlap]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
							}
						}
					}

					inputBegin += rowWidth;
				}
			}


			template<typename... CallArgs,
				REQUIRES(is_skepu_matrix<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				switch (this->m_overlapPolicy)
				{
					case Overlap::ColWise:
						if (this->m_updateMode == UpdateMode::Normal)
						{
							this->apply_colwise(Parity::None, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
						{
							DEBUG_TEXT_LEVEL1("Red");
							this->apply_colwise(Parity::Odd, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
						{
							DEBUG_TEXT_LEVEL1("Black");
							this->apply_colwise(Parity::Even, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
						}
						break;

					case Overlap::RowWise:
						if (this->m_updateMode == UpdateMode::Normal)
						{
							this->apply_rowwise(Parity::None, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
						{
							DEBUG_TEXT_LEVEL1("Red");
							this->apply_rowwise(Parity::Odd, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
						{
							DEBUG_TEXT_LEVEL1("Black");
							this->apply_rowwise(Parity::Even, this->out_indices, this->elwise_indices, this->any_indices, this->const_indices, std::forward<CallArgs>(args)...);
						}
						break;

					default:
						SKEPU_ERROR("MapOverlap: Invalid overlap policy");
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}

//		protected:
			MapFunc mapFunc;
			MapOverlap1D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::Duplicate;
			}

			friend MapOverlap1D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};

		template<typename Ret, typename... Args>
		class MapPool1D: public MapOverlap1D<Ret, Args...>
		{
			using MapFunc = std::function<Ret(Args...)>;

		public:
			void setOverlap(size_t) = delete;
			void setEdgeMode(Edge) = delete;
			void setUpdateMode(UpdateMode) = delete;

			void setPoolSize(size_t p)
			{

			}

			MapPool1D(MapFunc map): MapOverlap1D<Ret, Args...>(map) {}
		};

    } // impl

    
} // skepu