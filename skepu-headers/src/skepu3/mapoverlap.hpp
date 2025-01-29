#pragma once

#include "skepu3/impl/common.hpp"
#include "skepu3/impl/region.hpp"

namespace skepu
{
	namespace impl
	{
		template<typename, typename...>
		class MapOverlap1D;

		template<typename, typename...>
		class MapOverlap2D;

		template<typename, typename...>
		class MapOverlap3D;

		template<typename, typename...>
		class MapOverlap4D;

		template<typename, typename...>
		class MapPool1D;

		template<typename, typename...>
		class MapPool2D;

		template<typename, typename...>
		class MapPool3D;

		template<typename, typename...>
		class MapPool4D;
	}

	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 1)>
	impl::MapOverlap1D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap1D<Ret, Args...>(mapo);
	}
	/*
	// For function pointers
	template<typename Ret, typename... Args>
	impl::MapOverlap1D<Ret, Args...> MapOverlap(Ret(*mapo)(Region1D<T>, Args...))
	{
		return MapOverlapWrapper((std::function<Ret(Region1D<T>, Args...)>)mapo);
	}*/


	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 2)>
	impl::MapOverlap2D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap2D<Ret, Args...>(mapo);
	}

	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 3)>
	impl::MapOverlap3D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap3D<Ret, Args...>(mapo);
	}

	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 4)>
	impl::MapOverlap4D<Ret, Args...> MapOverlapWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapOverlap4D<Ret, Args...>(mapo);
	}

	// For function pointers
	template<typename Ret, typename... Args>
	auto MapOverlap(Ret(*mapo)(Args...)) -> decltype(MapOverlapWrapper((std::function<Ret(Args...)>)mapo))
	{
		return MapOverlapWrapper((std::function<Ret(Args...)>)mapo);
	}

	// For lambdas and functors
	template<typename T>
	auto MapOverlap(T mapo) -> decltype(MapOverlapWrapper(lambda_cast(mapo)))
	{
		return MapOverlapWrapper(lambda_cast(mapo));
	}





	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 1)>
	impl::MapPool1D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool1D<Ret, Args...>(mapo);
	}
	/*
	// For function pointers
	template<typename Ret, typename... Args>
	impl::MapPool1D<Ret, Args...> MapPool(Ret(*mapo)(Region1D<T>, Args...))
	{
		return MapPoolWrapper((std::function<Ret(Region1D<T>, Args...)>)mapo);
	}*/


	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 2)>
	impl::MapPool2D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool2D<Ret, Args...>(mapo);
	}

	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 3)>
	impl::MapPool3D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool3D<Ret, Args...>(mapo);
	}

	template<typename Ret, typename... Args, REQUIRES(mapoverlap_dimensionality<Args...>::value == 4)>
	impl::MapPool4D<Ret, Args...> MapPoolWrapper(std::function<Ret(Args...)> mapo)
	{
		return impl::MapPool4D<Ret, Args...>(mapo);
	}

	// For function pointers
	template<typename Ret, typename... Args>
	auto MapPool(Ret(*mapo)(Args...)) -> decltype(MapPoolWrapper((std::function<Ret(Args...)>)mapo))
	{
		return MapPoolWrapper((std::function<Ret(Args...)>)mapo);
	}

	// For lambdas and functors
	template<typename T>
	auto MapPool(T mapo) -> decltype(MapPoolWrapper(lambda_cast(mapo)))
	{
		return MapPoolWrapper(lambda_cast(mapo));
	}






	namespace impl
	{
		template<typename T>
		class MapOverlapBase
		{
		public:

			void setOverlapMode(Overlap mode)
			{
				this->m_overlapPolicy = mode;
			}

			void setEdgeMode(Edge mode)
			{
				this->m_edge = mode;
			}

			void setPad(T pad)
			{
				this->m_pad = pad;
			}

			void setUpdateMode(UpdateMode mode)
			{
				this->m_updateMode = mode;
			}

		protected:
			Overlap m_overlapPolicy = skepu::Overlap::RowWise;
			Edge m_edge = Edge::Duplicate;
			UpdateMode m_updateMode = skepu::UpdateMode::Normal;
			T m_pad {};
		};


		template<typename Ret, typename... Args>
		class MapOverlap1D: public MapOverlapBase<typename region_type_ext<Args...>::type>, public SeqSkeletonBase
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr bool randomized = has_random<Args...>::value;
			static constexpr size_t randomCount = get_random_count<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;

			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};

			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, randomized, MapFunc>;
			using RegionType = typename pack_element<(indexed ? 1 : 0) + (randomized ? 1 : 0), Args...>::type;
			using T = typename region_type<RegionType>::type;
			static constexpr bool isPool = std::is_same<typename std::decay<RegionType>::type, Pool2D<T>>::value;

		public:

			void setOverlap(size_t o)
			{
				this->m_overlap = o;
			}

			size_t getOverlap() const
			{
				return this->m_overlap;
			}

			void setStride(size_t si)
			{
				this->m_strides = StrideList<1>(si);
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				const int overlap = (int)this->m_overlap;
				const size_t size = arg.size();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap1D: size = " << size);

				// Verify overlap radius is valid
				if (size < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");

				if (disjunction(get<OI>(std::forward<CallArgs>(args)...).size() < size...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction(get<EI>(std::forward<CallArgs>(args)...).size() != size...))
					SKEPU_ERROR("Non-matching input container sizes");

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

					for (size_t i = size - overlap; i < size; ++i)
					{
						if (p == Parity::None || index_parity(p, i))
						{
							auto res = F::forward(this->mapFunc, Index1D{i}, random_post, RegionType{overlap, 1, &end[i + 2 * overlap - size]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i)..., res);
						}
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
			}


			template<typename... CallArgs,
				REQUIRES(is_skepu_vector<typename std::remove_reference<typename pack_element<0, CallArgs...>::type>::type>::value)>
			auto operator()(CallArgs&&... args) -> decltype(get<0>(std::forward<CallArgs>(args)...))
			{
				if (this->m_updateMode == UpdateMode::Normal)
				{
					this->apply(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->apply(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->apply(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply_colwise(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap1D ColWise: size = " << size_i << " x " << size_j);

				// Verify overlap radius is valid
				if (this->m_edge != Edge::None && size_i < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");

				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				const int overlap = (int)this->m_overlap;
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

					if (this->m_edge != Edge::None)
					{
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

					for (size_t i = overlap; i < colWidth - overlap; ++i)
					{
						if (p == Parity::None || index_parity(p, i))
						{
							auto res = F::forward(this->mapFunc, Index1D{i}, random, RegionType{overlap, stride, &inputBegin[i*stride]},
								get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
							SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(i, col)..., res);
						}
					}

					inputBegin += 1;
				}
			}


			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply_rowwise(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap1D RowWise: size = " << size_i << " x " << size_j);

				// Verify overlap radius is valid
				if (this->m_edge != Edge::None && size_j < this->m_overlap * 2)
					SKEPU_ERROR("Non-matching overlap radius");

				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching input container sizes");

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				int overlap = (int)this->m_overlap;
				size_t size = arg.size();
				T start[3*overlap], end[3*overlap];

				size_t rowWidth = arg.total_cols();
				size_t stride = 1;

				const T *inputBegin = arg.getAddress();
				const T *inputEnd = inputBegin + size;

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

					if (this->m_edge != Edge::None)
					{
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

						for (size_t i = rowWidth - overlap; i < rowWidth; ++i)
						{
							if (p == Parity::None || index_parity(p, i))
							{
							 	auto res = F::forward(this->mapFunc, Index1D{i}, random_post, RegionType{overlap, stride, &end[i + 2 * overlap - rowWidth]}, get<AI>(std::forward<CallArgs>(args)...).hostProxy()..., get<CI>(std::forward<CallArgs>(args)...)...);
								SKEPU_VARIADIC_RETURN(get<OI>(std::forward<CallArgs>(args)...)(row, i)..., res);
							}
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
							this->apply_colwise(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
						{
							DEBUG_TEXT_LEVEL1("Red");
							this->apply_colwise(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
						{
							DEBUG_TEXT_LEVEL1("Black");
							this->apply_colwise(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						}
						break;

					case Overlap::RowWise:
						if (this->m_updateMode == UpdateMode::Normal)
						{
							this->apply_rowwise(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
						{
							DEBUG_TEXT_LEVEL1("Red");
							this->apply_rowwise(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						}
						if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
						{
							DEBUG_TEXT_LEVEL1("Black");
							this->apply_rowwise(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
						}
						break;

					default:
						SKEPU_ERROR("MapOverlap: Invalid overlap policy");
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}

//		protected:
			MapFunc mapFunc;
			StrideList<1> m_strides{};
			MapOverlap1D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::Duplicate;
			}

			size_t m_overlap = 1;

			friend MapOverlap1D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};




		template<typename Ret, typename... Args>
		class MapOverlap2D: public MapOverlapBase<typename region_type_ext<Args...>::type>, public SeqSkeletonBase
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr bool randomized = has_random<Args...>::value;
			static constexpr size_t randomCount = get_random_count<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;

			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};

			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, randomized, MapFunc>;
			using RegionType = typename pack_element<(indexed ? 1 : 0) + (randomized ? 1 : 0), Args...>::type;
			using T = typename region_type<RegionType>::type;
			static constexpr bool isPool = std::is_same<typename std::decay<RegionType>::type, Pool2D<T>>::value;

		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}

			void setOverlap(size_t o)
			{
				this->m_overlap[0] = o;
				this->m_overlap[1] = o;
			}

			void setOverlap(size_t i, size_t j)
			{
				this->m_overlap[0] = i;
				this->m_overlap[1] = j;
			}

			void setStride(size_t si, size_t sj)
			{
				this->m_strides = StrideList<2>(si, sj);
			}
			/*
			std::pair<size_t, size_t> getOverlap() const
			{
				return std::make_pair(this->m_overlap_x, this->m_overlap_y);
			}*/

			void setStride(size_t si, size_t sj, size_t sk)
			{
				this->m_strides = StrideList<3>(si, sj, sk);
			}

		private:

			template<size_t dim>
			size_t expectedInputSize(size_t size) const
			{
				if (isPool) return this->m_overlap[dim] + this->m_strides[dim] * (size - 1);
				else if (this->m_edge == Edge::None) return size + 2 * this->m_overlap[dim];
				else return size;
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap2D: size = " << size_i << " x" << size_j);

				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() != size_j) ...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != expectedInputSize<0>(size_i)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != expectedInputSize<1>(size_j)) ...))
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

				for (size_t i = 0; i < size_i; ++i)
					for (size_t j = 0; j < size_j; ++j)
						if (p == Parity::None || index_parity(p, i, j))
						{
							region.idx = Index2D{i * this->m_strides[0], j * this->m_strides[1]};
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
					this->apply(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->apply(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->apply(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}

//		protected:
			MapFunc mapFunc;
			StrideList<2> m_strides{};
			MapOverlap2D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::None;
			}

			int m_overlap[2] = {1,1};

			friend MapOverlap2D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};







		template<typename Ret, typename... Args>
		class MapOverlap3D: public MapOverlapBase<typename region_type_ext<Args...>::type>, public SeqSkeletonBase
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr bool randomized = has_random<Args...>::value;
			static constexpr size_t randomCount = get_random_count<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;

			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};

			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, randomized, MapFunc>;
			using RegionType = typename pack_element<(indexed ? 1 : 0) + (randomized ? 1 : 0), Args...>::type;
			using T = typename region_type<RegionType>::type;
			static constexpr bool isPool = std::is_same<typename std::decay<RegionType>::type, Pool3D<T>>::value;

		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}

			void setOverlap(int o)
			{
				this->m_overlap[0] = o;
				this->m_overlap[1] = o;
				this->m_overlap[2] = o;
			}

			void setOverlap(int oi, int oj, int ok)
			{
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
				this->m_overlap[2] = ok;
			}
			/*
			std::tuple<int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap_i, this->m_overlap_j, this->m_overlap_k);
			}*/

			void setStride(size_t si, size_t sj, size_t sk)
			{
				this->m_strides = StrideList<3>(si, sj, sk);
			}

		private:

			template<size_t dim>
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
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();

				DEBUG_TEXT_LEVEL1("Native C++ MapOverlap3D: size = " << size_i << " x " << size_j << " x" << size_k);

				if (disjunction(
					(get<OI>(std::forward<CallArgs>(args)...).size_i() != size_i) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_j() != size_j) &&
					(get<OI>(std::forward<CallArgs>(args)...).size_k() != size_k) ...))
					SKEPU_ERROR("Non-matching output container sizes");

				if (disjunction(
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != expectedInputSize<0>(size_i)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != expectedInputSize<1>(size_j)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_k() != expectedInputSize<2>(size_k)) ...))
					SKEPU_ERROR("Non-matching input container sizes");

				auto &arg = get<OutArity>(std::forward<CallArgs>(args)...);

				size_t offset[3];
				for (size_t d = 0; d < 3; ++d)
				{
					if (!isPool && this->m_edge != Edge::None)
					{
						offset[d] = 0;
					}
					else if (!isPool && this->m_edge == Edge::None)
					{
						offset[d] = this->m_overlap[d];
					}
					else if (isPool && this->m_edge != Edge::None)
					{
						offset[d] = -this->m_overlap[d] / 2;
					}
					else if (isPool && this->m_edge == Edge::None)
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
									i * this->m_strides[0],
									j * this->m_strides[1],
									k * this->m_strides[2]
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
					this->apply(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->apply(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->apply(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}

//		protected:
			MapFunc mapFunc;
			StrideList<3> m_strides{};
			MapOverlap3D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::None;
			}

			int m_overlap[3] = {1, 1, 1};

			friend MapOverlap3D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
		};





		template<typename Ret, typename... Args>
		class MapOverlap4D: public MapOverlapBase<typename region_type_ext<Args...>::type>, public SeqSkeletonBase
		{
			static constexpr bool indexed = is_indexed<Args...>::value;
			static constexpr bool randomized = has_random<Args...>::value;
			static constexpr size_t randomCount = get_random_count<Args...>::value;
			static constexpr size_t InArity = 1;
			static constexpr size_t OutArity = out_size<Ret>::value;
			static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0) - (randomized ? 1 : 0) + OutArity;
			static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;

			static constexpr typename make_pack_indices<OutArity, 0>::type out_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity, OutArity>::type elwise_indices{};
			static constexpr typename make_pack_indices<InArity + OutArity + anyCont, InArity + OutArity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, InArity + OutArity + anyCont>::type const_indices{};

			using MapFunc = std::function<Ret(Args...)>;
			using F = ConditionalIndexForwarder<indexed, randomized, MapFunc>;
			using RegionType = typename pack_element<(indexed ? 1 : 0) + (randomized ? 1 : 0), Args...>::type;
			using T = typename region_type<RegionType>::type;
			static constexpr bool isPool = std::is_same<typename std::decay<RegionType>::type, Pool4D<T>>::value;

		public:
			void setBackend(BackendSpec) {}
			void resetBackend() {}

			void setOverlap(int o)
			{
				this->m_overlap[0] = o;
				this->m_overlap[1] = o;
				this->m_overlap[2] = o;
				this->m_overlap[3] = o;
			}

			void setOverlap(int oi, int oj, int ok, int ol)
			{
				this->m_overlap[0] = oi;
				this->m_overlap[1] = oj;
				this->m_overlap[2] = ok;
				this->m_overlap[3] = ol;
			}
			/*
			std::tuple<int, int, int, int> getOverlap() const
			{
				return std::make_tuple(this->m_overlap_i, this->m_overlap_j, this->m_overlap_k, this->m_overlap_l);
			}*/


			void setStride(size_t si, size_t sj, size_t sk, size_t sl)
			{
				this->m_strides = StrideList<4>(si, sj, sk, sl);
			}

		private:

			template<size_t dim>
			size_t expectedInputSize(size_t size) const
			{
				if (isPool) return this->m_overlap[dim] + this->m_strides[dim] * (size - 1);
				else if (this->m_edge == Edge::None) return size + 2 * this->m_overlap[dim];
				else return size;
			}

			template<size_t... OI, size_t... EI, size_t... AI, size_t... CI, typename... CallArgs>
			void apply(Parity p, pack_indices<OI...>, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
			{
				size_t size_i = get<0>(std::forward<CallArgs>(args)...).size_i();
				size_t size_j = get<0>(std::forward<CallArgs>(args)...).size_j();
				size_t size_k = get<0>(std::forward<CallArgs>(args)...).size_k();
				size_t size_l = get<0>(std::forward<CallArgs>(args)...).size_l();

				const std::string Name = isPool ? "MapPool4D" : "MapOverlap4D";
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
					(get<EI>(std::forward<CallArgs>(args)...).size_i() != expectedInputSize<0>(size_i)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_j() != expectedInputSize<1>(size_j)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_k() != expectedInputSize<2>(size_k)) &&
					(get<EI>(std::forward<CallArgs>(args)...).size_l() != expectedInputSize<3>(size_l))...))
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
					this->apply(Parity::None, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Red)
				{
					DEBUG_TEXT_LEVEL1("Red");
					this->apply(Parity::Odd, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				if (this->m_updateMode == UpdateMode::RedBlack || this->m_updateMode == UpdateMode::Black)
				{
					DEBUG_TEXT_LEVEL1("Black");
					this->apply(Parity::Even, out_indices, elwise_indices, any_indices, const_indices, std::forward<CallArgs>(args)...);
				}
				return get<0>(std::forward<CallArgs>(args)...);
			}

//		protected:
			MapFunc mapFunc;
			StrideList<4> m_strides{};
			MapOverlap4D(MapFunc map): mapFunc(map)
			{
				this->m_edge = Edge::None;
			}

			int m_overlap[4] = {1, 1, 1, 1};

			friend MapOverlap4D<Ret, Args...> skepu::MapOverlapWrapper<Ret, Args...>(MapFunc);
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




		template<typename Ret, typename... Args>
		class MapPool3D: public MapOverlap3D<Ret, Args...>
		{
			using MapFunc = std::function<Ret(Args...)>;

		public:
			void setOverlap(size_t, size_t, size_t) = delete;
		//	void setEdgeMode(Edge) = delete;
			void setUpdateMode(UpdateMode) = delete;

			void setPoolSize(size_t pi, size_t pj, size_t pk)
			{
				this->m_overlap[0] = pi;
				this->m_overlap[1] = pj;
				this->m_overlap[2] = pk;
			}

			MapPool3D(MapFunc map): MapOverlap3D<Ret, Args...>(map) {}
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

	}
}
