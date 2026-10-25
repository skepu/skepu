#pragma once

#include "skepu3/impl/common.hpp"

namespace skepu
{

	namespace impl
	{
		template<typename>
		class Reduce1D;

		template<typename>
		class Reduce2D;
	}


	template<typename T>
	impl::Reduce1D<T> ReduceWrapper(std::function<T(T, T)> red)
	{
		return impl::Reduce1D<T>(red);
	}

	// Reduce1D utilizing function pointer
	template<typename T>
	impl::Reduce1D<T> Reduce(T(*red)(T, T))
	{
		return ReduceWrapper((std::function<T(T, T)>)red);
	}

	// Reduce1D utilizing lambda
	template<typename T>
	auto Reduce(T red) -> decltype(ReduceWrapper(lambda_cast(red)))
	{
		return ReduceWrapper(lambda_cast(red));
	}



	template<typename T>
	impl::Reduce2D<T> ReduceWrapper(std::function<T(T, T)> rowRed, std::function<T(T, T)> colRed)
	{
		return impl::Reduce2D<T>(rowRed, colRed);
	}

	// For function pointers
	template<typename T>
	impl::Reduce2D<T> Reduce(T(*rowRed)(T, T), T(*colRed)(T, T))
	{
		return ReduceWrapper((std::function<T(T, T)>)rowRed, (std::function<T(T, T)>)colRed);
	}

	// For lambdas and functors
	template<typename T1, typename T2>
	auto Reduce(T1 rowRed, T2 colRed) -> decltype(ReduceWrapper(lambda_cast(rowRed), lambda_cast(colRed)))
	{
		return ReduceWrapper(lambda_cast(rowRed), lambda_cast(colRed));
	}


	namespace impl
	{
		// Reduce1D for Vectors or Matrices
		template<typename T>
		class Reduce1D: public SeqSkeletonBase
		{
			using RedFunc = std::function<T(T, T)>;

		public:
			SKEPU_CHAIN_SKELETON_SETTERS(SeqSkeletonBase)

			auto &setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
				return *this;
			}

			auto &setStartValue(T val)
			{
				this->m_start = val;
				return *this;
			}


	/*		template<template<class> class Container>
			typename std::enable_if<is_skepu_container<Container<T>>::value, T>::type
			operator()(Container<T>& arg)
			{
				size_t size = arg.size();

				T res = this->m_start;

				res = *arg.begin();
				for (size_t i = 1; i < size; i++)
					res = this->redFunc(res, *(arg.begin() + i));

				return res;
			}*/

			template<size_t... OI, typename... CallArgs>
			T
			apply(pack_indices<OI...>, CallArgs&&... args)
			{
				SKEPU_TRACE_START_EVENT(trace_handle);
				size_t size = get<0>(std::forward<CallArgs>(args)...).size();

				T res = this->m_start;
				for (size_t i = 0; i < size; i++)
				//	res = redFunc(res, std::forward_as_tuple(*(args.begin() + i)...));
					pack_expand((get<OI>(res) = redFunc(get<OI>(res), get_or_return<OI>(*(args.begin() + i))), 0)...);

				return res;
			}

			template<size_t... OI, typename... CallArgs>
			Scalar<T>
			operator()(CallArgs&&... args)
			{
				static constexpr typename make_pack_indices<sizeof...(CallArgs), 0>::type out_indices{};
				T value = this->apply(out_indices, std::forward<CallArgs>(args)...);
				return Scalar<T>(value);
			}

			// The entries of each row of a sparse matrix are reduced, to an element of the output Vector
			Vector<T> &operator()(Vector<T> &res, SparseMatrix<T> &arg)
			{
				if (this->m_mode != ReduceMode::RowWise)
					SKEPU_ERROR("Reduce of a sparse matrix: only the reduction of the rows is supported (use the transpose for columns)");
				if (!arg.has_pattern())
					SKEPU_ERROR("Reduce: the sparse matrix has no pattern");
				const size_t rows = arg.rows();
				if (res.size() != rows)
					SKEPU_ERROR("Reduce: the output Vector has " << res.size() << " elements, and the sparse matrix " << rows << " rows");

				arg.updateHost();
				res.invalidateDeviceData();
				const SparseIndex *offsets = arg.pattern()->row_offsets().data();
				const T *values = arg.values().getAddress();
				for (size_t r = 0; r < rows; ++r)
				{
					T inner = this->m_start;
					for (SparseIndex k = offsets[r]; k < offsets[r + 1]; ++k)
						inner = this->redFunc(inner, values[k]);
					res.getAddress()[r] = inner;
				}
				return res;
			}

			Vector<T> &operator()(Vector<T> &res, Matrix<T>& arg)
			{
				size_t rows = arg.total_rows();
				size_t cols = arg.total_cols();

				if (this->m_mode == ReduceMode::RowWise)
				{
					if (res.size() != rows)
						SKEPU_ERROR("Reduce: Non-matching container sizes");

					for (size_t r = 0; r < rows; r++)
					{
						T inner = this->m_start;
						for (size_t c = 0; c < cols; c++)
							inner = this->redFunc(inner, arg(r, c));
						res(r) = inner;
					}
				}
				else if (this->m_mode == ReduceMode::ColWise)
				{
					if (res.size() != cols)
						SKEPU_ERROR("Reduce: Non-matching container sizes");

					for (size_t c = 0; c < cols; c++)
					{
						T inner = this->m_start;
						for (size_t r = 0; r < rows; r++)
							inner = this->redFunc(inner, arg(r, c));
						res(c) = inner;
					}
				}

				return res;
			}

//		protected:
			RedFunc redFunc;
			Reduce1D(RedFunc red): redFunc(red) {}

			ReduceMode m_mode = ReduceMode::RowWise;
			T m_start{};

			friend Reduce1D<T> ReduceWrapper<T>(RedFunc);
		};


		// Reduce 2D only for Matrix containers
		// Should be some way to determine order?
		template<typename T>
		class Reduce2D: public Reduce1D<T>
		{
			using RedFunc = std::function<T(T, T)>;

		public:
			using SkepuChainBase = Reduce1D<T>;
			SKEPU_CHAIN_SKELETON_SETTERS(SkepuChainBase)
			SKEPU_CHAIN(setReduceMode, SkepuChainBase)
			SKEPU_CHAIN(setStartValue, SkepuChainBase)

			T operator()(Vector<T>& arg)
			{
				return Reduce1D<T>::operator()(arg);
			}

			T operator()(Matrix<T>& arg)
			{
				size_t rows = arg.total_rows();
				size_t cols = arg.total_cols();

				T res = this->m_start;

				if (this->m_mode == ReduceMode::RowWise)
				{
					for (size_t r = 0; r < rows; r++)
					{
						T inner = this->m_start;
						for (size_t c = 0; c < cols; c++)
							inner = this->redFunc(inner, arg(r, c));
						res = this->colRedFunc(res, inner);
					}
				}
				else
				{
					for (size_t c = 0; c < cols; c++)
					{
						T inner = this->m_start;
						for (size_t r = 0; r < rows; r++)
							inner = this->redFunc(inner, arg(r, c));
						res = this->colRedFunc(res, inner);
					}
				}

				return res;
			}

		private:
			RedFunc colRedFunc;
			Reduce2D(RedFunc rowRed, RedFunc colRed) : Reduce1D<T>(rowRed), colRedFunc(colRed) {}

			T m_start{};

			friend Reduce2D<T> ReduceWrapper<T>(RedFunc, RedFunc);
		};
	}

}
