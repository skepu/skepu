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

	// For function pointers
	template<typename T>
	impl::Reduce1D<T> Reduce(T(*red)(T, T))
	{
		return ReduceWrapper((std::function<T(T, T)>)red);
	}

	// For lambdas and functors
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

			void setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
			}

			void setStartValue(T val)
			{
				this->m_start = val;
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
