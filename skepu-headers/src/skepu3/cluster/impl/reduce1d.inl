#ifndef REDUCE1D_INL
#define REDUCE1D_INL


#include "skepu3/cluster/reduce1d.hpp"
#include <omp.h>

namespace skepu
{
	namespace backend
	{
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::Reduce1D(CUDAKernel kernel) :
			m_cuda_kernel(kernel),
			m_start {},
			m_mode { ReduceMode::RowWise }
		{}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		void Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::setReduceMode(ReduceMode mode)
		{
			this->m_mode = mode;
		}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		void Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::setStartValue(typename ReduceFunc::Ret val)
		{
			this->m_start = val;
		}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<template<class> class Container>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::operator()(Container<typename ReduceFunc::Ret> &arg) {
			return this->backendDispatch(arg.size2D(),arg.begin());
		}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::operator()(Iterator arg) {
			return this->backendDispatch(arg.size2D(),arg);
		}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &
		Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::operator()(Vector<typename ReduceFunc::Ret> &res,
		             Matrix<typename ReduceFunc::Ret>& arg)
		{
			this->backendDispatch(arg.size2D(), res.begin(), arg.begin());
			return res;
		}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::backendDispatch(Size2D size, Iterator arg)
		{
			auto tmp_mode = m_mode;
			this->m_mode = skepu::ReduceMode::ElWise;
			std::vector<skepu::cluster::starpu_var<T>> partials;
			this->element_aligned_res_per_block(size, partials, arg);

			for (auto & val : partials) {
				val.broadcast();
			}
			auto res = m_start;
			for (auto & val : partials) {
				res = ReduceFunc::CPU(res, val.get());
			}
			m_mode = tmp_mode;
			return res;
		}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		void Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::backendDispatch(Size2D size, Iterator res, Iterator arg)
		{
			this->element_aligned_sweep(size, m_mode, res, arg);
		}

		template<typename T, typename FN>
		auto inline
		reducer(T & a, T & b, FN fn)
		-> T
		{
			return fn(a,b);
		}

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename MatT,
		         size_t... RI,
		         size_t... EI,
		         size_t... CI,
		         typename... Uniform>
		void Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::cpu(const void * self,
		      Size2D size,
		      Offset2D global_offset,
		      MatT && bufs,
		      pack_indices<RI...>,
		      pack_indices<EI...>,
		      pack_indices<CI...>,
		      Uniform... args)
		{
			auto mode = ((const Self*)self)->m_mode;
			auto start = ((const Self*)self)->m_start;

			auto & res = std::get<0>(bufs);
			auto & data = std::get<1>(bufs);

			omp_set_num_threads(starpu_combined_worker_get_size());

#pragma omp declare reduction(UFOMPReducer:T:omp_out=reducer(omp_out,omp_in,ReduceFunc::OMP))

			if (mode == skepu::ReduceMode::RowWise) {
				assert(res.cols >= size.row);

				// First initialize the result vector to start value if needed
				if (global_offset.col == 0) {
#pragma omp parallel for
					for (size_t row = 0; row < size.row; ++row) {
						res[row] = start;
					}
				}

				for (size_t row = 0; row < size.row; ++row) {
					auto row_res = res[row];
#pragma omp parallel for reduction(UFOMPReducer: row_res)
					for (size_t col = 0; col < size.col; ++col)
						row_res = ReduceFunc::OMP(row_res, data(row,col));
					res[row] = row_res;
				}
			}

			if (mode == skepu::ReduceMode::ColWise) {
				assert(res.cols >= size.col);

				// First initialize the result vector to start value if needed
				if (global_offset.row == 0) {
#pragma omp parallel for
					for (size_t col = 0; col < size.col; ++col) {
						res[col] = start;
					}
				}

				for (size_t row = 0; row < size.row; ++row) {
#pragma omp parallel for
					for (size_t col = 0; col < size.col; ++col)
						res[col] = ReduceFunc::OMP(res[col], data(row,col));
				}
			}

			if (mode == skepu::ReduceMode::ElWise) {
				assert(res.cols == res.rows == 1);
				auto r = data[0];
#pragma omp parallel for reduction(UFOMPReducer: r)
				for (size_t col = 1; col < size.col; ++col)
					r = ReduceFunc::OMP(r, data(0,col));

				for (size_t row = 1; row < size.row; ++row) {
#pragma omp parallel for reduction(UFOMPReducer: r)
					for (size_t col = 0; col < size.col; ++col)
						r = ReduceFunc::OMP(r, data(row,col));
				}
				res[0] = r;
			}



			return;
		}
	}
}


#endif /* REDUCE1D_INL */
