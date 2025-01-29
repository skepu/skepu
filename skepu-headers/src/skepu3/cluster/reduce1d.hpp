#ifndef REDUCE1D_HPP
#define REDUCE1D_HPP

#include <starpu.h>
#include <skepu3/impl/meta_helpers.hpp>
#include <skepu3/cluster/index.hpp>
#include <skepu3/cluster/cluster_meta_helpers.hpp>
#include <skepu3/cluster/matrix.hpp>
#include <skepu3/cluster/vector.hpp>
#include <skepu3/cluster/matrix_iterator.hpp>
#include <skepu3/cluster/multivariant_task.hpp>
#include <skepu3/cluster/reduce_mode.hpp>
#include <cstddef>

#include <tuple>


namespace skepu
{
	namespace backend
	{
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		class Reduce1D :
			cluster::multivariant_task<std::tuple<typename ReduceFunc::Ret>,
			                           std::tuple<typename ReduceFunc::Ret>,
			                           std::tuple<>,
			                           std::tuple<>,
			                           Reduce1D<ReduceFunc,
			                                    CUDAKernel,
			                                    CLKernel> >
		{
			using Self = Reduce1D<ReduceFunc, CUDAKernel, CLKernel>;
			using T = typename ReduceFunc::Ret;
			using F = ConditionalIndexForwarder<ReduceFunc::indexed,
			                                    decltype(&ReduceFunc::OMP)>;
			CUDAKernel m_cuda_kernel;
			ReduceMode m_mode {};
			T m_start {};

		public:
			Reduce1D(CUDAKernel kernel);
			void setReduceMode(ReduceMode mode);
			void setStartValue(T val);

			template<template<class> class Container>
			T operator()(Container<T> &arg);

			template<typename Iterator>
			T operator()(Iterator arg, Iterator arg_end);

			template<typename Iterator>
			T operator()(Iterator arg);

			Vector<T> &operator()(Vector<T> &res, Matrix<T>& arg);

			template<typename MatT,
			         size_t... RI,
			         size_t... EI,
			         size_t... CI,
			         typename... Uniform>
			static void cpu(const void * self,
			                Size2D size,
			                Offset2D global_offset,
			                MatT && bufs,
			                pack_indices<RI...>,
			                pack_indices<EI...>,
			                pack_indices<CI...>,
			                Uniform... args);
		private:
			template<typename Iterator>
			T backendDispatch(Size2D size, Iterator arg);

			template<typename Iterator>
			void backendDispatch(Size2D size, Iterator res, Iterator arg);
		};

	}
}

#include "skepu3/cluster/impl/reduce1d.inl"
#endif
