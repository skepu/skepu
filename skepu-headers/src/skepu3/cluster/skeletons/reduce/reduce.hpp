#pragma once
#ifndef SKEPU_STARPU_SKELETON_REDUCE_HPP
#define SKEPU_STARPU_SKELETON_REDUCE_HPP 1

#include <set>
#include <vector>

#include <omp.h>
#include <starpu_mpi.h>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>
#include <skepu3/cluster/containers/vector/vector.hpp>
#include "../skeleton_base.hpp"
#include "../skeleton_task.hpp"
#include "reduce_mode.hpp"
#include "reduce_cuda_helpers.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename ReduceFunc, typename CUKernel>
struct reduce1d
{
	typedef typename ReduceFunc::Ret T;

	#ifdef SKEPU_CUDA
	CUKernel static cu_kernel;
	#endif

	template<
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename Buffers>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t const count) noexcept
	-> void
	{
		#pragma omp declare \
			reduction(reducer:T:omp_out=ReduceFunc::OMP(omp_out,omp_in))
		auto & res = std::get<0>(buffers)[0];
		auto container = std::get<2>(buffers);

		res = container[0];
		#pragma omp parallel for \
			reduction(reducer:res) \
			num_threads(starpu_combined_worker_get_size())
		for(size_t i = 1; i < count; ++i)
			res = ReduceFunc::OMP(res,container[i]);
	}

#ifdef SKEPU_CUDA
	template<
		size_t... RI,
		size_t... EI,
		size_t... CI,
		typename Buffers>
	auto static
	CUDA(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t const count) noexcept
	-> void
	{
		size_t threads;
		size_t grid;
		size_t mem_size;
		cudaStream_t stream = starpu_cuda_get_local_stream();
		std::tie(threads, grid, mem_size) = util::generate_cuda_params<T>(count);

		auto res = std::get<0>(buffers);
		auto container = std::get<2>(buffers);
		if(grid > 1)
		{
			auto buffer = std::get<1>(buffers);
			cu_kernel<<<grid, threads, mem_size, stream>>>(
				container,
				buffer,
				count,
				threads,
				util::is_pow2(count));
			cu_kernel<<<1, grid /2, mem_size, stream>>>(
				buffer,
				res,
				grid,
				grid /2,
				util::is_pow2(grid));
		}
		else
			cu_kernel<<<1, threads, mem_size, stream>>>(
				container,
				res,
				count,
				threads,
				util::is_pow2(count));

		cudaStreamSynchronize(stream);
	}
#endif

	template<
		typename Buffers,
		size_t... RI,
		size_t... EI,
		size_t... CI>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t const count,
		size_t const width) noexcept
	-> void
	{
		auto res = std::get<0>(buffers);
		auto matrix = std::get<2>(buffers);

		#pragma omp parallel for \
			num_threads(starpu_combined_worker_get_size())
		for(size_t i = 0; i < count; ++i)
		{
			auto row = matrix + (i*width);
			T row_res = row[0];
			for(size_t j(1); j < width; ++j)
				row_res = ReduceFunc::OMP(row_res, row[j]);
			res[i] = row_res;
		}
	}

	#ifdef SKEPU_CUDA
	template<
		typename Buffers,
		size_t... RI,
		size_t... EI,
		size_t... CI>
	auto static
	CUDA(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		size_t const rows,
		size_t const width) noexcept
	-> void
	{
		size_t threads;
		size_t grid;
		size_t mem_size;
		cudaStream_t stream = starpu_cuda_get_local_stream();
		std::tie(threads, grid, mem_size) = util::generate_cuda_params<T>(width);
		bool width_pow2 = util::is_pow2(width);
		bool grid_pow2 = util::is_pow2(grid);

		auto res = std::get<0>(buffers);
		auto matrix = std::get<2>(buffers);
		if(grid > 1)
		{
			auto buffer = std::get<1>(buffers);
			for(size_t i = 0; i < rows; ++i)
			{
				cu_kernel<<<grid, threads, mem_size, stream>>>(
					matrix + (i*width),
					buffer,
					width,
					threads,
					width_pow2);
				cu_kernel<<<1, grid /2, mem_size, stream>>>(
					buffer,
					res +i,
					grid,
					grid /2,
					grid_pow2);
			}
		}
		else
			for(size_t i = 0; i < rows; ++i)
			{
				cu_kernel<<<1, threads, mem_size, stream>>>(
					matrix + (i*width),
					res +i,
					width,
					threads,
					width_pow2);
			}

		cudaStreamSynchronize(stream);
	}
	#endif // SKEPU_CUDA
};

#ifdef SKEPU_CUDA
template<typename ReduceFunc, typename CUKernel>
CUKernel reduce1d<ReduceFunc, CUKernel>::cu_kernel = 0;
#endif

} // namespace _starpu

template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
class Reduce1D
: public cluster::skeleton_task<
		_starpu::reduce1d<ReduceFunc, CUDAKernel>,
		std::tuple<typename ReduceFunc::Ret, typename ReduceFunc::Ret>,
		std::tuple<typename ReduceFunc::Ret>,
		std::tuple<>,
		std::tuple<>>
{
public:
	typedef typename ReduceFunc::Ret T;
	typedef cluster::skeleton_task<
			_starpu::reduce1d<ReduceFunc, CUDAKernel>,
			std::tuple<T, T>,
			std::tuple<T>,
			std::tuple<>,
			std::tuple<>>
		skeleton_task;

	auto static constexpr uniform_indices =
		make_pack_indices<0>::type{};

protected:
	ReduceMode m_mode = ReduceMode::RowWise;

protected:
	T m_start{};

	T m_result;
	std::vector<starpu_data_handle_t> m_result_handles;

	T m_buffer[util::max_threads *2];
	std::vector<starpu_data_handle_t> m_buffer_handles;

public:
	//static constexpr auto skeletonType = SkeletonType::Reduce1D;
	typedef std::tuple<T> ResultArg;
	typedef std::tuple<T> ElwiseArgs;
	typedef std::tuple<> ContainerArgs;

	static constexpr bool prefers_matrix = false;

	#ifndef SKEPU_CUDA
	Reduce1D(CUDAKernel) noexcept
	#else
	Reduce1D(CUDAKernel cu_kernel) noexcept
	#endif
	: skeleton_task("Reduce1D_ElWise"),
		m_start(T()),
		m_result(0),
		m_result_handles(skepu::cluster::mpi_size(), 0),
		m_buffer_handles(skepu::cluster::mpi_size(), 0)
	{
		#ifdef SKEPU_CUDA
		_starpu::reduce1d<ReduceFunc, CUDAKernel>::cu_kernel = cu_kernel;
		#endif

		auto rank = skepu::cluster::mpi_rank();
		for(size_t i(0); i < skepu::cluster::mpi_size(); ++i)
		{
			int home_node = -1;
			T * result_ptr(0);
			T * buffer_ptr(0);
			if(i == rank)
			{
				home_node = STARPU_MAIN_RAM;
				result_ptr = &m_result;
				buffer_ptr = m_buffer;
			}
			starpu_variable_data_register(
				&m_result_handles[i],
				home_node,
				(uintptr_t)result_ptr,
				sizeof(T));
			starpu_mpi_data_register(
				m_result_handles[i],
				skepu::cluster::mpi_tag(),
				i);
			starpu_vector_data_register(
				&m_buffer_handles[i],
				home_node,
				(uintptr_t)buffer_ptr,
				(uint32_t)util::max_threads *2,
				sizeof(T));
			starpu_mpi_data_register(
				m_buffer_handles[i],
				cluster::mpi_tag(),
				i);
		}
	}

	~Reduce1D() noexcept
	{
		for(auto & handle : m_result_handles)
			starpu_data_unregister_no_coherency(handle);
		for(auto & handle : m_buffer_handles)
			starpu_data_unregister_no_coherency(handle);

		skepu::cluster::barrier();
	}

	auto
	setReduceMode(ReduceMode mode) noexcept
	-> void
	{
		this->m_mode = mode;
	}

	auto
	setStartValue(T val) noexcept
	-> void
	{
		this->m_start = val;
	}

	template<typename... Args>
	auto tune(Args&&...) noexcept -> void {}

	template<template<class> class Container,
		REQUIRES_VALUE(is_skepu_container<Container<T>>)>
	auto
	operator()(Container<T> & arg) noexcept
	-> T
	{
		return backendDispatch(arg.begin(), arg.end());
	}

	template<typename Iterator,
		REQUIRES_VALUE(is_skepu_iterator<Iterator,T>)>
	auto
	operator()(Iterator begin, Iterator end) noexcept
	-> T
	{
		return backendDispatch(begin, end);
	}

	template<
		template<typename>class Vector,
		template<typename>class Matrix,
		REQUIRES_VALUE(is_skepu_vector<Vector<T>>),
		REQUIRES_VALUE(is_skepu_matrix<Matrix<T>>)>
	auto
	operator()(Vector<T> & res, Matrix<T> & arg) noexcept
	-> Vector<T> &
	{
		backendDispatch(res, arg);
		return res;
	}

protected:
	template<typename Iterator>
	auto
	backendDispatch(Iterator begin, Iterator end) noexcept
	-> T
	{
		return STARPU(begin, end);
	}

	template<typename Vector, typename Matrix>
	auto
	backendDispatch(Vector & res, Matrix & arg)
	-> void
	{
		switch(m_mode)
		{
		case ReduceMode::ColWise:
		{
			auto argt = Matrix(arg.size_j(), arg.size_i());
			cont::getParent(arg).transpose_to(cont::getParent(argt));
			STARPU(res, argt);
			break;
		}
		default:
			STARPU(res, arg);
		}
	}

private:
	template<typename Iterator>
	auto
	STARPU(Iterator begin, Iterator const end) noexcept
	-> T
	{
		begin.getParent().filter(
			begin.getParent().min_filter_parts());
		begin.getParent().partition();
		std::set<int> scheduled_ranks;

		while(begin != end)
		{
			auto pos = begin.offset();
			auto count = begin.getParent().block_count_from(pos);
			auto handle = begin.getParent().handle_for(pos);
			auto rank = starpu_mpi_data_get_rank(handle);
			scheduled_ranks.insert(rank);
			auto handles =
				std::make_tuple(
					m_result_handles[rank],
					m_buffer_handles[rank],
					handle);

			skeleton_task::schedule(
				handles,
				count);

			begin += count;
		}

		auto result = m_start;
		for(auto & i : scheduled_ranks)
		{
			auto & handle = m_result_handles[i];
			starpu_mpi_get_data_on_all_nodes_detached(MPI_COMM_WORLD, handle);
			starpu_data_acquire(handle, STARPU_RW);
			auto ptr = (T *)starpu_variable_get_local_ptr(handle);
			result = ReduceFunc::CPU(result, *ptr);
			starpu_data_release(handle);
			starpu_mpi_cache_flush(MPI_COMM_WORLD, handle);
		}

		return result;
	}

	template<typename Vector, typename Matrix>
	auto
	STARPU(Vector & res, Matrix & data)
	-> Vector
	{
		auto & rp = cont::getParent(res);
		auto & dp = cont::getParent(data);
		auto cols = data.size_j();

		auto filter_parts =
			std::max<size_t>(
				rp.min_filter_parts(),
				dp.min_filter_parts());
		rp.filter(filter_parts);
		dp.filter(filter_parts);

		rp.partition();
		rp.invalidate_local_storage();
		dp.partition();

		for(size_t i(0); i < res.size();)
		{
			auto count = rp.block_count_from(i);
			auto rp_handle = rp.handle_for(i);
			auto rank = starpu_mpi_data_get_rank(rp_handle);
			auto handles = std::make_tuple(
				rp.handle_for(i),
				m_buffer_handles[rank],
				dp.handle_for(i * cols));

			skeleton_task::schedule(
				handles,
				count,
				cols);

			i += count;
		}

		return res;
	}
};

template<
	typename ReduceFuncRowWise,
	typename ReduceFuncColWise,
	typename CUDARowWise,
	typename CUDAColWise,
	typename CLKernel>
class Reduce2D
: public Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>
{
	typedef Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel> row_reduce;
	typedef Reduce1D<ReduceFuncColWise, CUDAColWise, CLKernel> ColReduce;
	typedef typename ReduceFuncRowWise::Ret T;

	ColReduce col_reduce;

public:
	// static constexpr auto skeletonType = SkeletonType::Reduce2D;
	static constexpr bool prefers_matrix = true;

	Reduce2D(CUDARowWise row_kernel, CUDAColWise col_kernel)
	: row_reduce(row_kernel), col_reduce(col_kernel)
	{}

	~Reduce2D()
	{
		skepu::cluster::barrier();
	}

	template<template<typename>class Container,
		REQUIRES(
			is_skepu_container<Container<T>>::value
			&& !is_skepu_matrix<Container<T>>::value)>
	auto
	operator()(Container<T> & arg)
	-> T
	{
		return row_reduce::operator()(arg);
	}

	template<
		template<typename>class Vector,
		template<typename>class Matrix,
		REQUIRES_VALUE(is_skepu_vector<Vector<T>>),
		REQUIRES_VALUE(is_skepu_matrix<Matrix<T>>)>
	auto
	operator()(Vector<T> & v, Matrix<T> & m)
	-> Vector<T> &
	{
		return row_reduce::operator()(v, m);
	}

	template<
		template<typename>class Matrix,
		REQUIRES_VALUE(is_skepu_matrix<Matrix<T>>)>
	auto
	operator()(Matrix<T> & arg)
	-> T
	{
		auto size =
			row_reduce::m_mode == ReduceMode::RowWise
				? arg.size_i()
				: arg.size_j();
		Vector<T> v(size);
		row_reduce::operator()(v, arg);
		return col_reduce(v);
	}

	auto
	setBackend(BackendSpec spec) noexcept
	-> void
	{
		col_reduce.setBackend(spec);
		row_reduce::setBackend(spec);
	}

	auto
	resetBackend() noexcept
	-> void
	{
		col_reduce.resetBackend();
		row_reduce::resetBackend();
	}
};

} // namespace backend
} // namespace skepu

#endif // SKEPU_STARPU_SKELETON_REDUCE_HPP
