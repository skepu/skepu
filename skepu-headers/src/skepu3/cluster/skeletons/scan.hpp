#pragma once
#ifndef SKEPU_CLUSTER_SKELETON_SCAN_HPP
#define SKEPU_CLUSTER_SKELETON_SCAN_HPP 1

#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include <starpu_mpi.h>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>
#include <skepu3/cluster/skeletons/skeleton_task.hpp>

namespace skepu {

enum class ScanMode
{
	Inclusive,
	Exclusive,
}; // enum class ScanMode

namespace backend {
namespace _starpu {

template<typename UserFunc>
struct scan_first
{
	typedef typename UserFunc::Ret T;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename ... Args>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		std::tuple<T *, T *, T *> & buffers,
		size_t count,
		ScanMode const mode,
		T & initial,
		Args && ...) noexcept
	-> void
	{
		auto res = std::get<0>(buffers);
		auto & part_res = *std::get<1>(buffers);
		auto arg = std::get<2>(buffers);

		if(mode == ScanMode::Exclusive)
		{
			*res++ = initial;
			--count;
		}

		size_t const threads(
			std::max<size_t>(
				std::min<size_t>(starpu_combined_worker_get_size(), count / 2),
				1));
		size_t const part_count = count / threads;
		size_t const rest = count - (threads * part_count);

		// Array to store partial thread results in.
		std::vector<T> offset_array(threads);

		#pragma omp parallel num_threads(threads)
		{
			const size_t tid = omp_get_thread_num();
			const size_t first = tid * part_count;
			const size_t last =
				first + part_count
				+ (tid == threads - 1 ? rest : 0);

			if(tid)
				res[first] = arg[first];
			else
				res[first] =
					mode == ScanMode::Inclusive
					? arg[first]
					: UserFunc::OMP(initial, arg[first]);
			for(size_t i = first + 1; i < last; ++i)
				res[i] = UserFunc::OMP(res[i-1], arg[i]);
			offset_array[tid] = res[last-1];
			#pragma omp barrier

			// Let the master thread scan the partial result array
			#pragma omp master
			for(size_t i = 1; i < threads; ++i)
				offset_array[i] = UserFunc::OMP(offset_array[i-1], offset_array[i]);
			#pragma omp barrier

			if(tid != 0)
			{
				// Add the scanned partial results to each threads work batch.
				for(size_t i = first; i < last; ++i)
					res[i] = UserFunc::OMP(res[i], offset_array[tid-1]);
			}
		}
		if(mode == ScanMode::Inclusive)
			part_res = res[count -1];
		else
			part_res = UserFunc::OMP(res[count -1], arg[count]);
	}
}; // struct scan_first

template<typename UserFunc>
struct scan
{
	typedef typename UserFunc::Ret T;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename ... Args>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		std::tuple<T *, T *, T *> & buffers,
		size_t count,
		ScanMode const mode,
		Args && ...) noexcept
	-> void
	{
		auto res = std::get<0>(buffers);
		auto & part_res = *std::get<1>(buffers);
		auto arg = std::get<2>(buffers);

		if(mode == ScanMode::Exclusive)
		{
			*res++ = T();
			--count;
		}

		size_t const threads(
			std::max<size_t>(
				std::min<size_t>(starpu_combined_worker_get_size(), count / 2),
				1));
		size_t const part_count = count / threads;
		size_t const rest = count - (threads * part_count);

		// Array to store partial thread results in.
		std::vector<T> offset_array(threads);

		#pragma omp parallel num_threads(threads)
		{
			const size_t tid = omp_get_thread_num();
			const size_t first = tid * part_count;
			const size_t last =
				first + part_count
				+ (tid == threads - 1 ? rest : 0);

			// First let each thread make their own scan and saved the result in a
			// partial result array.
			res[first] = arg[first];
			for(size_t i = first + 1; i < last; ++i)
				res[i] = UserFunc::OMP(res[i-1], arg[i]);
			offset_array[tid] = res[last-1];
			#pragma omp barrier

			// Let the master thread scan the partial result array
			#pragma omp master
			for(size_t i = 1; i < threads; ++i)
				offset_array[i] = UserFunc::OMP(offset_array[i-1], offset_array[i]);
			#pragma omp barrier

			if(tid != 0)
			{
				// Add the scanned partial results to each threads work batch.
				for(size_t i = first; i < last; ++i)
					res[i] = UserFunc::OMP(res[i], offset_array[tid-1]);
			}
		}
		if(mode == ScanMode::Inclusive)
			part_res = res[count -1];
		else
			part_res = UserFunc::OMP(res[count -1], arg[count]);
	}
}; // struct scan

template<typename UserFunc>
struct scan_update
{
	typedef typename UserFunc::Ret T;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename ... Args>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		std::tuple<T *> & buffer,
		size_t count,
		Args && ...) noexcept
	-> void
	{
		auto partials = std::get<0>(buffer);
		for(size_t i(1); i < count; ++i)
			partials[i] = UserFunc::OMP(partials[i -1], partials[i]);
	}
}; // struct scan_update

template<typename UserFunc>
struct scan_add
{
	typedef typename UserFunc::Ret T;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename ... Args>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		std::tuple<T *, T*> & buffers,
		size_t count,
		Args && ...) noexcept
	-> void
	{
		auto res = std::get<0>(buffers);
		auto & part_res = *std::get<1>(buffers);
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t i = 0; i < count; ++i)
			res[i] = UserFunc::OMP(res[i], part_res);
	}
}; // struct scan_add

} // namespace _starpu

template<
	typename UserFunc,
	typename CUScan,
	typename CUUpdate,
	typename CUAdd,
	typename CLKernel>
class Scan
: public virtual SkeletonBase,
	private cluster::skeleton_task<
			_starpu::scan_first<UserFunc>,
			std::tuple<typename UserFunc::Ret, typename UserFunc::Ret>,
			std::tuple<typename UserFunc::Ret>,
			std::tuple<>,
			std::tuple<>>,
	private cluster::skeleton_task<
			_starpu::scan<UserFunc>,
			std::tuple<typename UserFunc::Ret, typename UserFunc::Ret>,
			std::tuple<typename UserFunc::Ret>,
			std::tuple<>,
			std::tuple<>>,
	private cluster::skeleton_task<
			_starpu::scan_update<UserFunc>,
			std::tuple<typename UserFunc::Ret>,
			std::tuple<>,
			std::tuple<>,
			std::tuple<>>,
	private cluster::skeleton_task<
			_starpu::scan_add<UserFunc>,
			std::tuple<typename UserFunc::Ret>,
			std::tuple<typename UserFunc::Ret>,
			std::tuple<>,
			std::tuple<>>
{
	typedef typename UserFunc::Ret T;

	typedef cluster::skeleton_task<
			_starpu::scan_first<UserFunc>,
			std::tuple<T, T>,
			std::tuple<T>,
			std::tuple<>,
			std::tuple<>>
		scan_task_first;
	typedef cluster::skeleton_task<
			_starpu::scan<UserFunc>,
			std::tuple<T, T>,
			std::tuple<T>,
			std::tuple<>,
			std::tuple<>>
		scan_task;
	typedef cluster::skeleton_task<
			_starpu::scan_update<UserFunc>,
			std::tuple<T>,
			std::tuple<>,
			std::tuple<>,
			std::tuple<>>
		scan_update;
	typedef cluster::skeleton_task<
			_starpu::scan_add<UserFunc>,
			std::tuple<T>,
			std::tuple<T>,
			std::tuple<>,
			std::tuple<>>
		scan_add;

	auto constexpr static uniform_indices = typename make_pack_indices<0>::type{};

	T m_initial;
	ScanMode m_mode;

public:
	Scan(CUScan, CUUpdate, CUAdd) noexcept
	: scan_task_first("Scan first"),
		scan_task("Scan"),
		scan_update("Scan Update"),
		scan_add("Scan Add"),
		m_initial(),
		m_mode(ScanMode::Inclusive)
	{
	}

	~Scan() noexcept
	{
		skepu::cluster::barrier();
	}

	template<
		template<typename>class Container,
		REQUIRES_VALUE(is_skepu_container<Container<T>>)>
	auto
	operator()(Container<T> & res, Container<T> & arg) noexcept
	-> Container<T> &
	{
		if(!res.size())
			return res;

		if(res.size() != arg.size())
		{
			if(!skepu::cluster::mpi_rank())
				std::cerr << "[SkePU][Scan][operator()] "
					"Input and output sizes does not match.\n";
			std::abort();
		}

		dispatch(cont::getParent(res), cont::getParent(arg));

		return res;
	}

	template<
		template<typename>class Container,
		typename Iterator,
		REQUIRES_VALUE(is_skepu_container<Container<T>>),
		REQUIRES_VALUE(is_skepu_iterator<Iterator,T>)>
	auto
	operator()(Container<T> & res, Iterator arg) noexcept
	-> Container<T> &
	{
		if(arg.offset())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Scan][operator()] "
					"SkePU StarPU-MPI only supports iterators from begin() to end()\n";
			std::abort();
		}

		dispatch(cont::getParent(res), arg.getParent());
		return res;
	}

	template<
		typename OutIter,
		template<typename>class Container,
		REQUIRES_VALUE(is_skepu_iterator<OutIter,T>),
		REQUIRES_VALUE(is_skepu_container<Container<T>>)>
	auto
	operator()(OutIter begin, OutIter end, Container<T> & arg) noexcept
	-> OutIter
	{
		if(begin.offset()
				|| end - begin != begin.getParent().size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Scan][operator()] "
					"SkePU StarPU-MPI only supports iterators from begin() to end()\n";
			std::abort();
		}

		dispatch(begin.getParent(), cont::getParent(arg));
		return begin;
	}

	template<
		typename InIter,
		typename OutIter,
		REQUIRES_VALUE(is_skepu_iterator<InIter,T>),
		REQUIRES_VALUE(is_skepu_iterator<OutIter,T>)>
	auto
	operator()(OutIter begin, OutIter end, InIter arg) noexcept
	-> OutIter
	{
		if(begin.offset()
				|| arg.offset()
				|| end - begin != begin.getParent().size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Scan][operator()] "
					"SkePU StarPU-MPI only supports iterators from begin() to end()\n";
			std::abort();
		}

		dispatch(begin.getParent(), arg.getParent());
		return begin;
	}

	auto
	setScanMode(ScanMode const mode) noexcept
	-> void
	{
		m_mode = mode;
	}

	auto
	setStartValue(T const & initial) noexcept
	-> void
	{
		m_initial = initial;
	}

private:
	template<typename Res, typename Arg>
	auto
	dispatch(Res & res, Arg & arg) noexcept
	-> void
	{
		arg.partition();
		arg.filter(0);
		res.partition();
		res.invalidate_local_storage();
		res.filter(0);

		auto num_parts = res.num_parts();
		starpu_data_filter part_res_filter;
		std::memset(&part_res_filter, 0, sizeof(starpu_data_filter));
		part_res_filter.filter_func = starpu_vector_filter_block;
		part_res_filter.nchildren = num_parts;
		starpu_data_handle_t part_res;
		std::vector<starpu_data_handle_t> part_res_handles(num_parts, 0);
		auto part_res_data =
			std::unique_ptr<T[]>(new T[num_parts]);
		starpu_vector_data_register(
			&part_res,
			STARPU_MAIN_RAM,
			(uintptr_t)part_res_data.get(),
			(uint32_t)num_parts,
			sizeof(T));
		starpu_mpi_data_register(
			part_res,
			cluster::mpi_tag(),
			0);
		starpu_data_acquire(part_res, STARPU_W);
		starpu_data_release(part_res);
		starpu_data_partition_plan(
			part_res,
			&part_res_filter,
			part_res_handles.data());
		for(auto handle : part_res_handles)
			starpu_mpi_data_register(handle, cluster::mpi_tag(), 0);

		auto count = res.block_count_from(0);
		auto first_handles =
			std::make_tuple(
				res.handle_for(0),
				part_res_handles[0],
				arg.handle_for(0));
		scan_task_first::schedule(
			first_handles,
			count,
			m_mode,
			m_initial);

		size_t part(1);
		size_t pos(count);
		while(pos < res.size())
		{
			count = res.block_count_from(pos);
			auto handles =
				std::make_tuple(
					res.handle_for(pos),
					part_res_handles[part],
					arg.handle_for(pos));

			scan_task::schedule(
				handles,
				count,
				m_mode,
				m_initial);

			pos += count;
			++part;
		}
		auto update_handles = std::make_tuple(part_res);
		scan_update::schedule(
			update_handles,
			num_parts);
		pos = res.block_count_from(0);
		part = 0;
		while(pos < res.size())
		{
			auto count = res.block_count_from(pos);
			auto handles =
				std::make_tuple(
					res.handle_for(pos),
					part_res_handles[part]);

			scan_add::schedule(
				handles,
				count);

			pos += count;
			++part;
		}

		starpu_data_partition_clean(part_res, num_parts, part_res_handles.data());
		starpu_data_unregister_no_coherency(part_res);
	}
}; // class Scan

} // namespace backend
} // namespace skepu

#endif // SKEPU_CLUSTER_SKELETON_SCAN_HPP
