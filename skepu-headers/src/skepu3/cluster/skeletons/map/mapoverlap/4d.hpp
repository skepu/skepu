#pragma once
#ifndef SKEPU_CLUSTER_MAPOVERLAP_4D_HPP
#define SKEPU_CLUSTER_MAPOVERLAP_4D_HPP 1

#include <omp.h>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/common.hpp>
#include <skepu3/cluster/skeletons/skeleton_task.hpp>
#include <skepu3/cluster/skeletons/skeleton_utils.hpp>
#include "util.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename UserFunc>
struct map_overlap_4d
{
	typedef ConditionalIndexForwarder<UserFunc::indexed, decltype(&UserFunc::CPU)>
		F;
	typedef typename util::MapOverlapBaseType<UserFunc>::type T;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename Tensor4,
		typename Buffers,
		typename ... Args>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		Edge edge,
		T const & pad,
		Tensor4 const * const t4,
		int overlap_i, int overlap_j, int overlap_k, int overlap_l,
		size_t start_i,
		size_t i_count,
		Args && ... args) noexcept
	-> void
	{
		auto se_count = overlap_i * t4->stride_i();
		auto offset = start_i * t4->stride_i();
		auto count = i_count * t4->stride_i();
		auto in_buffers =
			std::make_tuple(std::get<EI>(buffers)...);
		auto ol_start = std::get<0>(in_buffers) + se_count;
		auto ol_end = std::get<1>(in_buffers) - count;
		auto ew_buf = std::get<2>(in_buffers);

		#pragma omp parallel num_threads(starpu_combined_worker_get_size())
		{
			auto region =
				Region4D<T>(
					edge, pad, start_i,
					ol_start, ew_buf, ol_end,
					overlap_i, overlap_j, overlap_k, overlap_l,
					t4->size_i(), i_count,
					t4->size_j(), t4->size_k(), t4->size_l(),
					t4->stride_i(), t4->stride_j());

			if(edge == Edge::None)
			{
				size_t start_row = start_i < (size_t)overlap_i ? overlap_i -start_i : 0;
				size_t end_row =
					start_i + i_count < t4->size_i() -overlap_i
					? i_count
					: start_i < t4->size_i() -overlap_i
						? t4->size_i() -overlap_i -start_i
						: 0;

				#pragma omp for
				for(size_t i = start_row; i < end_row; ++i)
					for(size_t j(overlap_j); j < t4->size_j() - overlap_j; ++j)
						for(size_t k(overlap_k); k < t4->size_k() - overlap_k; ++k)
						{
							auto k_offset =
								i * t4->stride_i()
								+ j * t4->stride_j()
								+ k * t4->size_l();
							for(size_t l(overlap_l); l < t4->size_l() - overlap_l; ++l)
							{
								util::set_index(region, Index4D{i, j, k, l});
								auto res =
									F::forward(
										UserFunc::OMP,
										Index4D{start_i + i, j, k, l},
										region,
										std::get<CI>(buffers)...,
										std::forward<Args>(args)...);
								std::tie(std::get<RI>(buffers)[k_offset +l]...) = res;
							}
						}
			}
			else
			{
				#pragma omp for
				for(size_t i = 0; i < count; ++i)
				{
					auto idx = t4->index(i);
					util::set_index(region, idx);
					auto res =
						F::forward(
							UserFunc::OMP,
							t4->index(offset +i),
							region,
							std::get<CI>(buffers)...,
							std::forward<Args>(args)...);
					std::tie(std::get<RI>(buffers)[i]...) = res;
				}
			}
		}
	}
}; // class map_overlap_4d

template<typename T>
auto static
copy_4d(void ** buffers, void * args) noexcept
-> void
{
	auto out = (T *)STARPU_VECTOR_GET_PTR(*buffers);
	auto in = (T *)STARPU_TENSOR_GET_PTR(buffers[1]);
	size_t out_offset(0);
	size_t in_offset(0);
	size_t count(0);
	starpu_codelet_unpack_args(args, &out_offset, &in_offset, &count, 0);
	out += out_offset;
	in += in_offset;

	auto threads = starpu_combined_worker_get_size();
	#pragma omp parallel num_threads(threads)
	{
		auto tid = omp_get_thread_num();
		auto lcount = count / threads;
		auto rest = count - (lcount * threads);
		auto start = tid * lcount;
		auto end = start + lcount + (tid == threads -1 ? rest : 0);

		std::copy(in + start, in + end, out + start);
	}
}

} // namespace _starpu

template<typename UserFunc, typename CUDAKernel, typename CLKernel>
class MapOverlap4D
: public util::MapOverlapBase<
		typename util::MapOverlapBaseType<UserFunc>::type>,
	public cluster::skeleton_task<
		_starpu::map_overlap_4d<UserFunc>,
		typename cluster::result_tuple<typename UserFunc::Ret>::type,
		std::tuple<
			typename util::MapOverlapBaseType<UserFunc>::type,
			typename util::MapOverlapBaseType<UserFunc>::type,
			typename util::MapOverlapBaseType<UserFunc>::type>,
		typename UserFunc::ContainerArgs,
		typename UserFunc::UniformArgs>
{
	typedef typename UserFunc::Ret Ret;
	typedef typename util::MapOverlapBaseType<UserFunc>::type T;

	typedef typename cluster::result_tuple<typename UserFunc::Ret>::type
		ResultArgs;
	typedef typename UserFunc::ElwiseArgs ElwiseArgs;
	typedef typename UserFunc::ContainerArgs ContainerArgs;
	typedef typename UserFunc::UniformArgs UniformArgs;
	typename UserFunc::ProxyTags const proxy_tags{};


	typedef util::MapOverlapBase<T> base;
	typedef cluster::skeleton_task<
			_starpu::map_overlap_4d<UserFunc>,
			typename cluster::result_tuple<Ret>::type,
			std::tuple<T, T, T>,
		typename UserFunc::ContainerArgs,
			typename UserFunc::UniformArgs>
		skeleton_task;

	auto static constexpr nresult = UserFunc::outArity;
	auto static constexpr nelwise = 1;
	auto static constexpr ncontainer =
		std::tuple_size<ContainerArgs>::value;
	auto static constexpr nuniform =
		std::tuple_size<UniformArgs>::value;

	auto static constexpr result_indices =
		typename make_pack_indices<nresult>::type{};
	auto static constexpr elwise_indices =
		typename make_pack_indices<nresult + nelwise, nresult>::type{};
	auto static constexpr container_indices =
		typename make_pack_indices<
			nresult + nelwise + ncontainer,
			nresult + nelwise>::type{};
	auto static constexpr uniform_indices =
		typename make_pack_indices<
			nresult + nelwise + ncontainer + nuniform,
			nresult + nelwise + ncontainer>::type{};

	int m_overlap_i;
	int m_overlap_j;
	int m_overlap_k;
	int m_overlap_l;

	starpu_codelet copy_cl;

public:
	MapOverlap4D(CUDAKernel)
	: skeleton_task("MapOverlap4D"),
		m_overlap_i(0), m_overlap_j(0), m_overlap_k(0), m_overlap_l(0)
	{
		starpu_codelet_init(&copy_cl);
		copy_cl.name = "copy4d";
		copy_cl.cpu_funcs[0] = _starpu::copy_4d<T>;
		copy_cl.nbuffers = STARPU_VARIABLE_NBUFFERS;
		copy_cl.type = STARPU_FORKJOIN;
		copy_cl.max_parallelism = INT_MAX;
	}

	~MapOverlap4D() noexcept
	{
		skepu::cluster::barrier();
	}

	auto
	setOverlap(int overlap) noexcept
	-> void
	{
		m_overlap_i = overlap;
		m_overlap_j = overlap;
		m_overlap_k = overlap;
		m_overlap_l = overlap;
	}

	auto
	setOverlap(
		int overlap_i,
		int overlap_j,
		int overlap_k,
		int overlap_l) noexcept
	-> void
	{
		m_overlap_i = overlap_i;
		m_overlap_j = overlap_j;
		m_overlap_k = overlap_k;
		m_overlap_l = overlap_l;
	}

	auto
	getOverlap() noexcept
	-> std::tuple<int, int, int, int>
	{
		return std::make_tuple(m_overlap_i, m_overlap_j, m_overlap_k, m_overlap_l);
	}

	template<typename... Args>
	auto
	operator()(Args &&... args) noexcept
	-> decltype(get<0>(args...))
	{
		check_sizes(
			result_indices,
			elwise_indices,
			std::forward<Args>(args)...);

		if(!get<0>(args...).size())
			return get<0>(args...);

		dispatch(
			result_indices,
			elwise_indices,
			container_indices,
			uniform_indices,
			std::forward<Args>(args)...);

		return get<0>(args...);
	}

private:
	template<
		size_t... OI,
		size_t... EI,
		typename... CallArgs>
	auto
	check_sizes(
		pack_indices<OI...>,
		pack_indices<EI...>,
		CallArgs&&... args) noexcept
	-> void
	{
		static_assert(
			conjunction(
				is_skepu_tensor4<
						typename std::remove_reference<decltype(get<OI>(args...))>::type
					>::value...),
			"[SkePU][MapOverlap] 4D MapOverlap requires all output containers "
			"to be of type skepu::Tensor4");
		static_assert(
			conjunction(
				is_skepu_tensor4<
						typename std::remove_reference<decltype(get<EI>(args...))>::type
					>::value...),
			"[SkePU][MapOverlap] 4D MapOverlap requires all element-wise arguments "
			"to be of type skepu::Tensor4");

		size_t size_i = get<0>(args...).size_i();
		size_t size_j = get<0>(args...).size_j();
		size_t size_k = get<0>(args...).size_k();
		size_t size_l = get<0>(args...).size_l();

		if(
				size_i < m_overlap_i
				|| size_j < m_overlap_j
				|| size_k < m_overlap_k
				|| size_l < m_overlap_l)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Tensor4 must be at least of size "
					"(overlap_i, overlap_j, overlap_k, overlap_l)"
					"\n";
			std::abort();
		}

		if(disjunction(
			((get<OI>(args...).size_i() < size_i)
			&& (get<OI>(args...).size_j() < size_j)
			&& (get<OI>(args...).size_k() < size_k)
			&& (get<OI>(args...).size_l() < size_l))...))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Non-matching output container sizes.\n";
			std::abort();
		}

		if(disjunction(
			((get<EI>(args...).size_i() != size_i)
			&& (get<EI>(args...).size_j() != size_j)
			&& (get<EI>(args...).size_k() != size_k)
			&& (get<EI>(args...).size_l() != size_l))...))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Non-matching input container sizes.\n";
			std::abort();
		}
	}

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		size_t ... UI,
		typename ... Args>
	auto
	dispatch(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		pack_indices<UI...>,
		Args && ... args) noexcept
	-> void
	{
		dispatch(
			typename make_pack_indices<nresult>::type{},
			typename make_pack_indices<nelwise>::type{},
			typename make_pack_indices<ncontainer>::type{},
			std::tie(cont::getParent(get<RI>(args...))...),
			std::tie(cont::getParent(get<EI>(args...))...),
			std::tie(cont::getParent(get<CI>(args...))...),
			std::forward<decltype(get<UI>(args...))>(get<UI>(args...))...);
	}

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename ... RA,
		typename ... EA,
		typename ... CA,
		typename ... Args>
	auto
	dispatch(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		std::tuple<RA...> result_args,
		std::tuple<EA...> elwise_args,
		std::tuple<CA...> container_args,
		Args && ... args) noexcept
	-> void
	{
		pack_expand((
			std::get<EI>(elwise_args).partition(),
			std::get<EI>(elwise_args).filter(0),
			0)...);
		pack_expand((
			// Gather everything...
			std::get<CI>(container_args).filter(0),
			skeleton_task::handle_container_arg(
				std::get<CI>(container_args),
				std::get<CI>(proxy_tags)),
			0)...);
		pack_expand((
			std::get<RI>(result_args).partition(),
			std::get<RI>(result_args).invalidate_local_storage(),
			std::get<RI>(result_args).filter(0),
			0)...);

		auto * res_0_ptr = &std::get<0>(result_args);
		std::vector<util::border_region<T>> borders;
		auto border_size = m_overlap_i * std::get<0>(result_args).stride_i();
		size_t i(0);
		while(i < std::get<0>(result_args).size_i())
		{
			auto i_count =
				std::get<0>(result_args).block_count_i(i);
			auto rank =
				starpu_mpi_data_get_rank(std::get<0>(result_args).handle_for_i(i));

			borders.emplace_back(border_size, rank);
			borders.emplace_back(border_size, rank);
			auto & start_border = *(borders.end() -2);
			auto & end_border = borders.back();
			fill(start_border, end_border, std::get<0>(elwise_args), i, i_count);

			auto handles =
				std::make_tuple(
					std::get<RI>(result_args).handle_for_i(i)...,
					start_border.handle(),
					end_border.handle(),
					std::get<0>(elwise_args).handle_for_i(i),
					skeleton_task::container_handle(
						std::get<CI>(container_args),
						std::get<CI>(proxy_tags),
						i)...);

			skeleton_task::schedule(
				handles,
				base::m_edge,
				base::m_pad,
				res_0_ptr,
				m_overlap_i, m_overlap_j, m_overlap_k, m_overlap_l,
				i, i_count,
				std::forward<Args>(args)...);

			i += i_count;
		}
	}

	template<
		typename T,
		typename Tensor4>
	auto
	fill(
		util::border_region<T> & start,
		util::border_region<T> & end,
		Tensor4 & t4,
		size_t i_pos,
		size_t i_count) noexcept
	-> void
	{
		size_t ol_start(i_pos -m_overlap_i);
		if(ol_start >= t4.size_i())
			ol_start += t4.size_i();

		size_t i(0);
		while(i < m_overlap_i)
		{
			auto block_i = ol_start +i;
			if(block_i >= t4.size_i())
				block_i -= t4.size_i();

			auto b_offset = i * t4.stride_i();
			auto t4_offset = t4.block_offset_i(block_i);
			auto block_count = std::min(m_overlap_i -i, t4.block_count_i(block_i));
			auto count = block_count * t4.stride_i();
			auto b_handle = start.handle();
			auto t4_handle = t4.handle_for_i(block_i);

			starpu_mpi_task_insert(
				MPI_COMM_WORLD,
				&copy_cl,
				STARPU_RW|STARPU_SSEND, b_handle,
				STARPU_R, t4_handle,
				STARPU_VALUE, &b_offset, sizeof(b_offset),
				STARPU_VALUE, &t4_offset, sizeof(t4_offset),
				STARPU_VALUE, &count, sizeof(count),
				STARPU_EXECUTE_ON_DATA, t4_handle,
				0);

			i += block_count;
		}

		ol_start = i_pos + i_count;

		i = 0;
		while(i < m_overlap_i)
		{
			auto block_i = ol_start +i;
			if(block_i >= t4.size_i())
				block_i -= t4.size_i();

			auto b_offset = i * t4.stride_i();
			auto t4_offset = t4.block_offset_i(block_i);
			auto block_count = std::min(m_overlap_i -i, t4.block_count_i(block_i));
			auto count = block_count * t4.stride_i();
			auto b_handle = end.handle();
			auto t4_handle = t4.handle_for_i(block_i);

			starpu_mpi_task_insert(
				MPI_COMM_WORLD,
				&copy_cl,
				STARPU_RW|STARPU_SSEND, b_handle,
				STARPU_R, t4_handle,
				STARPU_VALUE, &b_offset, sizeof(b_offset),
				STARPU_VALUE, &t4_offset, sizeof(t4_offset),
				STARPU_VALUE, &count, sizeof(count),
				STARPU_EXECUTE_ON_DATA, t4_handle,
				0);

			i += block_count;
		}
	}
}; // class MapOverlap4D

} // namespace backend
} // namespace skepu

#endif // SKEPU_CLUSTER_MAPOVERLAP_4D_HPP
