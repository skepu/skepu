#pragma once
#ifndef SKEPU_CLUSTER_MAPOVERLAP_2D_HPP
#define SKEPU_CLUSTER_MAPOVERLAP_2D_HPP 1

#include <sstream>
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
struct map_overlap_2d
{
	typedef ConditionalIndexForwarder<UserFunc::indexed, decltype(&UserFunc::CPU)>
		F;
	typedef typename util::MapOverlapBaseType<UserFunc>::type T;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename Matrix,
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
		Matrix const * const m,
		int overlap_i, int overlap_j,
		size_t row,
		size_t row_count,
		Args && ... args) noexcept
	-> void
	{
		auto se_count = overlap_i * m->size_j();
		auto offset = row * m->size_j();
		auto count = row_count * m->size_j();
		auto in_buffers =
			std::make_tuple(std::get<EI>(buffers)...);
		auto ol_start = std::get<0>(in_buffers) + se_count;
		auto ol_end = std::get<1>(in_buffers) - count;
		auto ew_buf = std::get<2>(in_buffers);

		#pragma omp parallel num_threads(starpu_combined_worker_get_size())
		{
			auto region =
				Region2D<T>(
					edge, pad, row,
					ol_start, ew_buf, ol_end,
					overlap_i, overlap_j,
					m->size_i(), row_count,
					m->size_j());

			if(edge == Edge::None)
			{
				size_t start_row = row < (size_t)overlap_i ? overlap_i -row : 0;
				size_t end_row =
					row + row_count < m->size_i() - overlap_i
					? row_count
					: row < m->size_i() - overlap_i
						? m->size_i() -overlap_i - row
						: 0;

				#pragma omp for
				for(size_t i = start_row; i < end_row; ++i)
				{
					auto row_offset = i * m->size_j();
					for(size_t j(overlap_j); j < m->size_j() - overlap_j; ++j)
					{
						util::set_index(region, Index2D{i, j});
						auto res =
							F::forward(
								UserFunc::OMP,
								Index2D{start_row + i, j},
								region,
								std::get<CI>(buffers)...,
								std::forward<Args>(args)...);
						std::tie(std::get<RI>(buffers)[row_offset +j]...) = res;
					}
				}
			}
			else
			{
				#pragma omp for
				for(size_t i = 0; i < count; ++i)
				{
					util::set_index(region, m->index(i));
					auto res =
						F::forward(
							UserFunc::OMP,
							m->index(offset +i),
							region,
							std::get<CI>(buffers)...,
							std::forward<Args>(args)...);
					std::tie(std::get<RI>(buffers)[i]...) = res;
				}
			}
		}
	}
}; // class map_overlap_2d

template<typename T>
auto static
copy_2d(void ** buffers, void * args) noexcept
-> void
{
	auto out = (T *)STARPU_VECTOR_GET_PTR(*buffers);
	auto in = (T *)STARPU_MATRIX_GET_PTR(buffers[1]);
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
class MapOverlap2D
: public util::MapOverlapBase<
		typename util::MapOverlapBaseType<UserFunc>::type>,
	public cluster::skeleton_task<
		_starpu::map_overlap_2d<UserFunc>,
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
			_starpu::map_overlap_2d<UserFunc>,
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

	starpu_codelet copy_cl;

public:
	MapOverlap2D(CUDAKernel)
	: skeleton_task("MapOverlap2D"),
		m_overlap_i(0), m_overlap_j(0)
	{
		starpu_codelet_init(&copy_cl);
		copy_cl.name = "copy2d";
		copy_cl.cpu_funcs[0] = _starpu::copy_2d<T>;
		copy_cl.nbuffers = STARPU_VARIABLE_NBUFFERS;
		copy_cl.type = STARPU_FORKJOIN;
		copy_cl.max_parallelism = INT_MAX;
	}

	~MapOverlap2D() noexcept
	{
		skepu::cluster::barrier();
	}

	auto
	setOverlap(int overlap) noexcept
	-> void
	{
		m_overlap_i = overlap;
		m_overlap_j = overlap;
	}

	auto
	setOverlap(
		int overlap_i,
		int overlap_j) noexcept
	-> void
	{
		m_overlap_i = overlap_i;
		m_overlap_j = overlap_j;
	}

	auto
	getOverlap() noexcept
	-> std::tuple<int, int>
	{
		return std::make_tuple(m_overlap_i, m_overlap_j);
	}

	template<typename... Args>
	auto
	operator()(Args&&... args) noexcept
	-> decltype(get<0>(args...))
	{
		check_type_and_sizes(
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
	check_type_and_sizes(
		pack_indices<OI...>,
		pack_indices<EI...>,
		CallArgs&&... args) noexcept
	-> void
	{
		static_assert(
			conjunction(
				is_skepu_matrix<
						typename std::remove_reference<decltype(get<OI>(args...))>::type
					>::value...),
			"[SkePU][MapOverlap] 2D MapOverlap requires all output containers "
			"to be of type skepu::Matrix");
		static_assert(
			conjunction(
				is_skepu_matrix<
						typename std::remove_reference<decltype(get<EI>(args...))>::type
					>::value...),
			"[SkePU][MapOverlap] 2D MapOverlap requires all element-wise arguments "
			"to be of type skepu::Matrix");

		size_t size_i = get<0>(args...).size_i();
		size_t size_j = get<0>(args...).size_j();

		if(size_i < m_overlap_i || size_j < m_overlap_i)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Matrix must be at least of size (overlap_i, overlap_j)\n";
			std::abort();
		}

		if(disjunction(
			((get<OI>(args...).size_i() < size_i)
			&& (get<OI>(args...).size_j() < size_j))...))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Non-matching output container sizes.\n";
			std::abort();
		}

		if(disjunction(
			((get<EI>(args...).size_i() != size_i)
			&& (get<EI>(args...).size_j() != size_j))...))
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
		auto border_size = m_overlap_i * std::get<0>(result_args).size_j();
		size_t row(0);
		while(row < std::get<0>(result_args).size_i())
		{
			auto row_count =
				std::get<0>(result_args).block_count_row(row);
			auto rank =
				starpu_mpi_data_get_rank(std::get<0>(result_args).handle_for_row(row));
			borders.emplace_back(border_size, rank);
			borders.emplace_back(border_size, rank);
			auto & start_border = *(borders.end() -2);
			auto & end_border = borders.back();
			fill(start_border, end_border, std::get<0>(elwise_args), row, row_count);

			auto handles =
				std::make_tuple(
					std::get<RI>(result_args).handle_for_row(row)...,
					start_border.handle(),
					end_border.handle(),
					std::get<0>(elwise_args).handle_for_row(row),
					skeleton_task::container_handle(
						std::get<CI>(container_args),
						std::get<CI>(proxy_tags),
						row)...);

			skeleton_task::schedule(
				handles,
				base::m_edge,
				base::m_pad,
				res_0_ptr,
				m_overlap_i, m_overlap_j,
				row, row_count,
				std::forward<Args>(args)...);

			row += row_count;
			cluster::barrier();
		}
	}

	template<
		typename T,
		typename Matrix>
	auto
	fill(
		util::border_region<T> & start,
		util::border_region<T> & end,
		Matrix & m,
		size_t row,
		size_t row_count) noexcept
	-> void
	{
		size_t ol_start = row - m_overlap_i;
		if(ol_start >= m.size_i())
			ol_start += m.size_i();

		size_t i = 0;
		while(i < m_overlap_i)
		{
			auto current_row = ol_start +i;
			if(current_row >= m.size_i())
				current_row -= m.size_i();
			size_t m_offset = m.block_offset_row(current_row);
			auto i_count =
				std::min(m_overlap_i -i, m.block_count_row(current_row));
			size_t count = i_count * m.size_j();
			auto b_handle = start.handle();
			auto m_handle = m.handle_for_row(current_row);
			size_t b_offset = i * m.size_j();

			starpu_mpi_task_insert(
				MPI_COMM_WORLD,
				&copy_cl,
				STARPU_RW|STARPU_SSEND, b_handle,
				STARPU_R, m_handle,
				STARPU_VALUE, &b_offset, sizeof(b_offset),
				STARPU_VALUE, &m_offset, sizeof(m_offset),
				STARPU_VALUE, &count, sizeof(count),
				STARPU_EXECUTE_ON_DATA, m_handle,
				0);

			i += i_count;
		}

		ol_start = (row + row_count);

		i = 0;
		while(i < m_overlap_i)
		{
			auto current_row = (ol_start +i);
			if(current_row >= m.size_i())
				current_row -= m.size_i();
			size_t m_offset = m.block_offset_row(current_row);
			auto m_handle = m.handle_for_row(current_row);
			auto i_count =
				std::min(m_overlap_i -i, m.block_count_row(current_row));
			size_t count = i_count * m.size_j();
			auto b_handle = end.handle();
			size_t b_offset = i * m.size_j();

			starpu_mpi_task_insert(
				MPI_COMM_WORLD,
				&copy_cl,
				STARPU_RW|STARPU_SSEND, b_handle,
				STARPU_R, m_handle,
				STARPU_VALUE, &b_offset, sizeof(b_offset),
				STARPU_VALUE, &m_offset, sizeof(m_offset),
				STARPU_VALUE, &count, sizeof(count),
				STARPU_EXECUTE_ON_DATA, m_handle,
				0);

			i += i_count;
		}
	}
}; // class MapOverlap2D

} // namespace backend
} // namespace skepu

#endif // SKEPU_CLUSTER_MAPOVERLAP_2D_HPP
