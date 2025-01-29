#pragma once
#ifndef SKEPU_CLUSTER_MAPOVERLAP_1D_HPP
#define SKEPU_CLUSTER_MAPOVERLAP_1D_HPP 1

#include <typeinfo>

#include <omp.h>

#include <skepu3/cluster/cluster.hpp>
#include <skepu3/cluster/skeletons/skeleton_task.hpp>
#include "util.hpp"

namespace skepu {
namespace backend {
namespace _starpu {

template<typename UserFunc>
struct map_overlap_1d
{
	typedef ConditionalIndexForwarder<UserFunc::indexed, decltype(&UserFunc::CPU)>
		F;
	typedef typename UserFunc::Ret out_t;
	typedef typename util::MapOverlapBaseType<UserFunc>::type in_t;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename Buffers,
		typename ... Args>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		int overlap,
		Edge edge,
		in_t pad,
		size_t v_size,
		size_t pos,
		size_t count,
		Args &&... args) noexcept
	-> void
	{
		auto in_buffers =
			std::make_tuple(std::get<EI>(buffers)...);
		auto ol_start = std::get<0>(in_buffers) + overlap;
		auto ol_end = std::get<1>(in_buffers) - count;
		auto ew_buf = std::get<2>(in_buffers);

		#pragma omp parallel num_threads(starpu_combined_worker_get_size())
		{
			auto region =
				Region1D<in_t>(
					edge, pad, pos,
					ol_start, ew_buf, ol_end,
					overlap,
					v_size, count);

			if(edge == Edge::None)
			{
				size_t start_i = pos < (size_t)overlap ? overlap - pos : 0;
				size_t end_i =
					pos + count < v_size -overlap
					? count
					: pos < v_size -overlap
						? v_size -overlap -pos
						: 0;

				#pragma omp for
				for(size_t i = start_i; i < end_i; ++i)
				{
					util::set_index(region, Index1D{i});
					auto res =
						F::forward(
							UserFunc::OMP,
							Index1D{pos + i},
							region,
							std::get<CI>(buffers)...,
							std::forward<Args>(args)...);
					std::tie(std::get<RI>(buffers)[i]...) = res;
				}
			}
			else
			{
				#pragma omp for
				for(size_t i = 0; i < count; ++i)
				{
					util::set_index(region, Index1D{i});
					auto res =
						F::forward(
							UserFunc::OMP,
							Index1D{pos +i},
							region,
							std::get<CI>(buffers)...,
							std::forward<Args>(args)...);
					std::tie(std::get<RI>(buffers)[i]...) = res;
				}
			}
		}
	}
}; // struct map_overlap_1d

template<typename UserFunc>
struct map_overlap_1d_rowwise
{
	typedef ConditionalIndexForwarder<UserFunc::indexed, decltype(&UserFunc::CPU)>
		F;
	typedef typename UserFunc::Ret out_t;
	typedef typename util::MapOverlapBaseType<UserFunc>::type in_t;

	template<
		size_t ... RI,
		size_t ... EI,
		size_t ... CI,
		typename Buffers,
		typename ... Args>
	auto static
	run(
		pack_indices<RI...>,
		pack_indices<EI...>,
		pack_indices<CI...>,
		Buffers && buffers,
		int overlap,
		Edge edge,
		in_t pad,
		size_t count,
		size_t cols,
		Args &&... args) noexcept
	-> void
	{
		auto matrix = std::get<0>(std::tie(std::get<EI>(buffers)...));

		size_t start_col = (edge == Edge::None ? overlap : 0);
		size_t end_col = (edge == Edge::None ? cols - overlap : cols);
		#pragma omp parallel for num_threads(starpu_combined_worker_get_size())
		for(size_t row = 0; row < count; ++row)
		{
			auto row_offset = row * cols;
			auto row_ptr = matrix + row_offset;
			auto end = row_ptr - cols;
			auto start = row_ptr + cols;
			Region1D<in_t> region(
				edge, pad, 0,
				start, row_ptr, end,
				overlap,
				cols, cols);

			for(size_t col(start_col); col < end_col; ++col)
			{
				util::set_index(region, Index1D{col});
				auto res =
					F::forward(
						UserFunc::OMP,
						Index1D{col},
						region,
						std::get<CI>(buffers)...,
						std::forward<Args>(args)...);
				std::tie(std::get<RI>(buffers)[row_offset +col]...) = res;
			}
		}
	}
}; // struct map_overlap_1d

template<typename T>
auto static
copy_1d(void ** buffers, void * args) noexcept
-> void
{
	auto out = (T *)STARPU_VECTOR_GET_PTR(*buffers);
	auto in = (T *)STARPU_VECTOR_GET_PTR(buffers[1]);
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

template<
	typename UserFunc,
	typename CUDAKernel,
	typename C2,
	typename C3,
	typename C4,
	typename CLKernel>
class MapOverlap1D
: public util::MapOverlapBase<typename UserFunc::Ret>,
	public cluster::skeleton_task<
		_starpu::map_overlap_1d<UserFunc>,
		typename cluster::result_tuple<typename UserFunc::Ret>::type,
		std::tuple<
			typename util::MapOverlapBaseType<UserFunc>::type,
			typename util::MapOverlapBaseType<UserFunc>::type,
			typename util::MapOverlapBaseType<UserFunc>::type>,
		typename UserFunc::ContainerArgs,
		typename UserFunc::UniformArgs>,
	public cluster::skeleton_task<
		_starpu::map_overlap_1d_rowwise<UserFunc>,
		typename cluster::result_tuple<typename UserFunc::Ret>::type,
		std::tuple<typename util::MapOverlapBaseType<UserFunc>::type>,
		typename UserFunc::ContainerArgs,
		typename UserFunc::UniformArgs>
{
	typedef typename UserFunc::Ret out_t;
	typedef typename util::MapOverlapBaseType<UserFunc>::type in_t;
	typedef util::MapOverlapBase<out_t> base;

	typedef
			typename cluster::result_tuple<typename UserFunc::Ret>::type
		ResultArgs;
	typedef std::tuple<in_t, in_t, in_t> VectorElwiseArgs;
	typedef std::tuple<in_t> MatrixElwiseArgs;
	typedef typename UserFunc::ContainerArgs ContainerArgs;
	typedef typename UserFunc::UniformArgs UniformArgs;
	typedef typename UserFunc::ProxyTags ProxyTags;

	typedef cluster::skeleton_task<
			_starpu::map_overlap_1d<UserFunc>,
			ResultArgs,
			VectorElwiseArgs,
			ContainerArgs,
			UniformArgs>
		vector_task;
	typedef cluster::skeleton_task<
			_starpu::map_overlap_1d_rowwise<UserFunc>,
			ResultArgs,
			MatrixElwiseArgs,
			ContainerArgs,
			UniformArgs>
		matrix_task;

	auto constexpr static nresult =
		std::tuple_size<ResultArgs>::value;
	auto constexpr static ncontainer =
		std::tuple_size<ContainerArgs>::value;
	auto constexpr static nuniform =
		std::tuple_size<UniformArgs>::value;
	auto constexpr static nproxy =
		std::tuple_size<ProxyTags>::value;

	auto constexpr static result_indices =
		typename make_pack_indices<nresult>::type{};
	auto constexpr static elwise_indices =
		typename make_pack_indices<nresult +1, nresult>::type{};
	auto constexpr static container_indices =
		typename make_pack_indices<ncontainer + nresult +1, nresult +1>::type{};
	auto constexpr static uniform_indices =
		typename make_pack_indices<
				nuniform + ncontainer + nresult +1,
				ncontainer + nresult +1>
			::type{};

	auto constexpr static proxy_tags = ProxyTags{};

	size_t m_overlap;

	starpu_codelet copy_cl;

public:
	MapOverlap1D(CUDAKernel, C2, C3, C4)
	: vector_task("MapOverlap1D Vector"),
		matrix_task("MapOverlap1D Matrix"),
		m_overlap(0)
	{
		starpu_codelet_init(&copy_cl);
		copy_cl.cpu_funcs[0] = _starpu::copy_1d<in_t>;
		copy_cl.type = STARPU_FORKJOIN;
		copy_cl.nbuffers = 2;
	}

	~MapOverlap1D() noexcept
	{
		skepu::cluster::barrier();
	}

	auto
	setOverlap(size_t count) noexcept
	-> void
	{
		m_overlap = count;
	}

	auto
	getOverlap() const noexcept
	-> size_t
	{
		return m_overlap;
	}

	template<
		typename ResultT,
		typename ... Args>
	auto
	operator()(ResultT & res, Args &&... args) noexcept
	-> ResultT &
	{
		dispatch(
			result_indices,
			elwise_indices,
			container_indices,
			uniform_indices,
			res,
			std::forward<Args>(args)...);

		return res;
	}

private:
	template<
		size_t... OI,
		typename ... OutVectors,
		typename InVector>
	auto
	check_vector_sizes(
		pack_indices<OI...>,
		std::tuple<OutVectors...> & out,
		InVector & in) noexcept
	-> void
	{
		size_t size = std::get<0>(out).size();
		if(size < m_overlap)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Vector must be at least of size overlap\n";
			std::abort();
		}

		if(disjunction(
			(std::get<OI>(out).size() < size)...))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Non-matching output container sizes.\n";
			std::abort();
		}

		if(in.size() != size)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Non-matching input container sizes.\n";
			std::abort();
		}
	}

	template<
		size_t... OI,
		typename ... OutMatrices,
		typename InMatrix>
	auto
	check_matrix_sizes(
		pack_indices<OI...>,
		std::tuple<OutMatrices...> & out,
		InMatrix & in) noexcept
	-> void
	{
		size_t size_i = std::get<0>(out).size_i();
		size_t size_j = std::get<0>(out).size_j();

		if(base::m_overlap_mode == Overlap::RowWise && size_i < m_overlap)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"With skepu::Overlap::RowWise, the skepu::Matrix needs to have at "
					"least overlap number of rows.\n";
			std::abort();
		}
		else if(base::m_overlap_mode == Overlap::ColWise && size_j < m_overlap)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"With skepu::Overlap::ColWise, the skepu::Matrix needs to have at "
					"least overlap number of columns.\n";
			std::abort();
		}

		if(disjunction(
			(std::get<OI>(out).size_i() < size_i)...,
			(std::get<OI>(out).size_j() < size_j)...))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][MapOverlap] "
					"Non-matching output container sizes.\n";
			std::abort();
		}

		if(in.size_i() != size_i
				&& in.size_j() != size_i)
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
		Args &&... args) noexcept
	-> void
	{
		dispatch(
			typename make_pack_indices<nresult>::type{},
			typename make_pack_indices<ncontainer>::type{},
			std::tie(get<RI>(args...)...),
			std::forward<decltype(get<EI>(args...))>(get<EI>(args...))...,
			std::tie(get<CI>(args...)...),
			std::forward<decltype(get<UI>(args...))>(get<UI>(args...))...);
	}

	template<
		size_t ... RI,
		size_t ... CI,
		typename ... ResVectors,
		typename ArgVector,
		typename ... Containers,
		typename ... Args,
		REQUIRES_VALUE(is_skepu_vector<ArgVector>)>
	auto
	dispatch(
		pack_indices<RI...> ri,
		pack_indices<CI...>,
		std::tuple<ResVectors...> res,
		ArgVector & in,
		std::tuple<Containers...> containers,
		Args &&... args) noexcept
	-> void
	{
		static_assert(
			conjunction(
				is_skepu_vector<typename std::tuple_element<RI, decltype(res)>::type>::value...),
			"[SkePU][MapOverlap][operator(skepu::Vector ...)] "
			"All result containers must be of type skepu::Vector.");
		static_assert(
			conjunction(
				std::is_same<
					typename std::decay<typename std::tuple_element<RI, decltype(res)>::type>::type::value_type,
					typename std::tuple_element<RI, ResultArgs>::type>::value...),
			"[SkePU][MapOverlap][operator(skepu::Vector ...)] "
			"Output container types does not match the return types of the user "
			"function");
		static_assert(
			std::is_same<typename ArgVector::value_type, in_t>::value,
			"[SkePU][MapOverlap][operator(skepu::Matrix ...)] "
			"Element-wise skepu::Matrix is not of the same type as the Region1D in "
			"the user function.");

		check_vector_sizes(ri, res, in);

		auto & arg_part = cont::getParent(in);
		arg_part.partition();
		arg_part.filter(0);

		pack_expand(
			handle_container_arg(
				cont::getParent(get<CI>(args...)),
				std::get<CI>(proxy_tags))...);

		// Refering to that the implementation of a skepu container is called
		// <container>_partition
		auto res_parts =
			std::tie(cont::getParent(std::get<RI>(res))...);
		pack_expand((
			std::get<RI>(res_parts).filter(0),
			std::get<RI>(res_parts).partition(),
			std::get<RI>(res_parts).invalidate_local_storage(),
			0)...);

		std::vector<util::border_region<in_t>> borders;
		size_t pos{0};
		size_t size(std::get<0>(res_parts).size());
		while(pos < size)
		{
			auto count = std::get<0>(res_parts).block_count_from(pos);
			auto res_handle = std::get<0>(res_parts).handle_for(pos);
			auto rank =
				starpu_mpi_data_get_rank(
					std::get<0>(res_parts).handle_for(pos));

			// create start + end region buffer
			borders.emplace_back(m_overlap, rank);
			borders.emplace_back(m_overlap, rank);
			auto & start_region = *(borders.end() -2);
			auto & end_region = borders.back();
			fill(start_region, end_region, arg_part, pos, count);

			// Schedule StarPU task.
			auto handles =
				std::make_tuple(
					std::get<RI>(res_parts).handle_for(pos)...,
					start_region.handle(),
					end_region.handle(),
					arg_part.handle_for(pos),
					container_handle(
						cont::getParent(std::get<CI>(containers),
						cont::getParent(std::get<CI>(proxy_tags)),
						pos))...);

			vector_task::schedule(
				handles,
				m_overlap,
				base::m_edge,
				base::m_pad,
				size,
				pos,
				count,
				std::forward<Args>(args)...);

			pos += count;
		}
	}

	template<
		size_t ... RI,
		size_t ... CI,
		typename ... ResMatrices,
		typename ArgMatrix,
		typename ... Containers,
		typename ... Args,
		REQUIRES_VALUE(is_skepu_matrix<ArgMatrix>)>
	auto
	dispatch(
		pack_indices<RI...> ri,
		pack_indices<CI...> ci,
		std::tuple<ResMatrices...> res,
		ArgMatrix & arg,
		std::tuple<Containers...> containers,
		Args &&... args) noexcept
	-> void
	{
		static_assert(
			conjunction(
				is_skepu_matrix<typename std::tuple_element<RI, decltype(res)>::type>::value...),
			"[SkePU][MapOverlap][operator(skepu::Matrix ...)] "
			"All result containers must be of type skepu::Matrix.");
		static_assert(
			conjunction(
				std::is_same<
					typename std::decay<typename std::tuple_element<RI, decltype(res)>::type>::type::value_type,
					typename std::tuple_element<RI, ResultArgs>::type>::value...),
			"[SkePU][MapOverlap][operator(skepu::Matrix ...)] "
			"Output container types does not match the return types of the user "
			"function");
		static_assert(
			std::is_same<typename ArgMatrix::value_type, in_t>::value,
			"[SkePU][MapOverlap][operator(skepu::Matrix ...)] "
			"Element-wise skepu::Matrix is not of the same type as the Region1D in "
			"the user function.");

		check_matrix_sizes(ri, res, arg);

		switch(base::m_overlap_mode)
		{
			case Overlap::ColWise:
			{
				/*
				 * To simplify the communication between ranks (and hopefully for larger
				 * skepu::Region1Ds, reduce the number of cache misses), we transpose
				 * the arg parameter into a temporary container.
				 */
				auto transposed_arg =
					ArgMatrix(arg.size_j(), arg.size_i());
				cont::getParent(arg).transpose_to(cont::getParent(transposed_arg));

				/*
				 * The result matrices also needs to be transposed. Ideally, since we
				 * don't care about the current contents of the result matrices, we
				 * could just flip the matrix dimentions. But currently, this feature
				 * contains bugs.
				 */
				auto transposed_res =
					std::make_tuple(
						typename std::remove_reference<ResMatrices>::type(std::get<RI>(res).size_j(), std::get<RI>(res).size_i())...);
				pack_expand((
					cont::getParent(std::get<RI>(res)).transpose_to(
						cont::getParent(std::get<RI>(transposed_res))),
					0)...);

				/*
				 * We can now use the rowwise matrix implementation, since everything
				 * container is tranposed.
				 */
				dispatch_rowwise(
					ri,
					ci,
					transposed_res,
					transposed_arg,
					containers,
					std::forward<Args>(args)...);

				/*
				 * Since the result matrices are transposed, lets transpose them back.
				 */
				pack_expand((
					cont::getParent(std::get<RI>(transposed_res)).transpose_to(
						cont::getParent(std::get<RI>(res))),
					0)...);
				break;
			}
			case Overlap::RowWise:
				dispatch_rowwise(
					ri,
					ci,
					res,
					arg,
					containers,
					std::forward<Args>(args)...);
				break;
			default:
			{
				if(!cluster::mpi_rank())
					std::cerr << "[SkePU][MapOverlap] Overlap mode not supported.\n";
				std::abort();
			}
		}
	}

	template<
		size_t ... RI,
		size_t ... CI,
		typename ... ResMatrices,
		typename ArgMatrix,
		typename ... Containers,
		typename ... Args>
	auto
	dispatch_rowwise(
		pack_indices<RI...>,
		pack_indices<CI...>,
		std::tuple<ResMatrices...> & res,
		ArgMatrix & arg,
		std::tuple<Containers...> containers,
		Args &&... args) noexcept
	-> void
	{
		auto & arg_ref = cont::getParent(arg);
		arg_ref.filter(0);
		arg_ref.partition();

		pack_expand((
			cont::getParent(std::get<CI>(containers)).filter(0),
			matrix_task::handle_container_arg(
				cont::getParent(std::get<CI>(containers)),
				std::get<CI>(proxy_tags)),
			0)...);

		// Referencing that the implementation class of the containers are called
		// <container>_partition
		auto res_parts =
			std::tie(cont::getParent(std::get<RI>(res))...);
		pack_expand((
			std::get<RI>(res_parts).filter(0),
			std::get<RI>(res_parts).partition(),
			std::get<RI>(res_parts).invalidate_local_storage(),
			0)...);

		size_t row(0);
		size_t rows(std::get<0>(res_parts).size_i());
		size_t cols(std::get<0>(res_parts).size_j());
		while(row < rows)
		{
			auto count = std::get<0>(res_parts).block_count_row(row);
			auto handles =
				std::make_tuple(
					std::get<RI>(res_parts).handle_for_row(row)...,
					arg_ref.handle_for_row(row),
					matrix_task::container_handle(
						std::get<CI>(containers),
						std::get<CI>(proxy_tags),
						row)...);

			matrix_task::schedule(
				handles,
				m_overlap,
				base::m_edge,
				base::m_pad,
				count,
				cols,
				std::forward<Args>(args)...);

			row += count;
		}
	}

	template<
		typename T,
		typename Vector>
	auto
	fill(
		util::border_region<T> & start,
		util::border_region<T> & end,
		Vector & v,
		size_t pos,
		size_t task_count) noexcept
	-> void
	{
		size_t ol_start(pos -m_overlap);
		if(ol_start >= v.size())
			ol_start += v.size();

		size_t i(0);
		while(i < m_overlap)
		{
			auto block_i = ol_start +i;
			if(block_i >= v.size())
				block_i -= v.size();

			auto count = std::min(m_overlap -i, v.block_count_from(block_i));
			auto b_handle = start.handle();
			auto v_offset = v.part_offset(block_i);
			auto v_handle = v.handle_for(block_i);

			starpu_mpi_task_insert(
				MPI_COMM_WORLD,
				&copy_cl,
				STARPU_RW|STARPU_SSEND, b_handle,
				STARPU_R, v_handle,
				STARPU_VALUE, &i, sizeof(i),
				STARPU_VALUE, &v_offset, sizeof(v_offset),
				STARPU_VALUE, &count, sizeof(count),
				STARPU_EXECUTE_ON_DATA, v_handle,
				0);

			i += count;
		}

		ol_start = pos + task_count;

		i = 0;
		while(i < m_overlap)
		{
			auto block_i = ol_start +i;
			if(block_i >= v.size())
				block_i -= v.size();

			auto count = std::min(m_overlap -i, v.block_count_from(block_i));
			auto b_handle = end.handle();
			auto v_offset = v.part_offset(block_i);
			auto v_handle = v.handle_for(block_i);

			starpu_mpi_task_insert(
				MPI_COMM_WORLD,
				&copy_cl,
				STARPU_RW|STARPU_SSEND, b_handle,
				STARPU_R, v_handle,
				STARPU_VALUE, &i, sizeof(i),
				STARPU_VALUE, &v_offset, sizeof(v_offset),
				STARPU_VALUE, &count, sizeof(count),
				STARPU_EXECUTE_ON_DATA, v_handle,
				0);

			i += count;
		}
	}
}; // class MapOverlap1D

} // namespace backend
} // namespace skepu

#endif // SKEPU_CLUSTER_MAPOVERLAP_1D_HPP
