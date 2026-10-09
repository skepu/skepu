#pragma once

// MapOverlap on distributed containers: a process computes its part of the output from its part of the input and the
// rows of the input that are next to it (the halo), which it gets from the processes that have them.
//
// The containers are distributed by blocks of the first dimension. For its output rows a process builds a block of the input
// (the "extended" block) that has everything that the node-level skeleton needs for them: the halo rows from other
// processes, and the elements beyond the edges of the whole input, as the edge mode of the skeleton says. The node-level
// skeleton then works on the extended block with the edge mode None, and its output is exactly the rows of the process.

#include <array>
#include <vector>

#include "skepu3/mpi/rows.hpp"
#include "skepu3/mpi/skeletons.hpp"
#include "skepu3/mpi/matrix.hpp"
#include "skepu3/mpi/tensor.hpp"

namespace skepu
{
	namespace backend
	{
		namespace mpi_detail
		{

			// The rows of the input (of n rows) that a process needs for its output rows [first, last): the ones that exist, as ranges
			// in order. (The ones beyond the edges are made from them, or from the pad value.)
			inline std::vector<RowRange> needed_rows(Edge edge, size_t n, long overlap, size_t first, size_t last)
			{
				std::vector<RowRange> ranges;
				if (last <= first)
					return ranges;

				long from, to;
				if (edge == Edge::None)
				{
					from = (long)first;
					to = (long)last + 2 * overlap;
				}
				else
				{
					from = (long)first - overlap;
					to = (long)last + overlap;
				}

				if (edge == Edge::Cyclic)
				{
					// The wrapped rows: the range cut at the multiples of n
					const long N = (long)n;
					for (long k = (from >= 0 ? from / N : -((-from + N - 1) / N)); k * N < to; ++k)
					{
						const long a = std::max(from, k * N), b = std::min(to, (k + 1) * N);
						if (a < b)
							ranges.push_back({(size_t)(a - k * N), (size_t)(b - k * N)});
					}
					std::sort(ranges.begin(), ranges.end());
					// merge
					std::vector<RowRange> merged;
					for (const RowRange &r : ranges)
					{
						if (!merged.empty() && r.first <= merged.back().second)
							merged.back().second = std::max(merged.back().second, r.second);
						else
							merged.push_back(r);
					}
					return merged;
				}

				const long a = std::max(from, 0L), b = std::min(to, (long)n);
				if (a < b)
					ranges.push_back({(size_t)a, (size_t)b});
				return ranges;
			}

			// The rows of the input that a process needs for its output rows [first, last) of a pool: the windows start every
			// 'stride' rows, and are 'size' rows
			inline std::vector<RowRange> needed_rows_pool(long size, long stride, size_t first, size_t last)
			{
				std::vector<RowRange> ranges;
				if (last > first)
					ranges.push_back({first * (size_t)stride, (last - 1) * (size_t)stride + (size_t)size});
				return ranges;
			}


			// The element of the input that is at g, which may be beyond an edge: the index of the element that gives it, or none
			inline bool source_coordinate(long g, size_t n, Edge edge, size_t &source)
			{
				if (g >= 0 && g < (long)n)
				{
					source = (size_t)g;
					return true;
				}
				switch (edge)
				{
				case Edge::Duplicate:
					source = g < 0 ? 0 : n - 1;
					return true;
				case Edge::Cyclic:
					source = (size_t)(((g % (long)n) + (long)n) % (long)n);
					return true;
				default: // Pad (and None, which has no elements beyond the edges)
					return false;
				}
			}

			// The block of the input that the node-level skeleton needs for some output rows, as a node-level container: ext_d0 rows,
			// the first of which is the row row_shift of the input (a row beyond the edges is made with the edge mode), and in
			// the other dimensions col_pad elements more on each side than the input (made with the edge mode).
			template<typename Container, typename T>
			typename container_traits<Container>::Node extended_block(const Shape &in, size_t ext_d0, long row_shift, const std::array<long, 4> &col_pad,
				Edge edge, T pad, const std::vector<RowRange> &ranges, const std::vector<T> &rows)
			{
				Shape ext = in;
				ext.d[0] = ext_d0;
				for (int k = 1; k < in.rank; ++k)
					ext.d[k] = in.d[k] + 2 * col_pad[k];

				auto block = container_traits<Container>::make(ext);
				T *out = block.getAddress();

				// Where the rows of the ranges are in the buffer
				std::vector<size_t> range_offset(ranges.size() + 1, 0);
				for (size_t i = 0; i < ranges.size(); ++i)
					range_offset[i + 1] = range_offset[i] + (ranges[i].second - ranges[i].first);
				auto row_position = [&](size_t g) -> size_t
				{
					size_t i = 0;
					while (i + 1 < ranges.size() && g >= ranges[i].second)
						++i;
					return range_offset[i] + (g - ranges[i].first);
				};

				size_t unit = 1;
				for (int k = 1; k < in.rank; ++k)
					unit *= in.d[k];

				size_t at = 0;
				for (size_t x0 = 0; x0 < ext.d[0]; ++x0)
				{
					const long g0 = row_shift + (long)x0;
					size_t s0 = 0;
					const bool row_exists = source_coordinate(g0, in.d[0], edge, s0);
					const size_t base = row_exists ? row_position(s0) * unit : 0;
					for (size_t x1 = 0; x1 < ext.d[1]; ++x1)
						for (size_t x2 = 0; x2 < ext.d[2]; ++x2)
							for (size_t x3 = 0; x3 < ext.d[3]; ++x3, ++at)
							{
								const size_t x[4] = {x0, x1, x2, x3};
								size_t source[4] = {s0, 0, 0, 0};
								bool exists = row_exists;
								for (int k = 1; k < in.rank && exists; ++k)
								{
									const long g = (long)x[k] - col_pad[k];
									exists = source_coordinate(g, in.d[k], edge, source[k]);
								}
								if (!exists)
								{
									out[at] = pad;
									continue;
								}
								size_t within = 0;
								for (int k = 1; k < in.rank; ++k)
									within = within * in.d[k] + source[k];
								out[at] = rows[base + within];
							}
				}
				block.invalidateDeviceData();
				return block;
			}


			// The argument of the node-level skeleton for an argument of the call: the tile of a result, the extended block for
			// the input, the whole container for a random access one
			template<size_t I, size_t outArity, size_t anyArity, typename Block, typename Arg>
			decltype(auto) overlap_argument(Block &block, Arg &&arg)
			{
				if constexpr (is_distributed_container<Arg>::value)
				{
					if constexpr (I < outArity)
						return (arg.tile_for_write());
					else if constexpr (I == outArity)
						return (block);
					else
					{
						static_assert(I < outArity + 1 + anyArity, "A distributed container is not a valid uniform argument");
						return arg.replicated_view();
					}
				}
				else
					return std::forward<Arg>(arg);
			}

			template<size_t outArity, size_t anyArity, typename Block, typename Call, size_t... I, typename... CallArgs>
			void call_overlap(Call &call, Block &block, std::index_sequence<I...>, CallArgs&&... args)
			{
				std::tuple<decltype(overlap_argument<I, outArity, anyArity>(block, get<I>(std::forward<CallArgs>(args)...)))...> local(
					overlap_argument<I, outArity, anyArity>(block, get<I>(std::forward<CallArgs>(args)...))...);
				std::apply(call, local);
			}

			// A call on the tiles of the containers (what a skeleton does when the processes do not need data of each other)
			template<size_t outArity, size_t anyArity, typename Call, size_t... I, typename... CallArgs>
			void call_on_tiles(Call &call, std::index_sequence<I...>, CallArgs&&... args)
			{
				std::tuple<decltype(local_argument<I, outArity, 1, anyArity>(get<I>(std::forward<CallArgs>(args)...)))...> local(
					local_argument<I, outArity, 1, anyArity>(get<I>(std::forward<CallArgs>(args)...))...);
				std::apply(call, local);
			}

			template<size_t outArity, size_t... I, typename... CallArgs>
			void mark_results(std::index_sequence<I...>, CallArgs&&... args)
			{
				pack_expand((get<I>(std::forward<CallArgs>(args)...).tile_written(), 0)...);
			}

			// What a window of the user function is: an overlap (a region around an element, with an edge mode) or a pool
			// (a window of the size of the pool that starts every stride elements, with no edges)
			struct WindowGeometry
			{
				bool pool = false;
				std::array<long, 4> overlap{{0, 0, 0, 0}}; // the overlap, or the size of the pool
				std::array<long, 4> stride{{1, 1, 1, 1}};
				Edge edge = Edge::None;
				// The 2D skeleton with the edge mode None and an input of the size of the output computes the interior of the output
				// only ("mirror mode"; the elements of the output near the edges keep their values)
				bool allow_mirror = false;
			};

			// The argument of the node-level skeleton in mirror mode: the blocks for the results, the block for the input
			template<size_t I, size_t outArity, size_t anyArity, typename Outs, typename Block, typename Arg>
			decltype(auto) mirror_argument(Outs &outs, Block &block, Arg &&arg)
			{
				if constexpr (is_distributed_container<Arg>::value)
				{
					if constexpr (I < outArity)
						return (std::get<I>(outs));
					else if constexpr (I == outArity)
						return (block);
					else
					{
						static_assert(I < outArity + 1 + anyArity, "A distributed container is not a valid uniform argument");
						return arg.replicated_view();
					}
				}
				else
					return std::forward<Arg>(arg);
			}

			template<size_t outArity, size_t anyArity, typename Outs, typename Block, typename Call, size_t... I, typename... CallArgs>
			void call_mirror(Call &call, Outs &outs, Block &block, std::index_sequence<I...>, CallArgs&&... args)
			{
				std::tuple<decltype(mirror_argument<I, outArity, anyArity>(outs, block, get<I>(std::forward<CallArgs>(args)...)))...> local(
					mirror_argument<I, outArity, anyArity>(outs, block, get<I>(std::forward<CallArgs>(args)...))...);
				std::apply(call, local);
			}

			// A block of a result for the mirror mode: ext_d0 rows of which the rows [skip, skip + rows) are the rows [from, from + rows)
			// of the tile (the other elements are not used)
			template<typename Container>
			typename container_traits<std::decay_t<Container>>::Node mirror_out_block(Container &c, const Shape &ext, size_t skip, size_t from, size_t rows)
			{
				auto block = container_traits<std::decay_t<Container>>::make(ext);
				auto &tile = c.tile_for_read();
				tile.updateHost();
				size_t unit = 1;
				for (int k = 1; k < ext.rank; ++k)
					unit *= ext.d[k];
				auto *source = tile.getAddress();
				auto *target = block.getAddress();
				std::copy(source + from * unit, source + (from + rows) * unit, target + skip * unit);
				return block;
			}

			template<size_t outArity, size_t... OI, typename... CallArgs>
			auto make_mirror_outs(std::index_sequence<OI...>, const RowRange &mine, size_t first, const Shape &in_shape, long o, CallArgs&&... args)
			{
				const size_t rows = mine.second > mine.first ? mine.second - mine.first : 0;
				Shape ext = in_shape;
				ext.d[0] = rows + 2 * (size_t)o;
				auto make = [&](auto &c)
				{
					return mirror_out_block(c, ext, (size_t)o, rows ? mine.first - first : 0, rows);
				};
				auto blocks = std::make_tuple(make(get<OI>(std::forward<CallArgs>(args)...))...);
				// (the index origin of the blocks: the node-level skeleton gives the rows of the block to the user function. Set after the
				// blocks are in the tuple, as copies of containers do not have it.)
				(std::get<OI>(blocks).setIndexOrigin(rows ? mine.first - (size_t)o : 0), ...);
				return blocks;
			}

			// A process that computes nothing keeps the rows of its tiles as they are
			template<size_t... OI, typename... CallArgs>
			void make_tiles_valid(std::index_sequence<OI...>, CallArgs&&... args)
			{
				auto touch = [](auto &c) { if (c.local_size() > 0) c.tile_for_read(); };
				(touch(get<OI>(std::forward<CallArgs>(args)...)), ...);
			}

			template<size_t... OI, typename Outs, typename... CallArgs>
			void copy_back_mirror(std::index_sequence<OI...>, Outs &outs, const RowRange &mine, size_t first, long o, CallArgs&&... args)
			{
				auto copy = [&](auto &c, auto &block)
				{
					auto &tile = c.tile_for_write();
					size_t unit = 1;
					const Shape s = shape_of(c);
					for (int k = 1; k < s.rank; ++k)
						unit *= s.d[k];
					block.updateHost(); // (the node-level skeleton may have run on a device)
					auto *source = block.getAddress();
					auto *target = tile.getAddress();
					std::copy(source + (size_t)o * unit, source + ((size_t)o + (mine.second - mine.first)) * unit, target + (mine.first - first) * unit);
					tile.invalidateDeviceData();
				};
				(copy(get<OI>(std::forward<CallArgs>(args)...), std::get<OI>(outs)), ...);
			}

			// The mirror mode of a MapOverlap: the process computes the rows [a, b) of the interior that it has (the rows [o, n - o) of
			// the output), from the rows [a - o, b + o) of the input, with the edge mode None and an output block that has the size of
			// the input block, so that the node-level skeleton is in mirror mode too. Then the rows of the block are copied back into the tile.
			template<size_t outArity, size_t anyArity, typename T, typename Call, typename... CallArgs>
			auto &distributed_mirror(Call call, const WindowGeometry &geometry, CallArgs&&... args)
			{
				auto &first_out = get<0>(std::forward<CallArgs>(args)...);
				auto &input = get<outArity>(std::forward<CallArgs>(args)...);
				using InputContainer = std::decay_t<decltype(input)>;
				const long o = geometry.overlap[0];
				const Shape in_shape = shape_of(input);
				const size_t n = in_shape.d[0];

				const int me = mpi::rank(), p = mpi::size();
				const mpi::BlockDistribution &out_distribution = first_out.distribution();
				const mpi::BlockDistribution &in_distribution = input.distribution();

				// The rows of the interior of each process
				auto interior = [&](int r) -> RowRange
				{
					const size_t first = out_distribution.offset(r), last = first + out_distribution.count(r);
					const size_t a = std::max<size_t>(first, (size_t)o), b = (n > (size_t)o) ? std::min<size_t>(last, n - (size_t)o) : 0;
					return a < b ? RowRange{a, b} : RowRange{0, 0};
				};

				std::vector<std::vector<RowRange>> needed(p);
				for (int r = 0; r < p; ++r)
				{
					const RowRange rows = interior(r);
					if (rows.first < rows.second)
						needed[r].push_back({rows.first - (size_t)o, rows.second + (size_t)o});
				}

				auto &tile = input.tile_for_read();
				const T *tile_data = nullptr;
				if (input.local_size() > 0)
				{
					tile.updateHost();
					tile_data = tile.getAddress();
				}
				const std::vector<T> rows = exchange_rows(tile_data, in_distribution.unit, in_distribution, needed, "halo exchange", {input.getObjectID()});

				const RowRange mine = interior(me);
				// (the results are made valid on all the processes that have rows, also when they compute none)
				if (mine.first < mine.second)
				{
					auto outs = make_mirror_outs<outArity>(std::make_index_sequence<outArity>{}, mine, out_distribution.offset(me), in_shape, o, std::forward<CallArgs>(args)...);
					const std::array<long, 4> no_pad{{0, 0, 0, 0}};
					const size_t count = mine.second - mine.first;
					auto block = extended_block<InputContainer>(in_shape, count + 2 * (size_t)o, (long)mine.first - o, no_pad, Edge::None, T(), needed[me], rows);
					call_mirror<outArity, anyArity>(call, outs, block, std::make_index_sequence<sizeof...(CallArgs)>{}, std::forward<CallArgs>(args)...);
					copy_back_mirror(std::make_index_sequence<outArity>{}, outs, mine, out_distribution.offset(me), o, std::forward<CallArgs>(args)...);
				}
				else
					make_tiles_valid(std::make_index_sequence<outArity>{}, std::forward<CallArgs>(args)...);

				mark_results<outArity>(std::make_index_sequence<outArity>{}, std::forward<CallArgs>(args)...);
				return first_out;
			}

			// The call of a skeleton with windows on distributed containers. 'call' calls the node-level skeleton with the arguments it
			// is given, 'set_edge' sets its edge mode. The arguments are the results, the input, the random access containers and the
			// uniforms.
			template<size_t outArity, size_t anyArity, typename T, typename Call, typename SetEdge, typename... CallArgs>
			auto &distributed_window(Call call, SetEdge set_edge, const WindowGeometry &geometry, T pad, CallArgs&&... args)
			{
				auto &first_out = get<0>(std::forward<CallArgs>(args)...);
				auto &input = get<outArity>(std::forward<CallArgs>(args)...);
				using InputContainer = std::decay_t<decltype(input)>;
				const std::array<long, 4> &overlap = geometry.overlap;
				const Edge edge = geometry.edge;

				const Shape out_shape = shape_of(first_out), in_shape = shape_of(input);
				if (in_shape.rank != out_shape.rank)
					SKEPU_ERROR("Non-matching container dimensions");
				bool mirror = geometry.allow_mirror && !geometry.pool && edge == Edge::None;
				for (int k = 0; k < in_shape.rank && mirror; ++k)
					mirror = in_shape.d[k] == out_shape.d[k];
				if (mirror)
					return distributed_mirror<outArity, anyArity, T>(call, geometry, std::forward<CallArgs>(args)...);

				for (int k = 0; k < in_shape.rank; ++k)
				{
					if (geometry.pool)
					{
						// (the sizes that the node-level skeleton wants: the windows must be in the input)
						const size_t needed_size = (out_shape.d[k] - 1) * (size_t)geometry.stride[k] + (size_t)overlap[k];
						if (in_shape.d[k] < needed_size)
							SKEPU_ERROR("MapPool on distributed containers: the input is too small for the output: dimension " << k
								<< ": input " << in_shape.d[k] << ", at least " << needed_size << " needed for the output " << out_shape.d[k]);
					}
					else
					{
						const bool expect_larger = (edge == Edge::None) && overlap[k] > 0;
						const size_t expected = out_shape.d[k] + (expect_larger ? 2 * overlap[k] : 0);
						if (in_shape.d[k] != expected)
							SKEPU_ERROR("MapOverlap on distributed containers: the input must have the size of the output (and, with the edge mode None, "
								"twice the overlap more in each dimension): dimension " << k << ": input " << in_shape.d[k] << ", output " << out_shape.d[k]);
					}
				}

				const int me = mpi::rank(), p = mpi::size();
				const mpi::BlockDistribution &out_distribution = first_out.distribution();
				const mpi::BlockDistribution &in_distribution = input.distribution();

				std::vector<std::vector<RowRange>> needed(p);
				for (int r = 0; r < p; ++r)
				{
					const size_t first = out_distribution.offset(r), last = first + out_distribution.count(r);
					needed[r] = geometry.pool ? needed_rows_pool(overlap[0], geometry.stride[0], first, last)
						: needed_rows(edge, in_shape.d[0], overlap[0], first, last);
				}

				// The rows that this process has of the input, to the others
				auto &tile = input.tile_for_read();
				const T *tile_data = nullptr;
				if (input.local_size() > 0)
				{
					tile.updateHost();
					tile_data = tile.getAddress();
				}
				const std::vector<T> rows = exchange_rows(tile_data, in_distribution.unit, in_distribution, needed, "halo exchange", {input.getObjectID()});

				const size_t count = out_distribution.count(me);
				if (count > 0)
				{
					const size_t first = out_distribution.offset(me);
					const std::array<long, 4> no_pad{{0, 0, 0, 0}};
					size_t ext_d0;
					long row_shift;
					if (geometry.pool)
					{
						ext_d0 = (count - 1) * (size_t)geometry.stride[0] + (size_t)overlap[0];
						row_shift = (long)(first * (size_t)geometry.stride[0]);
					}
					else
					{
						ext_d0 = count + 2 * overlap[0];
						row_shift = (edge == Edge::None) ? (long)first : (long)first - overlap[0];
					}
					auto block = extended_block<InputContainer>(in_shape, ext_d0, row_shift, (!geometry.pool && edge != Edge::None) ? overlap : no_pad,
						edge, pad, needed[me], rows);

					// The node-level skeleton works on the extended block without handling the edges
					if (!geometry.pool)
						set_edge(Edge::None);
					call_overlap<outArity, anyArity>(call, block, std::make_index_sequence<sizeof...(CallArgs)>{}, std::forward<CallArgs>(args)...);
					if (!geometry.pool)
						set_edge(edge);
				}

				mark_results<outArity>(std::make_index_sequence<outArity>{}, std::forward<CallArgs>(args)...);
				return first_out;
			}

			// An overlap
			template<size_t outArity, size_t anyArity, typename T, typename Call, typename SetEdge, typename... CallArgs>
			auto &distributed_overlap(Call call, SetEdge set_edge, const std::array<long, 4> &overlap, Edge edge, T pad, CallArgs&&... args)
			{
				WindowGeometry geometry;
				geometry.overlap = overlap;
				geometry.edge = edge;
				return distributed_window<outArity, anyArity>(call, set_edge, geometry, pad, std::forward<CallArgs>(args)...);
			}

			inline std::array<long, 4> overlap_array(int o) { return {{o, 0, 0, 0}}; }

			template<typename... Ints>
			std::array<long, 4> overlap_array(const std::tuple<Ints...> &o)
			{
				std::array<long, 4> a{{0, 0, 0, 0}};
				size_t k = 0;
				std::apply([&](auto... v) { ((a[k++] = (long)v), ...); }, o);
				return a;
			}
		}

		// ===================================================================================
		//   MapOverlap
		// ===================================================================================

		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		class MapOverlap1D : public ::skepu_node::backend::MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		{
			using Base = ::skepu_node::backend::MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setOverlap, Base)
			SKEPU_CHAIN(setOverlapMode, Base)
			SKEPU_CHAIN(setEdgeMode, Base)
			// (a pad that is a braced list is converted to the type of the pad)
			template<typename U = decltype(std::declval<const Base &>().getPad())>
			auto &setPad(U value)
			{
				Base::setPad(value);
				return *this;
			}
			SKEPU_CHAIN(setUpdateMode, Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
				{
					return this->distribute(std::forward<CallArgs>(args)...);
				}
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<typename... CallArgs>
			auto &distribute(CallArgs&&... args)
			{
				using Input = std::decay_t<decltype(get<outArity>(std::forward<CallArgs>(args)...))>;
				if constexpr (MapOverlapFunc::usesPRNG && is_distributed_matrix<Input>::value)
					SKEPU_ERROR("A MapOverlap 1D of a matrix with random numbers does not work on distributed containers yet");
				if constexpr (MapOverlapFunc::usesPRNG)
				{
					// (the elements of the output are used in the order of the memory; the node-level 1D overlap of a
					// vector takes the numbers for the size of its input)
					auto &first_out = get<0>(std::forward<CallArgs>(args)...);
					auto &input = get<outArity>(std::forward<CallArgs>(args)...);
					mpi_detail::PrngShift shift(this->m_prng, first_out.local_offset() * MapOverlapFunc::randomCount, input.size() * MapOverlapFunc::randomCount);
					return this->run_distributed(std::forward<CallArgs>(args)...);
				}
				else
					return this->run_distributed(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_distributed(CallArgs&&... args)
			{
				using Input = std::decay_t<decltype(get<outArity>(std::forward<CallArgs>(args)...))>;
				if constexpr (is_distributed_matrix<Input>::value)
				{
					// The overlap of a matrix: along its rows (row-wise) or its columns (column-wise). Along the rows the processes
					// do not need each other, as every process has whole rows. Along the columns it is the halo of the rows, as for
					// a vector.
					if (this->getOverlapMode() == Overlap::RowWise)
						return this->run_row_wise(std::forward<CallArgs>(args)...);
				}
				return mpi_detail::distributed_overlap<outArity, anyArity>(
					[this](auto &... a) { Base::operator()(a...); }, [this](Edge e) { Base::setEdgeMode(e); },
					mpi_detail::overlap_array(this->getOverlap()), this->getEdgeMode(), this->getPad(), std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_row_wise(CallArgs&&... args)
			{
				auto &first_out = get<0>(std::forward<CallArgs>(args)...);
				auto &input = get<outArity>(std::forward<CallArgs>(args)...);
				if (first_out.total_rows() != input.total_rows())
					SKEPU_ERROR("Non-matching container sizes: the rows of the output and of the input");
				if (first_out.local_size() > 0)
				{
					auto call = [this](auto &... a) { Base::operator()(a...); };
					mpi_detail::call_on_tiles<outArity, anyArity>(call, std::make_index_sequence<sizeof...(CallArgs)>{}, std::forward<CallArgs>(args)...);
				}
				mpi_detail::mark_results<outArity>(std::make_index_sequence<outArity>{}, std::forward<CallArgs>(args)...);
				return first_out;
			}
		};

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapOverlap2D : public ::skepu_node::backend::MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setOverlap, Base)
			SKEPU_CHAIN(setEdgeMode, Base)
			// (a pad that is a braced list is converted to the type of the pad)
			template<typename U = decltype(std::declval<const Base &>().getPad())>
			auto &setPad(U value)
			{
				Base::setPad(value);
				return *this;
			}
			SKEPU_CHAIN(setUpdateMode, Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
				{
					if (this->getUpdateMode() != UpdateMode::Normal)
						SKEPU_ERROR("A MapOverlap with an update mode other than Normal does not work on distributed containers yet");
					return this->distribute(std::forward<CallArgs>(args)...);
				}
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<typename... CallArgs>
			auto &distribute(CallArgs&&... args)
			{
				if constexpr (MapOverlapFunc::usesPRNG)
				{
					// (the elements of the output are used in the order of the memory)
					auto &first_out = get<0>(std::forward<CallArgs>(args)...);
					mpi_detail::PrngShift shift(this->m_prng, first_out.local_offset() * MapOverlapFunc::randomCount, first_out.size() * MapOverlapFunc::randomCount);
					return this->run_distributed(std::forward<CallArgs>(args)...);
				}
				else
					return this->run_distributed(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_distributed(CallArgs&&... args)
			{
				mpi_detail::WindowGeometry geometry;
				geometry.overlap = mpi_detail::overlap_array(this->getOverlap());
				geometry.edge = this->getEdgeMode();
				geometry.allow_mirror = true;
				return mpi_detail::distributed_window<outArity, anyArity>(
					[this](auto &... a) { Base::operator()(a...); }, [this](Edge e) { Base::setEdgeMode(e); },
					geometry, this->getPad(), std::forward<CallArgs>(args)...);
			}
		};

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapOverlap3D : public ::skepu_node::backend::MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapOverlap3D<MapOverlapFunc, CUDAKernel, CLKernel>;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setOverlap, Base)
			SKEPU_CHAIN(setEdgeMode, Base)
			// (a pad that is a braced list is converted to the type of the pad)
			template<typename U = decltype(std::declval<const Base &>().getPad())>
			auto &setPad(U value)
			{
				Base::setPad(value);
				return *this;
			}
			SKEPU_CHAIN(setUpdateMode, Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
				{
					return this->distribute(std::forward<CallArgs>(args)...);
				}
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<typename... CallArgs>
			auto &distribute(CallArgs&&... args)
			{
				if constexpr (MapOverlapFunc::usesPRNG)
				{
					// (the elements of the output are used in the order of the memory)
					auto &first_out = get<0>(std::forward<CallArgs>(args)...);
					mpi_detail::PrngShift shift(this->m_prng, first_out.local_offset() * MapOverlapFunc::randomCount, first_out.size() * MapOverlapFunc::randomCount);
					return this->run_distributed(std::forward<CallArgs>(args)...);
				}
				else
					return this->run_distributed(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_distributed(CallArgs&&... args)
			{
				return mpi_detail::distributed_overlap<outArity, anyArity>(
					[this](auto &... a) { Base::operator()(a...); }, [this](Edge e) { Base::setEdgeMode(e); },
					mpi_detail::overlap_array(this->getOverlap()), this->getEdgeMode(), this->getPad(), std::forward<CallArgs>(args)...);
			}
		};

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapOverlap4D : public ::skepu_node::backend::MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapOverlap4D<MapOverlapFunc, CUDAKernel, CLKernel>;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setOverlap, Base)
			SKEPU_CHAIN(setEdgeMode, Base)
			// (a pad that is a braced list is converted to the type of the pad)
			template<typename U = decltype(std::declval<const Base &>().getPad())>
			auto &setPad(U value)
			{
				Base::setPad(value);
				return *this;
			}
			SKEPU_CHAIN(setUpdateMode, Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
				{
					return this->distribute(std::forward<CallArgs>(args)...);
				}
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<typename... CallArgs>
			auto &distribute(CallArgs&&... args)
			{
				if constexpr (MapOverlapFunc::usesPRNG)
				{
					// (the elements of the output are used in the order of the memory)
					auto &first_out = get<0>(std::forward<CallArgs>(args)...);
					mpi_detail::PrngShift shift(this->m_prng, first_out.local_offset() * MapOverlapFunc::randomCount, first_out.size() * MapOverlapFunc::randomCount);
					return this->run_distributed(std::forward<CallArgs>(args)...);
				}
				else
					return this->run_distributed(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_distributed(CallArgs&&... args)
			{
				return mpi_detail::distributed_overlap<outArity, anyArity>(
					[this](auto &... a) { Base::operator()(a...); }, [this](Edge e) { Base::setEdgeMode(e); },
					mpi_detail::overlap_array(this->getOverlap()), this->getEdgeMode(), this->getPad(), std::forward<CallArgs>(args)...);
			}
		};

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapPool2D : public ::skepu_node::backend::MapPool2D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapPool2D<MapOverlapFunc, CUDAKernel, CLKernel>;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setPoolSize, Base)
			SKEPU_CHAIN(setPoolSizeAndStride, Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
					return this->distribute(std::forward<CallArgs>(args)...);
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<typename... CallArgs>
			auto &distribute(CallArgs&&... args)
			{
				if constexpr (MapOverlapFunc::usesPRNG)
				{
					// (the elements of the output are used in the order of the memory)
					auto &first_out = get<0>(std::forward<CallArgs>(args)...);
					mpi_detail::PrngShift shift(this->m_prng, first_out.local_offset() * MapOverlapFunc::randomCount, first_out.size() * MapOverlapFunc::randomCount);
					return this->run_distributed(std::forward<CallArgs>(args)...);
				}
				else
					return this->run_distributed(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_distributed(CallArgs&&... args)
			{
				using Element = typename std::decay_t<decltype(get<outArity>(std::forward<CallArgs>(args)...))>::value_type;
				mpi_detail::WindowGeometry geometry;
				geometry.pool = true;
				geometry.overlap = mpi_detail::overlap_array(this->getPoolSize());
				geometry.stride = {1, 1, 1, 1};
				const std::array<long, 4> stride = mpi_detail::overlap_array(this->getStride());
				for (int k = 0; k < 2; ++k)
					geometry.stride[k] = stride[k];
				return mpi_detail::distributed_window<outArity, anyArity>(
					[this](auto &... a) { Base::operator()(a...); }, [](Edge) {}, geometry, Element{}, std::forward<CallArgs>(args)...);
			}
		};

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapPool3D : public ::skepu_node::backend::MapPool3D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapPool3D<MapOverlapFunc, CUDAKernel, CLKernel>;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setPoolSize, Base)
			SKEPU_CHAIN(setPoolSizeAndStride, Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
					return this->distribute(std::forward<CallArgs>(args)...);
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<typename... CallArgs>
			auto &distribute(CallArgs&&... args)
			{
				if constexpr (MapOverlapFunc::usesPRNG)
				{
					// (the elements of the output are used in the order of the memory)
					auto &first_out = get<0>(std::forward<CallArgs>(args)...);
					mpi_detail::PrngShift shift(this->m_prng, first_out.local_offset() * MapOverlapFunc::randomCount, first_out.size() * MapOverlapFunc::randomCount);
					return this->run_distributed(std::forward<CallArgs>(args)...);
				}
				else
					return this->run_distributed(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_distributed(CallArgs&&... args)
			{
				using Element = typename std::decay_t<decltype(get<outArity>(std::forward<CallArgs>(args)...))>::value_type;
				mpi_detail::WindowGeometry geometry;
				geometry.pool = true;
				geometry.overlap = mpi_detail::overlap_array(this->getPoolSize());
				geometry.stride = {1, 1, 1, 1};
				const std::array<long, 4> stride = mpi_detail::overlap_array(this->getStride());
				for (int k = 0; k < 3; ++k)
					geometry.stride[k] = stride[k];
				return mpi_detail::distributed_window<outArity, anyArity>(
					[this](auto &... a) { Base::operator()(a...); }, [](Edge) {}, geometry, Element{}, std::forward<CallArgs>(args)...);
			}
		};

		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		class MapPool4D : public ::skepu_node::backend::MapPool4D<MapOverlapFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapPool4D<MapOverlapFunc, CUDAKernel, CLKernel>;
			static constexpr size_t outArity = MapOverlapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapOverlapFunc::ContainerArgs>::value;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setPoolSize, Base)
			SKEPU_CHAIN(setPoolSizeAndStride, Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
					return this->distribute(std::forward<CallArgs>(args)...);
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<typename... CallArgs>
			auto &distribute(CallArgs&&... args)
			{
				if constexpr (MapOverlapFunc::usesPRNG)
				{
					// (the elements of the output are used in the order of the memory)
					auto &first_out = get<0>(std::forward<CallArgs>(args)...);
					mpi_detail::PrngShift shift(this->m_prng, first_out.local_offset() * MapOverlapFunc::randomCount, first_out.size() * MapOverlapFunc::randomCount);
					return this->run_distributed(std::forward<CallArgs>(args)...);
				}
				else
					return this->run_distributed(std::forward<CallArgs>(args)...);
			}

			template<typename... CallArgs>
			auto &run_distributed(CallArgs&&... args)
			{
				using Element = typename std::decay_t<decltype(get<outArity>(std::forward<CallArgs>(args)...))>::value_type;
				mpi_detail::WindowGeometry geometry;
				geometry.pool = true;
				geometry.overlap = mpi_detail::overlap_array(this->getPoolSize());
				geometry.stride = {1, 1, 1, 1};
				const std::array<long, 4> stride = mpi_detail::overlap_array(this->getStride());
				for (int k = 0; k < 4; ++k)
					geometry.stride[k] = stride[k];
				return mpi_detail::distributed_window<outArity, anyArity>(
					[this](auto &... a) { Base::operator()(a...); }, [](Edge) {}, geometry, Element{}, std::forward<CallArgs>(args)...);
			}
		};
	}
}
