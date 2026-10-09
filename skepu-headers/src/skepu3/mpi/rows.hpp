#pragma once

// Row layouts. The first dimension of the distributed containers is distributed in blocks, and a call computes the elements
// (the indices of the first dimension) of each process. Most calls compute those that the results are distributed as, and the
// element-wise containers are in the same blocks. A call with a MatRow argument, though, needs for element i the row i of a
// matrix, whose blocks need not be those of the results (the matrix may have more rows than the results have elements).
// Then the call is made with another layout of the elements, that is the cheapest in the data that has to move: the blocks
// of the results (the rows of the matrix go to the process that has the element) or the blocks of the matrix (the results and
// the element-wise containers go to the process that has the row; the results go back afterwards). With the same sizes
// (the usual case) nothing moves, and a process has just the rows of its elements.

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "skepu3/mpi/vector.hpp"
#include "skepu3/mpi/matrix.hpp"
#include "skepu3/mpi/tensor.hpp"

namespace skepu
{
	namespace backend
	{
		namespace mpi_detail
		{
			struct Shape
			{
				std::array<size_t, 4> d{{1, 1, 1, 1}};
				int rank = 1;
			};

			// What a distributed container is, as a node-level container of the same kind
			template<typename C>
			struct container_traits;

			template<typename T>
			struct container_traits<skepu::Vector<T>>
			{
				using Node = ::skepu_node::Vector<T>;
				static Shape shape(const skepu::Vector<T> &c) { return {{{c.size(), 1, 1, 1}}, 1}; }
				static Node make(const Shape &s) { return Node(s.d[0]); }
			};

			template<typename T>
			struct container_traits<skepu::Matrix<T>>
			{
				using Node = ::skepu_node::Matrix<T>;
				static Shape shape(const skepu::Matrix<T> &c) { return {{{c.total_rows(), c.total_cols(), 1, 1}}, 2}; }
				static Node make(const Shape &s) { return Node(s.d[0], s.d[1]); }
			};

			template<typename T>
			struct container_traits<skepu::Tensor3<T>>
			{
				using Node = ::skepu_node::Tensor3<T>;
				static Shape shape(const skepu::Tensor3<T> &c) { return {{{c.size_i(), c.size_j(), c.size_k(), 1}}, 3}; }
				static Node make(const Shape &s) { return Node(s.d[0], s.d[1], s.d[2]); }
			};

			template<typename T>
			struct container_traits<skepu::Tensor4<T>>
			{
				using Node = ::skepu_node::Tensor4<T>;
				static Shape shape(const skepu::Tensor4<T> &c) { return {{{c.size_i(), c.size_j(), c.size_k(), c.size_l()}}, 4}; }
				static Node make(const Shape &s) { return Node(s.d[0], s.d[1], s.d[2], s.d[3]); }
			};

			using RowRange = std::pair<size_t, size_t>; // [first, last) rows of the input, in the first dimension

			// Every process gets the rows that it needs (needed[r] for the process r), from the processes that have them. The
			// result is the rows of the ranges of this process, one after the other, in the order of the ranges.
			// ('distribution' says which rows each process has in 'tile': offset(r) and count(r))
			// ('reason' and 'ids' say why, for the trace)
			template<typename T, typename Distribution>
			std::vector<T> exchange_rows(const T *tile, size_t unit, const Distribution &distribution,
				const std::vector<std::vector<RowRange>> &needed, const char *reason = "row exchange", std::vector<int> ids = {})
			{
				const int me = mpi::rank(), p = mpi::size();
				const size_t my_first = distribution.offset(me), my_last = my_first + distribution.count(me);

				auto intersection = [](const RowRange &range, size_t first, size_t last) -> RowRange
				{
					const size_t a = std::max(range.first, first), b = std::min(range.second, last);
					return a < b ? RowRange{a, b} : RowRange{0, 0};
				};

				// What this process sends to each process: the rows of its part that the other needs
				std::vector<int> send_counts(p, 0), send_displs(p, 0), receive_counts(p, 0), receive_displs(p, 0);
				std::vector<T> send;
				for (int r = 0; r < p; ++r)
				{
					send_displs[r] = mpi::checked_count(send.size(), "halo exchange");
					for (const RowRange &range : needed[r])
					{
						const RowRange piece = intersection(range, my_first, my_last);
						if (piece.first < piece.second)
							send.insert(send.end(), tile + (piece.first - my_first) * unit, tile + (piece.second - my_first) * unit);
					}
					send_counts[r] = mpi::checked_count(send.size() - send_displs[r], "halo exchange");
				}

				// What it receives from each: its ranges, cut at the parts of the processes
				size_t total_rows = 0;
				for (const RowRange &range : needed[me])
					total_rows += range.second - range.first;
				int total = 0;
				for (int s = 0; s < p; ++s)
				{
					const size_t s_first = distribution.offset(s), s_last = s_first + distribution.count(s);
					size_t rows = 0;
					for (const RowRange &range : needed[me])
					{
						const RowRange piece = intersection(range, s_first, s_last);
						rows += piece.second - piece.first;
					}
					receive_counts[s] = mpi::checked_count(rows * unit, "halo exchange");
					receive_displs[s] = total;
					total += receive_counts[s];
				}

				std::vector<T> received(total > 0 ? total : 1);
				{
					mpi::trace::CommScope comm("alltoallv", reason, std::move(ids));
					unsigned long long sent = 0, got = 0;
					for (int r = 0; r < p; ++r)
						if (r != me)
						{
							sent += (unsigned long long)send_counts[r];
							got += (unsigned long long)receive_counts[r];
						}
					comm.bytes(sent * sizeof(T), got * sizeof(T));
					MPI_Alltoallv(send.empty() ? nullptr : send.data(), send_counts.data(), send_displs.data(), mpi::datatype<T>(),
						received.data(), receive_counts.data(), receive_displs.data(), mpi::datatype<T>(), mpi::comm());
				}

				// The rows in the order of the ranges. (From each process they come in the order of the ranges.)
				std::vector<T> rows(total_rows * unit > 0 ? total_rows * unit : 1);
				std::vector<size_t> cursor(p, 0);
				size_t range_start = 0;
				for (const RowRange &range : needed[me])
				{
					for (int s = 0; s < p; ++s)
					{
						const size_t s_first = distribution.offset(s), s_last = s_first + distribution.count(s);
						const RowRange piece = intersection(range, s_first, s_last);
						if (piece.first >= piece.second)
							continue;
						const size_t count = (piece.second - piece.first) * unit;
						std::copy(received.begin() + receive_displs[s] + cursor[s], received.begin() + receive_displs[s] + cursor[s] + count,
							rows.begin() + (range_start + (piece.first - range.first)) * unit);
						cursor[s] += count;
					}
					range_start += range.second - range.first;
				}
				return rows;
			}

			// The shape of a distributed container as the check for the sizes of a MapOverlap needs it
			template<typename Container>
			Shape shape_of(const Container &c) { return container_traits<Container>::shape(c); }

			// The elements (rows of the first dimension) that each process computes: [first, last) for every process, in order
			struct RowLayout
			{
				std::vector<RowRange> ranges;

				size_t offset(int r) const { return ranges[r].first; }
				size_t count(int r) const { return ranges[r].second - ranges[r].first; }
				RowRange mine() const { return ranges[mpi::rank()]; }
				size_t my_count() const { return this->count(mpi::rank()); }

				// A container whose blocks are exactly the elements of the layout
				bool exact(const mpi::BlockDistribution &d) const
				{
					for (int r = 0; r < (int)ranges.size(); ++r)
						if (d.offset(r) != this->offset(r) || d.count(r) != this->count(r))
							return false;
					return true;
				}

				// A matrix that has the row of every element of the layout (and possibly more rows after them), in the
				// process of the element, starting at the same row
				bool covers_start(const mpi::BlockDistribution &d) const
				{
					for (int r = 0; r < (int)ranges.size(); ++r)
						if (this->count(r) > 0 && (d.offset(r) != this->offset(r) || d.count(r) < this->count(r)))
							return false;
					return true;
				}
			};

			// The blocks of a distribution, cut at the first 'clip' indices
			inline RowLayout layout_of(const mpi::BlockDistribution &d, size_t clip)
			{
				RowLayout layout;
				for (int r = 0; r < d.p; ++r)
				{
					const size_t a = std::min(d.offset(r), clip), b = std::min(d.offset(r) + d.count(r), clip);
					layout.ranges.push_back({a, b});
				}
				return layout;
			}

			// What an argument of a call is, for the layout
			enum class Role
			{
				Result,      // written; the elements of the layout (the container has to be able to take the rows)
				ResultKeep,  // as Result, but its values are used (an in place call)
				Elwise,      // element i is used for element i
				MatRow,      // row i of the matrix is used for element i
				Other        // not by element: needed in its entirety (or not distributed)
			};

			// The role of the random access argument I (they start at first_any) with the proxy tags of the user function
			template<size_t I, size_t first_any, typename Tags>
			constexpr Role any_role()
			{
				if constexpr (I >= first_any && (I - first_any) < std::tuple_size<Tags>::value)
					return std::is_same<std::tuple_element_t<I - first_any, Tags>, skepu::ProxyTag::MatRow>::value ? Role::MatRow : Role::Other;
				else
					return Role::Other;
			}

			struct PartInfo
			{
				const mpi::BlockDistribution *distribution;
				double bytes_per_row;
				Role role;
			};

			template<typename Arg>
			void add_part(std::vector<PartInfo> &parts, Role role, Arg &&arg)
			{
				using C = std::decay_t<Arg>;
				if constexpr (is_distributed_container<C>::value)
				{
					if (role != Role::Other)
					{
						const mpi::BlockDistribution &d = arg.distribution();
						parts.push_back({&d, (double)(d.unit * sizeof(typename C::value_type)), role});
					}
				}
			}

			// The layout for a call of n elements that moves the least data. The candidates are the blocks of n elements
			// (the results' own) and, if matrices_may_lead, the blocks of each MatRow matrix.
			inline RowLayout choose_layout(size_t n, const std::vector<PartInfo> &parts, bool matrices_may_lead)
			{
				std::vector<RowLayout> candidates;
				candidates.push_back(layout_of(mpi::BlockDistribution{n, mpi::size(), 1}, n));
				if (matrices_may_lead)
					for (const PartInfo &part : parts)
						if (part.role == Role::MatRow && part.distribution->n >= n)
							candidates.push_back(layout_of(*part.distribution, n));

				auto cost = [&](const RowLayout &layout)
				{
					double bytes = 0;
					for (const PartInfo &part : parts)
					{
						const bool in_place = part.role == Role::MatRow ? layout.covers_start(*part.distribution) : layout.exact(*part.distribution);
						if (!in_place)
							bytes += (double)n * part.bytes_per_row;
					}
					return bytes;
				};

				size_t best = 0;
				double best_cost = cost(candidates[0]);
				for (size_t i = 1; i < candidates.size(); ++i)
				{
					const double c = cost(candidates[i]);
					if (c < best_cost)
					{
						best = i;
						best_cost = c;
					}
				}
				// (SKEPU_MPI_TRACE_LAYOUT=1: which layout a call has, and the bytes that it moves)
				if (std::getenv("SKEPU_MPI_TRACE_LAYOUT") && mpi::rank() == 0)
					std::cerr << "[SkePU MPI] " << n << " elements: " << (best == 0 ? "the blocks of the results" : "the blocks of a matrix")
						<< ", " << best_cost << " bytes move\n";
				return candidates[best];
			}

			// A node-level container that a call made with a layout has made for an argument that is not in the blocks of the layout
			template<typename Arg, bool = is_distributed_container<Arg>::value>
			struct RowHolder
			{
			};

			template<typename Arg>
			struct RowHolder<Arg, true>
			{
				std::unique_ptr<typename std::decay_t<Arg>::tile_type> temp;
			};

			// The argument of the node-level skeleton for an argument of a call with a layout: the tile of the container if it
			// has the right rows, else a node-level container with them (fetched from the processes that have them for what is
			// read; empty for a result: see finish_argument). A process with no elements is not run: it gets its own tile.
			template<Role R, typename Arg, typename Hold>
			decltype(auto) exec_argument(const RowLayout &layout, Hold &hold, Arg &&arg)
			{
				using C = std::decay_t<Arg>;
				if constexpr (!is_distributed_container<C>::value)
					return std::forward<Arg>(arg);
				else if constexpr (R == Role::Other)
					return arg.replicated_view();
				else
				{
					using Node = typename C::tile_type;
					using T = typename C::value_type;
					const mpi::BlockDistribution &d = arg.distribution();
					const bool direct = (R == Role::MatRow) ? layout.covers_start(d) : layout.exact(d);
					const RowRange mine = layout.mine();
					const size_t count = mine.second - mine.first;

					Node *chosen = nullptr;
					if (direct)
					{
						if constexpr (R == Role::Result)
							chosen = &arg.tile_for_write();
						else
							chosen = &arg.tile_for_read();
					}
					else if constexpr (R == Role::Result)
					{
						chosen = &arg.tile_for_write();
						if (count > 0)
						{
							Shape s = shape_of(arg);
							s.d[0] = count;
							hold.temp.reset(new Node(container_traits<C>::make(s)));
							hold.temp->setIndexOrigin(mine.first);
							chosen = hold.temp.get();
						}
					}
					else if constexpr (R == Role::ResultKeep)
					{
						mpi::fatal("An in place result is not in the blocks of the layout of the call");
						chosen = &arg.tile_for_read();
					}
					else
					{
						Node &tile = arg.tile_for_read();
						chosen = &tile;
						const T *data = nullptr;
						if (arg.local_size() > 0)
						{
							tile.updateHost();
							data = tile.getAddress();
						}
						std::vector<std::vector<RowRange>> needed(mpi::size());
						for (int r = 0; r < mpi::size(); ++r)
							if (layout.count(r) > 0)
								needed[r].push_back(layout.ranges[r]);
						const std::vector<T> rows = exchange_rows(data, d.unit, d, needed,
							R == Role::MatRow ? "MatRow: the rows of the matrix go to the processes that have the elements" : "layout: an element-wise container goes to the processes that have the rows",
							{arg.getObjectID()});
						if (count > 0)
						{
							Shape s = shape_of(arg);
							s.d[0] = count;
							hold.temp.reset(new Node(container_traits<C>::make(s)));
							std::copy(rows.begin(), rows.begin() + count * d.unit, hold.temp->getAddress());
							hold.temp->setIndexOrigin(mine.first);
							hold.temp->invalidateDeviceData();
							chosen = hold.temp.get();
						}
					}
					return (*chosen);
				}
			}

			// After the call: a result that was made in a container of the layout goes to the process that owns its rows; a
			// result is written
			template<Role R, typename Arg, typename Hold>
			void finish_argument(const RowLayout &layout, Hold &hold, Arg &&arg)
			{
				using C = std::decay_t<Arg>;
				if constexpr ((R == Role::Result || R == Role::ResultKeep) && is_distributed_container<C>::value)
				{
					using T = typename C::value_type;
					const mpi::BlockDistribution &d = arg.distribution();
					if (!layout.exact(d))
					{
						const T *data = nullptr;
						if (layout.my_count() > 0)
						{
							hold.temp->updateHost();
							data = hold.temp->getAddress();
						}
						std::vector<std::vector<RowRange>> needed(mpi::size());
						for (int r = 0; r < mpi::size(); ++r)
							if (d.count(r) > 0)
								needed[r].push_back({d.offset(r), d.offset(r) + d.count(r)});
						const std::vector<T> rows = exchange_rows(data, d.unit, layout, needed, "layout: the results go back to their owners", {arg.getObjectID()});
						if (arg.local_size() > 0)
						{
							auto &tile = arg.tile_for_write();
							std::copy(rows.begin(), rows.begin() + arg.local_size(), tile.getAddress());
							tile.invalidateDeviceData();
						}
					}
					arg.tile_written();
				}
			}
		}
	}
}
