#pragma once

// The skeletons of the MPI backend. A skeleton instance is a node-level skeleton (of the backends that the program
// is built for: sequential, OpenMP, OpenCL) that also accepts distributed containers: it brings the data that its
// processes need to them, runs the node-level skeleton on the tile of each process, and combines the results.

#include <array>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

#include "skepu3/mpi/vector.hpp"
#include "skepu3/mpi/rows.hpp"

namespace skepu
{
	namespace backend
	{
		// Everything that is not replaced here is the node-level backend
		using namespace ::skepu_node::backend;

		namespace mpi_detail
		{
			template<typename... Args>
			struct any_distributed : std::disjunction<is_distributed_container<Args>...> {};

			// The arguments of a call are, in this order: the results (outArity), the element-wise containers (arity),
			// the random access containers (anyArity) and the uniforms.

			// The argument of the node-level skeleton for an argument of the call
			template<size_t I, size_t outArity, size_t arity, size_t anyArity, typename Arg>
			decltype(auto) local_argument(Arg &&arg)
			{
				if constexpr (is_distributed_container<Arg>::value)
				{
					if constexpr (I < outArity)
						return (arg.tile_for_write());
					else if constexpr (I < outArity + arity)
						return (arg.tile_for_read());
					else
					{
						static_assert(I < outArity + arity + anyArity, "A distributed container is not a valid uniform argument");
						// The whole container, on every process. A container that is a random access argument (Vec,
						// Mat, ...) is only read in the MPI backend, whether or not the user function declares it const.
						return arg.replicated_view();
					}
				}
				else
					return std::forward<Arg>(arg);
			}

			// The event of a call (see trace.hpp): the containers that it wrote and read, by their ids. The arguments of a call are as
			// above; 'results' are the ids of what the call gives besides the containers among its arguments (a scalar).
			template<size_t outArity, size_t arity, size_t anyArity, size_t... I, typename Skeleton, typename... CallArgs>
			void trace_call(mpi::trace::CallScope &scope, Skeleton *skeleton, std::vector<size_t> &&elements, size_t local_elements,
				std::vector<int> &&results, int prng, pack_indices<I...>, CallArgs&&... args)
			{
#ifdef SKEPU_TRACING
				std::vector<int> outputs = std::move(results), elwise, proxies;
				auto classify = [&](auto index, auto &&arg)
				{
					constexpr size_t K = decltype(index)::value;
					if constexpr (is_distributed_container<std::decay_t<decltype(arg)>>::value)
					{
						if constexpr (K < outArity)
							outputs.push_back(arg.getObjectID());
						else if constexpr (K < outArity + arity)
							elwise.push_back(arg.getObjectID());
						else if constexpr (K < outArity + arity + anyArity)
							proxies.push_back(arg.getObjectID());
					}
				};
				// (only the uniform arguments can be scalars: the others are given as an int that is not one)
				auto uniform = [&](auto index, auto &&arg) -> decltype(auto)
				{
					constexpr size_t K = decltype(index)::value;
					if constexpr (K >= outArity + arity + anyArity)
						return std::forward<decltype(arg)>(arg);
					else
						return 0;
				};
				(classify(std::integral_constant<size_t, I>{}, get<I>(std::forward<CallArgs>(args)...)), ...);
				std::vector<int> scalars = mpi::trace::scalar_ids(uniform(std::integral_constant<size_t, I>{}, get<I>(std::forward<CallArgs>(args)...))...);
				scope.finish(skeleton, std::move(elements), local_elements, std::move(outputs), std::move(elwise), std::move(proxies), std::move(scalars), prng);
#else
				(void)scope; (void)skeleton; (void)elements; (void)local_elements; (void)results; (void)prng;
#endif
			}

			// On a single node, the random numbers of a skeleton call are taken from the stream of the program in the order of
			// the elements, N numbers for each. A process does its elements with the numbers of the stream that a single node
			// would use for them: its skeleton gets a copy of the stream that is advanced to the first element of the process,
			// and the stream of the program is advanced as the whole call would have advanced it (on every process: they must
			// all have the same stream). The guard does that for the duration of the call.
			class PrngShift
			{
			public:
				PrngShift(PRNG *&slot, size_t skip_uses, size_t total_uses): m_slot(slot), m_user(slot), m_total(total_uses)
				{
					if (!m_user)
						return; // (the node-level skeleton says that there is no stream)
					m_local.reset(new PRNG(*m_user));
					m_local->forward(skip_uses);
					m_slot = m_local.get();
				}

				~PrngShift()
				{
					if (!m_user)
						return;
					m_slot = m_user;
					m_user->forward(m_total);
				}

				PrngShift(const PrngShift &) = delete;
				PrngShift &operator=(const PrngShift &) = delete;

			private:
				PRNG *&m_slot;
				PRNG *m_user;
				size_t m_total;
				std::unique_ptr<PRNG> m_local;
			};

			// One value of each process that has one, in the order of the processes
			template<typename T>
			std::vector<T> gather_partials(bool valid, const T &partial)
			{
				std::vector<char> mine(1 + sizeof(T), 0);
				if (valid)
				{
					mine[0] = 1;
					std::memcpy(mine.data() + 1, &partial, sizeof(T));
				}

				std::vector<char> all(mine.size() * (size_t)mpi::size());
				mpi::trace::CommScope comm("allgather", "the partial results of the processes are combined");
				comm.bytes((unsigned long long)mine.size() * (unsigned long long)(mpi::size() - 1), (unsigned long long)mine.size() * (unsigned long long)(mpi::size() - 1));
				MPI_Allgather(mine.data(), (int)mine.size(), MPI_BYTE, all.data(), (int)mine.size(), MPI_BYTE, mpi::comm());

				std::vector<T> partials;
				for (int r = 0; r < mpi::size(); ++r)
				{
					const char *entry = all.data() + mine.size() * (size_t)r;
					if (entry[0])
					{
						T value;
						std::memcpy(&value, entry + 1, sizeof(T));
						partials.push_back(value);
					}
				}
				return partials;
			}

			// Every process that has rows (the rows are distributed as rows) has a partial result for each of the Hsize columns.
			// The processes that have columns of the results (distributed as columns) each get the combination of the
			// partial results for their columns, in the order of the processes: result[k] for the k-th column of this process.
			template<typename Ret, typename Combine>
			void combine_columns(const Ret *partials, size_t Hsize, const mpi::BlockDistribution &rows, Ret *result, Combine combine)
			{
				const int me = mpi::rank(), p = mpi::size();
				const size_t local_rows = rows.count(me);
				const mpi::BlockDistribution columns{Hsize, p};

				std::vector<int> send_counts(p), send_displs(p), receive_counts(p), receive_displs(p);
				int total = 0;
				for (int q = 0; q < p; ++q)
				{
					send_counts[q] = local_rows > 0 ? mpi::checked_count(columns.count(q), "column-wise reduction") : 0;
					send_displs[q] = mpi::checked_count(columns.offset(q), "column-wise reduction");
					receive_counts[q] = rows.count(q) > 0 ? mpi::checked_count(columns.count(me), "column-wise reduction") : 0;
					receive_displs[q] = total;
					total += receive_counts[q];
				}

				std::vector<Ret> received(total > 0 ? total : 1);
				mpi::trace::CommScope comm("alltoallv", "the partial results of the columns are combined by the owners of the columns");
				{
					unsigned long long sent = 0, got = 0;
					for (int q = 0; q < p; ++q)
					{
						if (q == me) continue;
						sent += (unsigned long long)send_counts[q] * sizeof(Ret);
						got += (unsigned long long)receive_counts[q] * sizeof(Ret);
					}
					comm.bytes(sent, got);
				}
				MPI_Alltoallv(local_rows > 0 ? partials : nullptr, send_counts.data(), send_displs.data(), mpi::datatype<Ret>(),
					received.data(), receive_counts.data(), receive_displs.data(), mpi::datatype<Ret>(), mpi::comm());

				for (size_t k = 0; k < columns.count(me); ++k)
				{
					bool first = true;
					for (int q = 0; q < p; ++q)
					{
						if (receive_counts[q] == 0)
							continue;
						const Ret &value = received[receive_displs[q] + k];
						result[k] = first ? value : combine(result[k], value);
						first = false;
					}
				}
			}
		}

		// ===================================================================================
		//   Map
		// ===================================================================================

		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		class Map : public ::skepu_node::backend::Map<arity, MapFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::Map<arity, MapFunc, CUDAKernel, CLKernel>;

			static constexpr size_t outArity = MapFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0) - (MapFunc::usesPRNG ? 1 : 0) + outArity;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setStride, Base)

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
					return this->distributed(typename make_pack_indices<numArgs, 0>::type{}, std::forward<CallArgs>(args)...);
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			template<size_t I>
			static constexpr bool is_result = I < outArity;
			template<size_t I>
			static constexpr bool is_elwise = I >= outArity && I < outArity + arity;

			// What an argument is: a result, an element-wise container, a row of a matrix for each element (MatRow), or needed
			// in its entirety
			template<size_t I>
			static constexpr mpi_detail::Role role()
			{
				if constexpr (I < outArity)
					return mpi_detail::Role::Result;
				else if constexpr (I < outArity + arity)
					return mpi_detail::Role::Elwise;
				else
					return mpi_detail::any_role<I, outArity + arity, typename MapFunc::ProxyTags>();
			}

			template<size_t... I>
			static constexpr bool has_matrow(pack_indices<I...>)
			{
				return ((role<I>() == mpi_detail::Role::MatRow) || ...);
			}

			// With a MatRow argument the elements are laid out so that the data that moves is the least (see rows.hpp)
			template<size_t... I, typename... CallArgs>
			auto &distributed_with_rows(pack_indices<I...>, CallArgs&&... args)
			{
				auto &first = get<0>(std::forward<CallArgs>(args)...);
				const size_t n = first.size();
				if (disjunction(size_mismatch<I>(get<I>(std::forward<CallArgs>(args)...), n, first.first_dimension())...))
					SKEPU_ERROR("Non-matching container sizes");

				mpi::trace::CallScope trace_scope("Map");
				const size_t elements = first.first_dimension();
				std::vector<mpi_detail::PartInfo> parts;
				(mpi_detail::add_part(parts, role<I>(), get<I>(std::forward<CallArgs>(args)...)), ...);
				for (const mpi_detail::PartInfo &part : parts)
					if (part.role == mpi_detail::Role::MatRow && part.distribution->n < elements)
						SKEPU_ERROR("A MatRow matrix has fewer rows than there are elements");
				const mpi_detail::RowLayout layout = mpi_detail::choose_layout(elements, parts, true);

				std::tuple<mpi_detail::RowHolder<std::decay_t<CallArgs>>...> holders;
				std::tuple<decltype(mpi_detail::exec_argument<role<I>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...)))...> local{
					mpi_detail::exec_argument<role<I>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...))...};

				const size_t mine = layout.my_count();
				auto run = [&]
				{
					if (mine > 0)
					{
						trace_scope.work_begin();
						std::apply([this](auto &... a) { Base::operator()(a...); }, local);
						trace_scope.work_finish();
					}
				};
				if constexpr (MapFunc::usesPRNG)
				{
					mpi_detail::PrngShift shift(this->m_prng, layout.mine().first * first.distribution().unit * MapFunc::randomCount, first.size() * MapFunc::randomCount);
					run();
				}
				else
					run();
				trace_scope.work_finished();

				(mpi_detail::finish_argument<role<I>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...)), ...);
				this->trace_call(trace_scope, first.size(), mine * first.distribution().unit, pack_indices<I...>{}, std::forward<CallArgs>(args)...);
				return first;
			}

			template<size_t... I, typename... CallArgs>
			auto &distributed(pack_indices<I...> indices, CallArgs&&... args)
			{
				if constexpr (has_matrow(pack_indices<I...>{}))
					return this->distributed_with_rows(indices, std::forward<CallArgs>(args)...);
				else
					return this->distributed_plain(indices, std::forward<CallArgs>(args)...);
			}

			template<size_t... I, typename... CallArgs>
			auto &distributed_plain(pack_indices<I...>, CallArgs&&... args)
			{
				mpi::trace::CallScope trace_scope("Map");
				auto &first = get<0>(std::forward<CallArgs>(args)...);
				const size_t n = first.size();
				if (disjunction(size_mismatch<I>(get<I>(std::forward<CallArgs>(args)...), n, first.first_dimension())...))
					SKEPU_ERROR("Non-matching container sizes");

				// The arguments of the node-level skeleton (this brings the data that is needed to the processes)
				std::tuple<decltype(mpi_detail::local_argument<I, outArity, arity, anyArity>(get<I>(std::forward<CallArgs>(args)...)))...> local(
					mpi_detail::local_argument<I, outArity, arity, anyArity>(get<I>(std::forward<CallArgs>(args)...))...);

				auto run = [&]
				{
					if (first.local_size() > 0)
					{
						trace_scope.work_begin();
						std::apply([this](auto &... a) { Base::operator()(a...); }, local);
						trace_scope.work_finish();
					}
				};
				if constexpr (MapFunc::usesPRNG)
				{
					mpi_detail::PrngShift shift(this->m_prng, first.local_offset() * MapFunc::randomCount, first.size() * MapFunc::randomCount);
					run();
				}
				else
					run();
				trace_scope.work_finished();

				// The results are the tiles now
				pack_expand((mark_written<I>(get<I>(std::forward<CallArgs>(args)...)), 0)...);
				this->trace_call(trace_scope, first.size(), first.local_size(), pack_indices<I...>{}, std::forward<CallArgs>(args)...);
				return first;
			}

			// (the id of the random numbers that the call uses, for its event)
			int random_id()
			{
#ifdef SKEPU_TRACING
				return this->template randomID<MapFunc::randomCount>();
#else
				return 0;
#endif
			}

			template<size_t... I, typename... CallArgs>
			void trace_call(mpi::trace::CallScope &scope, size_t global_elements, size_t local_elements, pack_indices<I...> indices, CallArgs&&... args)
			{
				mpi_detail::trace_call<outArity, arity, anyArity>(scope, this, {global_elements}, local_elements, {},
					this->random_id(), indices, std::forward<CallArgs>(args)...);
			}

			template<size_t I, typename Arg>
			static bool size_mismatch(Arg &&arg, size_t n, size_t first_dimension)
			{
				if constexpr (is_distributed_container<Arg>::value && (is_result<I> || is_elwise<I>))
					return arg.size() != n || arg.first_dimension() != first_dimension;
				else
					return false;
			}

			template<size_t I, typename Arg>
			static void mark_written(Arg &&arg)
			{
				if constexpr (is_distributed_container<Arg>::value && is_result<I>)
					arg.tile_written();
			}
		};

		// ===================================================================================
		//   Reduce (of a whole container)
		// ===================================================================================

		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		class Reduce1D : public ::skepu_node::backend::Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::Reduce1D<ReduceFunc, CUDAKernel, CLKernel>;
			using T = typename ReduceFunc::Ret;

		public:
			using Base::Base;
			using Base::operator();

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setReduceMode, Base)
			// (a value that is a braced list is converted to the type of the start value)
			template<typename U = T>
			auto &setStartValue(U value)
			{
				Base::setStartValue(value);
				return *this;
			}

			template<template<class> class Container>
			Scalar<T> operator()(Container<T> &arg)
			{
				if constexpr (is_distributed_container<Container<T>>::value)
					return this->distributed(arg);
				else
					return Base::operator()(arg);
			}

			// The reduction of each row (or each column) of a matrix to an element of a vector
			template<typename V, typename M, typename = std::enable_if_t<is_distributed_container<V>::value && is_distributed_container<M>::value>>
			V &operator()(V &res, M &arg)
			{
				mpi::trace::CallScope trace_scope("Reduce 1D-Partial");
				const size_t rows = arg.total_rows(), cols = arg.total_cols();
				const mpi::BlockDistribution distribution{rows, mpi::size()};
				const size_t local_rows = distribution.count(mpi::rank());
				auto traced = [&]
				{
					trace_scope.work_finished();
					mpi_detail::trace_call<1, 1, 0>(trace_scope, this, {rows, cols}, local_rows * cols, {}, 0, make_pack_indices<2, 0>::type{}, res, arg);
				};

				if (this->m_mode == ReduceMode::RowWise)
				{
					// The rows are distributed, and so are the results: a process reduces its rows
					if (res.size() != rows)
						SKEPU_ERROR("Non-matching container sizes");
					if (local_rows > 0)
					{
						trace_scope.work_begin();
						Base::operator()(res.tile_for_write(), arg.tile_for_read());
						trace_scope.work_finish();
					}
					res.tile_written();
					traced();
					return res;
				}

				// Every process reduces its rows to a partial result for each column, and the owner of a column of the results
				// combines the partial results of the processes in their order
				if (res.size() != cols)
					SKEPU_ERROR("Non-matching container sizes");
				::skepu_node::Vector<T> partial;
				if (local_rows > 0)
				{
					trace_scope.work_begin();
					partial.init(cols);
					Base::operator()(partial, arg.tile_for_read());
					partial.flush();
					trace_scope.work_finish();
				}
				trace_scope.work_finished();
				const mpi::BlockDistribution columns{cols, mpi::size()};
				T *result = columns.count(mpi::rank()) > 0 ? res.tile_for_write().getAddress() : nullptr;
				mpi_detail::combine_columns(local_rows > 0 ? partial.getAddress() : nullptr, cols, distribution, result,
					[](const T &a, const T &b) { return ReduceFunc::CPU(a, b); });
				if (result)
					res.tile_for_write().invalidateDeviceData();
				res.tile_written();
				traced();
				return res;
			}

		private:
			// Each process reduces its tile, and the partial results are combined in the order of the processes, by
			// every process (so that all have the same result, also for a function that is not commutative).
			// The start value is the first operand of the whole reduction, and is not part of the partial results.
			template<typename Container>
			Scalar<T> distributed(Container &arg)
			{
				mpi::trace::CallScope trace_scope("Reduce 1D-Full");
				auto &tile = arg.tile_for_read();
				const size_t local = arg.local_size();
				const T start = this->m_start;

				T partial{};
				if (local > 0)
				{
					trace_scope.work_begin();
					tile.flush();
					T *data = tile.getAddress();
					partial = data[0];
					if (local > 1)
					{
						// The first element is the start value of the reduction of the rest of the tile
						::skepu_node::Vector<T> rest(data + 1, local - 1, false);
						this->m_start = partial;
						partial = Base::operator()(rest);
						this->m_start = start;
					}
					trace_scope.work_finish();
				}
				trace_scope.work_finished();

				T result = start;
				for (const T &p : mpi_detail::gather_partials(local > 0, partial))
					result = ReduceFunc::CPU(result, p);
				Scalar<T> scalar(result);
				mpi_detail::trace_call<0, 1, 0>(trace_scope, this, {arg.size()}, local, mpi::trace::scalar_ids(scalar), 0, make_pack_indices<1, 0>::type{}, arg);
				return scalar;
			}
		};

		// ===================================================================================
		//   MapReduce
		// ===================================================================================

		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		class MapReduce : public ::skepu_node::backend::MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>;
			using Ret = typename MapFunc::Ret;

			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0) - (MapFunc::usesPRNG ? 1 : 0);
			static constexpr int indexDimensions = MapFunc::indexed ? index_dimension<typename MapFunc::IndexType>::value : 1;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			// (a value that is a braced list is converted to the type of the start value)
			template<typename U = Ret>
			auto &setStartValue(U value)
			{
				Base::setStartValue(value);
				return *this;
			}

			// (the node-level skeleton does not let us read the default size)
			auto &setDefaultSize(size_t i, size_t j = 0, size_t k = 0, size_t l = 0)
			{
				this->m_default_size = {{i, j, k, l}};
				Base::setDefaultSize(i, j, k, l);
				return *this;
			}

			template<typename... S>
			auto &setStride(S... strides)
			{
				this->m_strided = ((strides != 1) || ...);
				Base::setStride(strides...);
				return *this;
			}

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				if constexpr (arity > 0)
				{
					if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
						return this->distributed(typename make_pack_indices<numArgs, 0>::type{}, std::forward<CallArgs>(args)...);
					else
						return Base::operator()(std::forward<CallArgs>(args)...);
				}
				else
				{
					// There are no containers that give the elements: the indices of the default size are distributed
					static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching MapReduce function");
					return this->distributed_indices(typename make_pack_indices<numArgs, 0>::type{}, std::forward<CallArgs>(args)...);
				}
			}

		private:
			bool m_strided = false;
			std::array<size_t, 4> m_default_size{{0, 0, 0, 0}};

			// What an argument is: an element-wise container, a row of a matrix for each element (MatRow), or needed in its entirety
			template<size_t I>
			static constexpr mpi_detail::Role role()
			{
				if constexpr (I < arity)
					return mpi_detail::Role::Elwise;
				else
					return mpi_detail::any_role<I, arity, typename MapFunc::ProxyTags>();
			}

			template<size_t... I>
			static constexpr bool has_matrow(pack_indices<I...>)
			{
				return ((role<I>() == mpi_detail::Role::MatRow) || ...);
			}

			// With a MatRow argument the elements are laid out so that the data that moves is the least (see rows.hpp). The
			// layout is in the order of the processes, so the partial results are combined as before.
			template<size_t... I, typename... CallArgs>
			scalar_wrapper_t<Ret> distributed_with_rows(pack_indices<I...>, CallArgs&&... args)
			{
				static_assert(MapFunc::outArity == 1, "A MapReduce with several results does not work on distributed containers yet");
				if (this->m_strided)
					SKEPU_ERROR("A MapReduce with strides does not work on distributed containers");

				auto &first = get<0>(std::forward<CallArgs>(args)...);
				const size_t n = first.size();
				if (disjunction(size_mismatch<I>(get<I>(std::forward<CallArgs>(args)...), n, first.first_dimension())...))
					SKEPU_ERROR("Non-matching container sizes");

				mpi::trace::CallScope trace_scope("MapReduce");
				const size_t elements = first.first_dimension();
				std::vector<mpi_detail::PartInfo> parts;
				(mpi_detail::add_part(parts, role<I>(), get<I>(std::forward<CallArgs>(args)...)), ...);
				for (const mpi_detail::PartInfo &part : parts)
					if (part.role == mpi_detail::Role::MatRow && part.distribution->n < elements)
						SKEPU_ERROR("A MatRow matrix has fewer rows than there are elements");
				const mpi_detail::RowLayout layout = mpi_detail::choose_layout(elements, parts, true);

				std::tuple<mpi_detail::RowHolder<std::decay_t<CallArgs>>...> holders;
				std::tuple<decltype(mpi_detail::exec_argument<role<I>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...)))...> local{
					mpi_detail::exec_argument<role<I>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...))...};

				const bool has_elements = layout.my_count() > 0;
				Ret partial{};
				auto run = [&]
				{
					if (has_elements)
					{
						trace_scope.work_begin();
						Base::setApplyStartValue(mpi::rank() == 0);
						partial = std::apply([this](auto &... a) -> Ret { return Base::operator()(a...); }, local);
						Base::setApplyStartValue(true);
						trace_scope.work_finish();
					}
				};
				if constexpr (MapFunc::usesPRNG)
				{
					mpi_detail::PrngShift shift(this->m_prng, layout.mine().first * first.distribution().unit * MapFunc::randomCount, first.size() * MapFunc::randomCount);
					run();
				}
				else
					run();
				trace_scope.work_finished();
				auto result = this->combine(has_elements, partial);
				this->trace_call(trace_scope, first.size(), layout.my_count() * first.distribution().unit, result, pack_indices<I...>{}, std::forward<CallArgs>(args)...);
				return result;
			}

			template<size_t... I, typename... CallArgs>
			scalar_wrapper_t<Ret> distributed(pack_indices<I...> indices, CallArgs&&... args)
			{
				if constexpr (has_matrow(pack_indices<I...>{}))
					return this->distributed_with_rows(indices, std::forward<CallArgs>(args)...);
				else
					return this->distributed_plain(indices, std::forward<CallArgs>(args)...);
			}

			// The first process has the start value of the reduction: it does the reduction of its tile with it, and the
			// others do theirs without it. The partial results are combined in the order of the processes, so the start
			// value is the first operand of the whole reduction, as in a single node.
			template<size_t... I, typename... CallArgs>
			scalar_wrapper_t<Ret> distributed_plain(pack_indices<I...>, CallArgs&&... args)
			{
				static_assert(MapFunc::outArity == 1, "A MapReduce with several results does not work on distributed containers yet");
				if (this->m_strided)
					SKEPU_ERROR("A MapReduce with strides does not work on distributed containers");

				mpi::trace::CallScope trace_scope("MapReduce");
				auto &first = get<0>(std::forward<CallArgs>(args)...);
				const size_t n = first.size();
				if (disjunction(size_mismatch<I>(get<I>(std::forward<CallArgs>(args)...), n, first.first_dimension())...))
					SKEPU_ERROR("Non-matching container sizes");

				std::tuple<decltype(mpi_detail::local_argument<I, 0, arity, anyArity>(get<I>(std::forward<CallArgs>(args)...)))...> local(
					mpi_detail::local_argument<I, 0, arity, anyArity>(get<I>(std::forward<CallArgs>(args)...))...);

				const bool has_elements = first.local_size() > 0;
				Ret partial{};
				auto run = [&]
				{
					if (has_elements)
					{
						trace_scope.work_begin();
						Base::setApplyStartValue(mpi::rank() == 0);
						partial = std::apply([this](auto &... a) -> Ret { return Base::operator()(a...); }, local);
						Base::setApplyStartValue(true);
						trace_scope.work_finish();
					}
				};
				if constexpr (MapFunc::usesPRNG)
				{
					mpi_detail::PrngShift shift(this->m_prng, first.local_offset() * MapFunc::randomCount, first.size() * MapFunc::randomCount);
					run();
				}
				else
					run();
				trace_scope.work_finished();
				auto result = this->combine(has_elements, partial);
				this->trace_call(trace_scope, first.size(), first.local_size(), result, pack_indices<I...>{}, std::forward<CallArgs>(args)...);
				return result;
			}

			// The indices of the default size: the first dimension is distributed
			template<size_t... I, typename... CallArgs>
			scalar_wrapper_t<Ret> distributed_indices(pack_indices<I...>, CallArgs&&... args)
			{
				static_assert(MapFunc::outArity == 1, "A MapReduce with several results does not work on distributed containers yet");

				mpi::trace::CallScope trace_scope("MapReduce");
				const mpi::BlockDistribution rows{this->m_default_size[0], mpi::size()};
				if (rows.n == 0)
					SKEPU_ERROR("MapReduce<0> default size not set or set to 0.");

				size_t unit = 1;
				for (int k = 1; k < indexDimensions; ++k)
					unit *= this->m_default_size[k];

				// The rows of the index space of each process. With a MatRow argument they are those that the matrix has (nothing
				// moves: there are no other containers by element); without one, the blocks of the index space.
				std::vector<mpi_detail::PartInfo> parts;
				(mpi_detail::add_part(parts, mpi_detail::any_role<I, 0, typename MapFunc::ProxyTags>(), get<I>(std::forward<CallArgs>(args)...)), ...);
				for (const mpi_detail::PartInfo &part : parts)
					if (part.role == mpi_detail::Role::MatRow && part.distribution->n < rows.n)
						SKEPU_ERROR("A MatRow matrix has fewer rows than there are indices");
				const mpi_detail::RowLayout layout = mpi_detail::choose_layout(rows.n, parts, true);
				const size_t local_rows = layout.my_count();
				const size_t first_row = layout.mine().first;

				// Random access containers that are distributed are needed in their entirety, but for the matrices of MatRow
				std::tuple<mpi_detail::RowHolder<std::decay_t<CallArgs>>...> holders;
				std::tuple<decltype(mpi_detail::exec_argument<mpi_detail::any_role<I, 0, typename MapFunc::ProxyTags>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...)))...> local{
					mpi_detail::exec_argument<mpi_detail::any_role<I, 0, typename MapFunc::ProxyTags>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...))...};

				Ret partial{};
				auto run = [&]
				{
					if (local_rows > 0)
					{
						trace_scope.work_begin();
						Base::setDefaultSize(local_rows, this->m_default_size[1], this->m_default_size[2], this->m_default_size[3]);
						Base::setDefaultOrigin(first_row * unit);
						Base::setApplyStartValue(mpi::rank() == 0);
						partial = std::apply([this](auto &... a) -> Ret { return Base::operator()(a...); }, local);
						Base::setApplyStartValue(true);
						Base::setDefaultOrigin(0);
						Base::setDefaultSize(this->m_default_size[0], this->m_default_size[1], this->m_default_size[2], this->m_default_size[3]);
						trace_scope.work_finish();
					}
				};
				if constexpr (MapFunc::usesPRNG)
				{
					mpi_detail::PrngShift shift(this->m_prng, first_row * unit * MapFunc::randomCount, rows.n * unit * MapFunc::randomCount);
					run();
				}
				else
					run();
				trace_scope.work_finished();
				auto result = this->combine(local_rows > 0, partial);
				this->trace_call(trace_scope, rows.n * unit, local_rows * unit, result, pack_indices<I...>{}, std::forward<CallArgs>(args)...);
				return result;
			}

			// (the id of the random numbers that the call uses, for its event)
			int random_id()
			{
#ifdef SKEPU_TRACING
				return this->template randomID<MapFunc::randomCount>();
#else
				return 0;
#endif
			}

			template<size_t... I, typename... CallArgs>
			void trace_call(mpi::trace::CallScope &scope, size_t global_elements, size_t local_elements, const scalar_wrapper_t<Ret> &result, pack_indices<I...> indices, CallArgs&&... args)
			{
				mpi_detail::trace_call<0, arity, anyArity>(scope, this, {global_elements}, local_elements, mpi::trace::scalar_ids(result),
					this->random_id(), indices, std::forward<CallArgs>(args)...);
			}

			scalar_wrapper_t<Ret> combine(bool has_partial, const Ret &partial)
			{
				auto partials = mpi_detail::gather_partials(has_partial, partial);
				Ret result = partials[0];
				for (size_t r = 1; r < partials.size(); ++r)
					result = ReduceFunc::CPU(result, partials[r]);
				return scalar_wrapper_t<Ret>(result);
			}

			template<size_t I, typename Arg>
			static bool size_mismatch(Arg &&arg, size_t n, size_t first_dimension)
			{
				if constexpr (is_distributed_container<Arg>::value && I < arity)
					return arg.size() != n || arg.first_dimension() != first_dimension;
				else
					return false;
			}
		};

		// ===================================================================================
		//   MapPairs
		// ===================================================================================

		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename CUDAKernel, typename CLKernel>
		class MapPairs : public ::skepu_node::backend::MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapPairs<Varity, Harity, MapPairsFunc, CUDAKernel, CLKernel>;

			static constexpr size_t outArity = MapPairsFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapPairsFunc::ContainerArgs>::value;
			static constexpr size_t numArgs = MapPairsFunc::totalArity - (MapPairsFunc::indexed ? 1 : 0) - (MapPairsFunc::usesPRNG ? 1 : 0) + outArity;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setDefaultSize, Base)

			auto &setInPlace(bool in_place)
			{
				this->m_in_place_requested = in_place;
				Base::setInPlace(in_place);
				return *this;
			}

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching MapPairs function");
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
					return this->distributed(typename make_pack_indices<numArgs, 0>::type{}, std::forward<CallArgs>(args)...);
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			bool m_in_place_requested = false;

			// The results (a matrix) are distributed by rows, and so are the vertical element-wise vectors (one element
			// per row). The horizontal ones (one element per column) and the random access containers are needed in
			// their entirety by every process.
			template<size_t I, typename Arg>
			decltype(auto) local_argument(Arg &&arg)
			{
				if constexpr (is_distributed_container<Arg>::value)
				{
					if constexpr (I < outArity)
					{
						if (this->m_in_place_requested)
							return (arg.tile_for_read()); // the results are added to what is there
						return (arg.tile_for_write());
					}
					else if constexpr (I < outArity + Varity)
						return (arg.tile_for_read());
					else
					{
						static_assert(I < outArity + Varity + Harity + anyArity, "A distributed container is not a valid uniform argument");
						return arg.replicated_view();
					}
				}
				else
					return std::forward<Arg>(arg);
			}

			// What an argument is (see rows.hpp): the horizontal vectors are needed in their entirety
			template<size_t I>
			constexpr mpi_detail::Role role() const
			{
				if constexpr (I < outArity)
					return mpi_detail::Role::Result;
				else if constexpr (I < outArity + Varity)
					return mpi_detail::Role::Elwise;
				else if constexpr (I < outArity + Varity + Harity)
					return mpi_detail::Role::Other;
				else
					return mpi_detail::any_role<I, outArity + Varity + Harity, typename MapPairsFunc::ProxyTags>();
			}

			template<size_t... I>
			constexpr bool has_matrow(pack_indices<I...>) const
			{
				return ((role<I>() == mpi_detail::Role::MatRow) || ...);
			}

			// With a MatRow argument the rows of the pairs are laid out so that the data that moves is the least (see rows.hpp).
			// The results of an in place call (added to) stay in their blocks.
			template<size_t... I, typename... CallArgs>
			auto &distributed_with_rows(pack_indices<I...>, CallArgs&&... args)
			{
				auto &first = get<0>(std::forward<CallArgs>(args)...);
				const size_t rows = first.total_rows();
				if (disjunction(is_mismatch<I>(get<I>(std::forward<CallArgs>(args)...), rows, first.total_cols())...))
					SKEPU_ERROR("Non-matching container sizes");

				mpi::trace::CallScope trace_scope("MapPairs");
				const bool keep = this->m_in_place_requested;
				std::vector<mpi_detail::PartInfo> parts;
				(mpi_detail::add_part(parts, (keep && role<I>() == mpi_detail::Role::Result) ? mpi_detail::Role::ResultKeep : role<I>(), get<I>(std::forward<CallArgs>(args)...)), ...);
				for (const mpi_detail::PartInfo &part : parts)
					if (part.role == mpi_detail::Role::MatRow && part.distribution->n < rows)
						SKEPU_ERROR("A MatRow matrix has fewer rows than the pairs have");
				const mpi_detail::RowLayout layout = mpi_detail::choose_layout(rows, parts, !keep);

				std::tuple<mpi_detail::RowHolder<std::decay_t<CallArgs>>...> holders;
				std::tuple<decltype(mpi_detail::exec_argument<role<I>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...)))...> local{
					exec<I>(keep, layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...))...};

				const size_t mine = layout.my_count();
				auto run = [&]
				{
					if (mine > 0)
					{
						trace_scope.work_begin();
						std::apply([this](auto &... a) { Base::operator()(a...); }, local);
						trace_scope.work_finish();
					}
				};
				if constexpr (MapPairsFunc::usesPRNG)
				{
					// (the elements are in the order of the rows)
					mpi_detail::PrngShift shift(this->m_prng, layout.mine().first * first.distribution().unit * MapPairsFunc::randomCount, first.size() * MapPairsFunc::randomCount);
					run();
				}
				else
					run();

				trace_scope.work_finished();
				(mpi_detail::finish_argument<role<I>()>(layout, std::get<I>(holders), get<I>(std::forward<CallArgs>(args)...)), ...);
				this->trace_call(trace_scope, first.total_rows(), first.total_cols(), mine * first.distribution().unit, pack_indices<I...>{}, std::forward<CallArgs>(args)...);
				return first;
			}

			// (the id of the random numbers that the call uses, for its event)
			int random_id()
			{
#ifdef SKEPU_TRACING
				return this->template randomID<MapPairsFunc::randomCount>();
#else
				return 0;
#endif
			}

			template<size_t... I, typename... CallArgs>
			void trace_call(mpi::trace::CallScope &scope, size_t rows, size_t cols, size_t local_elements, pack_indices<I...> indices, CallArgs&&... args)
			{
				mpi_detail::trace_call<outArity, Varity + Harity, anyArity>(scope, this, {cols, rows}, local_elements, {}, this->random_id(), indices,
					std::forward<CallArgs>(args)...);
			}

			// An in place result is read (and is not moved: see distributed_with_rows)
			template<size_t I, typename Hold, typename Arg>
			decltype(auto) exec(bool keep, const mpi_detail::RowLayout &layout, Hold &hold, Arg &&arg)
			{
				if constexpr (role<I>() == mpi_detail::Role::Result)
				{
					if (keep)
						return mpi_detail::exec_argument<mpi_detail::Role::ResultKeep>(layout, hold, std::forward<Arg>(arg));
				}
				return mpi_detail::exec_argument<role<I>()>(layout, hold, std::forward<Arg>(arg));
			}

			template<size_t... I, typename... CallArgs>
			auto &distributed(pack_indices<I...> indices, CallArgs&&... args)
			{
				if constexpr (has_matrow(pack_indices<I...>{}))
					return this->distributed_with_rows(indices, std::forward<CallArgs>(args)...);
				else
					return this->distributed_plain(indices, std::forward<CallArgs>(args)...);
			}

			template<size_t... I, typename... CallArgs>
			auto &distributed_plain(pack_indices<I...>, CallArgs&&... args)
			{
				mpi::trace::CallScope trace_scope("MapPairs");
				auto &first = get<0>(std::forward<CallArgs>(args)...);
				const size_t rows = first.total_rows();
				if (disjunction(is_mismatch<I>(get<I>(std::forward<CallArgs>(args)...), rows, first.total_cols())...))
					SKEPU_ERROR("Non-matching container sizes");

				std::tuple<decltype(this->template local_argument<I>(get<I>(std::forward<CallArgs>(args)...)))...> local(
					this->template local_argument<I>(get<I>(std::forward<CallArgs>(args)...))...);

				auto run = [&]
				{
					if (first.local_size() > 0)
					{
						trace_scope.work_begin();
						std::apply([this](auto &... a) { Base::operator()(a...); }, local);
						trace_scope.work_finish();
					}
				};
				if constexpr (MapPairsFunc::usesPRNG)
				{
					// (the elements are in the order of the rows)
					mpi_detail::PrngShift shift(this->m_prng, first.local_offset() * MapPairsFunc::randomCount, first.size() * MapPairsFunc::randomCount);
					run();
				}
				else
					run();

				trace_scope.work_finished();
				pack_expand((mark_written<I>(get<I>(std::forward<CallArgs>(args)...)), 0)...);
				this->trace_call(trace_scope, rows, first.total_cols(), first.local_size(), pack_indices<I...>{}, std::forward<CallArgs>(args)...);
				return first;
			}

			// The results have the size of the first; a vertical vector has a row of the results for each element, and a
			// horizontal one a column
			template<size_t I, typename Arg>
			static bool is_mismatch(Arg &&arg, size_t rows, size_t cols)
			{
				if constexpr (is_distributed_container<Arg>::value)
				{
					if constexpr (I < outArity)
						return arg.total_rows() != rows || arg.total_cols() != cols;
					else if constexpr (I < outArity + Varity)
						return arg.size() != rows;
					else if constexpr (I < outArity + Varity + Harity)
						return arg.size() != cols;
					else
						return false;
				}
				else
					return false;
			}

			template<size_t I, typename Arg>
			static void mark_written(Arg &&arg)
			{
				if constexpr (is_distributed_container<Arg>::value && I < outArity)
					arg.tile_written();
			}
		};

		// ===================================================================================
		//   MapPairsReduce
		// ===================================================================================

		template<size_t Varity, size_t Harity, typename MapPairsFunc, typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		class MapPairsReduce : public ::skepu_node::backend::MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CLKernel>
		{
			using Base = ::skepu_node::backend::MapPairsReduce<Varity, Harity, MapPairsFunc, ReduceFunc, CUDAKernel, CLKernel>;
			using Ret = typename ReduceFunc::Ret;

			static constexpr size_t outArity = MapPairsFunc::outArity;
			static constexpr size_t anyArity = std::tuple_size<typename MapPairsFunc::ContainerArgs>::value;
			static constexpr size_t numArgs = MapPairsFunc::totalArity - (MapPairsFunc::indexed ? 1 : 0) - (MapPairsFunc::usesPRNG ? 1 : 0) + outArity;

		public:
			using Base::Base;

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			// (a value that is a braced list is converted to the type of the start value)
			template<typename U = Ret>
			auto &setStartValue(U value)
			{
				Base::setStartValue(value);
				return *this;
			}

			// (the node-level skeleton does not let us read what it was given)
			auto &setDefaultSize(size_t i, size_t j = 0)
			{
				this->m_default_i = i;
				this->m_default_j = j;
				Base::setDefaultSize(i, j);
				return *this;
			}

			auto &setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
				Base::setReduceMode(mode);
				return *this;
			}

			template<typename... CallArgs>
			decltype(auto) operator()(CallArgs&&... args)
			{
				static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching MapPairsReduce function");
				if constexpr (mpi_detail::any_distributed<CallArgs...>::value)
					return this->distributed(typename make_pack_indices<numArgs, 0>::type{}, std::forward<CallArgs>(args)...);
				else
					return Base::operator()(std::forward<CallArgs>(args)...);
			}

		private:
			size_t m_default_i = 0, m_default_j = 0;
			ReduceMode m_mode = ReduceMode::RowWise;

			// What the node-level skeleton gets: the vertical element-wise vectors are distributed by rows (one element per
			// row), the horizontal ones and the random access containers are needed in their entirety by every process.
			// The results of a row-wise reduction are one per row (distributed by rows, like the vertical vectors); those of
			// a column-wise one are one per column, and every process has a part of each of them (see column_wise).
			template<size_t I, typename Arg>
			static decltype(auto) local_argument(Arg &&arg)
			{
				if constexpr (is_distributed_container<Arg>::value)
				{
					if constexpr (I < outArity)
						return (arg.tile_for_write());
					else if constexpr (I < outArity + Varity)
						return (arg.tile_for_read());
					else
					{
						static_assert(I < outArity + Varity + Harity + anyArity, "A distributed container is not a valid uniform argument");
						return arg.replicated_view();
					}
				}
				else
					return std::forward<Arg>(arg);
			}

			template<size_t... I, typename... CallArgs>
			auto &distributed(pack_indices<I...> indices, CallArgs&&... args)
			{
				static_assert(outArity == 1, "A MapPairsReduce with several results does not work on distributed containers yet");
				if constexpr (MapPairsFunc::usesPRNG)
					if (this->m_mode == ReduceMode::ColWise)
						SKEPU_ERROR("A column-wise MapPairsReduce with random numbers does not work on distributed containers (the elements are not used in the order of the rows)");

				mpi::trace::CallScope trace_scope("MapPairsReduce");
				auto &out = get<0>(std::forward<CallArgs>(args)...);
				const size_t Vsize = vertical_size(std::forward<CallArgs>(args)...);
				const size_t Hsize = horizontal_size(std::forward<CallArgs>(args)...);
				if (disjunction(is_mismatch<I>(get<I>(std::forward<CallArgs>(args)...), Vsize, Hsize)...)
					|| out.size() != (this->m_mode == ReduceMode::RowWise ? Vsize : Hsize))
					SKEPU_ERROR("Non-matching container sizes");

				// The rows of this process
				const mpi::BlockDistribution rows{Vsize, mpi::size()};
				const size_t local_rows = rows.count(mpi::rank()), origin = rows.offset(mpi::rank());

				if (this->m_mode == ReduceMode::RowWise)
				{
					std::tuple<decltype(local_argument<I>(get<I>(std::forward<CallArgs>(args)...)))...> local(
						local_argument<I>(get<I>(std::forward<CallArgs>(args)...))...);

					auto run = [&]
					{
						if (local_rows > 0)
						{
							trace_scope.work_begin();
							this->prepare(local_rows, Hsize, origin);
							std::apply([this](auto &... a) { Base::operator()(a...); }, local);
							this->finish();
							trace_scope.work_finish();
						}
					};
					if constexpr (MapPairsFunc::usesPRNG)
					{
						mpi_detail::PrngShift shift(this->m_prng, origin * Hsize * MapPairsFunc::randomCount, Vsize * Hsize * MapPairsFunc::randomCount);
						run();
					}
					else
						run();
					trace_scope.work_finished();
					out.tile_written();
				}
				else
					this->column_wise(trace_scope, out, indices, rows, Hsize, std::forward<CallArgs>(args)...);
				mpi_detail::trace_call<outArity, Varity + Harity, anyArity>(trace_scope, this, {Hsize, Vsize}, local_rows * Hsize, {}, this->random_id(), indices,
					std::forward<CallArgs>(args)...);
				return out;
			}

			// (the id of the random numbers that the call uses, for its event)
			int random_id()
			{
#ifdef SKEPU_TRACING
				return this->template randomID<MapPairsFunc::randomCount>();
#else
				return 0;
#endif
			}

			void prepare(size_t local_rows, size_t Hsize, size_t origin)
			{
				Base::setRowOrigin(origin);
				if constexpr (Varity == 0)
					Base::setDefaultSize(local_rows, this->m_default_j);
				(void)Hsize;
			}

			void finish()
			{
				Base::setRowOrigin(0);
				if constexpr (Varity == 0)
					Base::setDefaultSize(this->m_default_i, this->m_default_j);
			}

			// Each process reduces over its rows, to a partial result for every column; the process that has a column of
			// the results combines the partial results of the processes in the order of the processes. The first process
			// has the start value, the others do their part without it.
			template<size_t... I, typename Out, typename... CallArgs>
			void column_wise(mpi::trace::CallScope &trace_scope, Out &out, pack_indices<I...>, const mpi::BlockDistribution &rows, size_t Hsize, CallArgs&&... args)
			{
				const int me = mpi::rank(), p = mpi::size();
				const size_t local_rows = rows.count(me);

				// The arguments of the node-level skeleton, with a vector of partial results (for every column) in the place of
				// the results
				::skepu_node::Vector<Ret> partial;
				std::tuple<decltype(column_argument<I>(partial, get<I>(std::forward<CallArgs>(args)...)))...> local(
					column_argument<I>(partial, get<I>(std::forward<CallArgs>(args)...))...);

				if (local_rows > 0)
				{
					trace_scope.work_begin();
					partial.init(Hsize);
					this->prepare(local_rows, Hsize, rows.offset(me));
					Base::setApplyStartValue(me == 0);
					std::apply([this](auto &... a) { Base::operator()(a...); }, local);
					Base::setApplyStartValue(true);
					this->finish();
					partial.flush();
					trace_scope.work_finish();
				}
				trace_scope.work_finished();

				// The columns of the results that this process has
				const mpi::BlockDistribution columns{Hsize, p};
				const size_t mine = columns.count(me);
				if (mine > 0)
				{
					Ret *result = out.tile_for_write().getAddress();
					mpi_detail::combine_columns(local_rows > 0 ? partial.getAddress() : nullptr, Hsize, rows, result,
						[](const Ret &a, const Ret &b) { return ReduceFunc::CPU(a, b); });
					out.tile_for_write().invalidateDeviceData();
				}
				else
					mpi_detail::combine_columns<Ret>(local_rows > 0 ? partial.getAddress() : nullptr, Hsize, rows, nullptr,
						[](const Ret &a, const Ret &b) { return ReduceFunc::CPU(a, b); });
				out.tile_written();
			}

			template<size_t I, typename Arg>
			static decltype(auto) column_argument(::skepu_node::Vector<Ret> &partial, Arg &&arg)
			{
				if constexpr (I < outArity)
					return (partial);
				else
					return local_argument<I>(std::forward<Arg>(arg));
			}

			template<typename... CallArgs>
			size_t vertical_size(CallArgs&&... args)
			{
				if constexpr (Varity > 0)
					return get<outArity>(std::forward<CallArgs>(args)...).size();
				else
					return this->m_default_i;
			}

			template<typename... CallArgs>
			size_t horizontal_size(CallArgs&&... args)
			{
				if constexpr (Harity > 0)
					return get<outArity + Varity>(std::forward<CallArgs>(args)...).size();
				else
					return this->m_default_j;
			}

			template<size_t I, typename Arg>
			static bool is_mismatch(Arg &&arg, size_t Vsize, size_t Hsize)
			{
				if constexpr (is_distributed_container<Arg>::value)
				{
					if constexpr (I >= outArity && I < outArity + Varity)
						return arg.size() != Vsize;
					else if constexpr (I >= outArity + Varity && I < outArity + Varity + Harity)
						return arg.size() != Hsize;
					else
						return false;
				}
				else
					return false;
			}
		};

		// ===================================================================================
		//   Reduce of a matrix with two functions (first the rows, then the results of the rows)
		// ===================================================================================

		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		class Reduce2D : public ::skepu_node::backend::Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		{
			using Base = ::skepu_node::backend::Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>;
			using OneDimension = ::skepu_node::backend::Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>;
			using T = typename ReduceFuncRowWise::Ret;

		public:
			using Base::Base;
			using Base::operator();

			SKEPU_CHAIN_SKELETON_SETTERS(Base)
			SKEPU_CHAIN(setReduceMode, Base)
			// (a value that is a braced list is converted to the type of the start value)
			template<typename U = T>
			auto &setStartValue(U value)
			{
				Base::setStartValue(value);
				return *this;
			}

			template<template<class> class Container>
			T operator()(Container<T> &arg)
			{
				if constexpr (is_distributed_container<Container<T>>::value)
				{
					static_assert(!std::is_same<Container<T>, Vector<T>>::value, "A Reduce with two functions of a distributed vector does not work yet: use one function");
					return this->distributed(arg);
				}
				else
					return Base::operator()(arg);
			}

		private:
			// The rows (or the columns) are reduced with the first function, and the results with the second one, from the
			// start value (the first operand) on. A process does its part (and the first has the start value), and the parts
			// are combined with the second function in the order of the processes.
			template<typename Matrix>
			T distributed(Matrix &arg)
			{
				mpi::trace::CallScope trace_scope("Reduce 2D");
				const size_t rows = arg.total_rows(), cols = arg.total_cols();
				const mpi::BlockDistribution distribution{rows, mpi::size()};
				const int me = mpi::rank();
				const size_t local_rows = distribution.count(me);
				const T start = this->m_start;

				// The results of the reductions that this process has: of its rows, or of its columns
				std::vector<T> results;
				if (this->m_mode == ReduceMode::RowWise)
				{
					if (local_rows > 0)
					{
						trace_scope.work_begin();
						::skepu_node::Vector<T> rows_result(local_rows);
						OneDimension::operator()(rows_result, arg.tile_for_read());
						rows_result.flush();
						results.assign(rows_result.getAddress(), rows_result.getAddress() + local_rows);
						trace_scope.work_finish();
					}
					trace_scope.work_finished();
				}
				else
				{
					::skepu_node::Vector<T> partial;
					if (local_rows > 0)
					{
						trace_scope.work_begin();
						partial.init(cols);
						OneDimension::operator()(partial, arg.tile_for_read());
						partial.flush();
						trace_scope.work_finish();
					}
					trace_scope.work_finished();
					const mpi::BlockDistribution columns{cols, mpi::size()};
					results.resize(columns.count(me));
					mpi_detail::combine_columns(local_rows > 0 ? partial.getAddress() : nullptr, cols, distribution, results.data(),
						[](const T &a, const T &b) { return ReduceFuncRowWise::CPU(a, b); });
				}

				// Reduced with the second function; the first process has the start value
				T mine{};
				const bool has_results = !results.empty();
				if (has_results)
				{
					size_t first = 0;
					if (me == 0)
						mine = start;
					else
						mine = results[first++];
					for (size_t k = first; k < results.size(); ++k)
						mine = ReduceFuncColWise::CPU(mine, results[k]);
				}

				auto partials = mpi_detail::gather_partials(has_results, mine);
				T result = partials[0];
				for (size_t r = 1; r < partials.size(); ++r)
					result = ReduceFuncColWise::CPU(result, partials[r]);
				mpi_detail::trace_call<0, 1, 0>(trace_scope, this, {rows, cols}, local_rows * cols, {}, 0, make_pack_indices<1, 0>::type{}, arg);
				return result;
			}
		};

		// ===================================================================================
		//   Scan
		// ===================================================================================

		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		class Scan : public ::skepu_node::backend::Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		{
			using Base = ::skepu_node::backend::Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>;
			using T = typename ScanFunc::Ret;

		public:
			using Base::Base;
			using Base::operator();

			SKEPU_CHAIN_SKELETON_SETTERS(Base)

			// (the node-level skeleton does not let us read what it was given)
			auto &setScanMode(ScanMode mode)
			{
				this->m_scan_mode = mode;
				Base::setScanMode(mode);
				return *this;
			}

			auto &setStartValue(T value)
			{
				this->m_initial = value;
				Base::setStartValue(value);
				return *this;
			}

			template<template<class> class Container, typename In>
			Container<T> &operator()(Container<T> &res, In &&arg)
			{
				if constexpr (is_distributed_container<Container<T>>::value && is_distributed_container<In>::value)
					return this->distributed(res, arg);
				else
					return Base::operator()(res, std::forward<In>(arg));
			}

		private:
			ScanMode m_scan_mode = ScanMode::Inclusive;
			T m_initial{};

			// The elements are in their order in the memory. Every process scans its part, started from what comes before
			// it: the total of the parts of the processes before it, in the order of the processes (and the start value
			// in an exclusive scan: an inclusive scan does not have one). That total comes from a fold of the elements of each
			// part, which needs no identity element.
			template<typename Container, typename In>
			Container &distributed(Container &res, In &arg)
			{
				mpi::trace::CallScope trace_scope("Scan");
				if (res.size() != arg.size() || res.first_dimension() != arg.first_dimension())
					SKEPU_ERROR("Non-matching container sizes");

				const int me = mpi::rank();
				const size_t local = res.local_size();
				auto &in_tile = arg.tile_for_read();

				// The total of this part
				T total{};
				if (local > 0)
				{
					in_tile.updateHost();
					const T *data = in_tile.getAddress();
					total = data[0];
					for (size_t i = 1; i < local; ++i)
						total = ScanFunc::CPU(total, data[i]);
				}
				const auto totals = mpi_detail::gather_partials(local > 0, total);

				// How many of the totals are those of the processes before this one
				size_t before = 0;
				for (int q = 0; q < me; ++q)
					if (res.distribution().count(q) > 0)
						++before;

				const bool exclusive = this->m_scan_mode == ScanMode::Exclusive;
				if (local > 0)
				{
					trace_scope.work_begin();
					auto &out_tile = res.tile_for_write();
					if (exclusive)
					{
						T start = this->m_initial;
						for (size_t i = 0; i < before; ++i)
							start = ScanFunc::CPU(start, totals[i]);
						Base::setStartValue(start);
						Base::operator()(out_tile, in_tile);
						Base::setStartValue(this->m_initial);
					}
					else
					{
						Base::operator()(out_tile, in_tile);
						if (before > 0)
						{
							T prefix = totals[0];
							for (size_t i = 1; i < before; ++i)
								prefix = ScanFunc::CPU(prefix, totals[i]);
							out_tile.updateHost();
							T *data = out_tile.getAddress();
							for (size_t i = 0; i < local; ++i)
								data[i] = ScanFunc::CPU(prefix, data[i]);
							out_tile.invalidateDeviceData();
						}
					}
					trace_scope.work_finish();
				}
				trace_scope.work_finished();
				res.tile_written();
				mpi_detail::trace_call<1, 1, 0>(trace_scope, this, {res.size()}, local, {}, 0, make_pack_indices<2, 0>::type{}, res, arg);
				return res;
			}
		};
	}
}
