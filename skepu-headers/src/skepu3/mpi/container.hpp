#pragma once

// What the distributed containers (Vector, Matrix, Tensor3, Tensor4) have in common: a block of the first dimension
// per process, held in a node-level container (the tile), and the host access to the whole container.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>

#include "skepu3/mpi/context.hpp"
#include "skepu3/mpi/trace.hpp"

namespace skepu
{
	namespace mpi
	{
		// Host code runs on every process (a program is written for one process), but code in an skepu::external
		// call runs on the first process only: no communication is possible in it.
		inline int &external_depth()
		{
			static int depth = 0;
			return depth;
		}

		namespace detail
		{
			struct DistributedTag {};

			/*!
			 *  The state: the tile and the gathered copy (the "mirror"). Element access from the host gives access to the
			 *  whole container: the first access gathers it to every process (a collective operation, as the program
			 *  runs on all processes), and it stays gathered until a skeleton changes the container. Writes through
			 *  the element access are distributed to the owners before the container is used by a skeleton.
			 *
			 *  Which of the two is valid is the same on every process: it changes only in operations that every process
			 *  makes. That is what lets the processes agree on when to communicate.
			 */
			template<typename T, typename Tile>
			class Distributed : public DistributedTag
			{
			public:
				typedef T value_type;
				typedef size_t size_type;
				typedef ptrdiff_t difference_type;
				typedef T *pointer;
				typedef T &reference;
				typedef const T &const_reference;
				typedef T *iterator;
				typedef const T *const_iterator;
				typedef Tile tile_type;

				// -------------------- Size --------------------

				// The number of elements
				size_type size() const { return this->m_dist.total(); }
				bool empty() const { return this->size() == 0; }

				void setLabel(std::string label)
				{
					this->m_label = label;
					if (this->m_identity.live)
						this->m_identity.name = label; // (the later events of the container, the deallocation)
					SKEPU_TRACE_RELABEL(this->m_identity.id, label);
				}
				bool hasLabel() const { return !this->m_label.empty(); }
				std::string getLabel() const { return this->m_label; }

				// The id of the container in the traces (it goes with the data when the container is moved)
				int getObjectID() const { return this->m_identity.id; }

				// -------------------- Element access from the host --------------------

				iterator begin() { return this->write_access(); }
				iterator end() { return this->write_access() + this->size(); }
				const_iterator begin() const { return this->read_access(); }
				const_iterator end() const { return this->read_access() + this->size(); }
				T *data() { return this->write_access(); }
				const T *data() const { return this->read_access(); }
				T *getAddress() { return this->write_access(); }
				const T *getAddress() const { return this->read_access(); }

				// Bring the host data up to date (on every process)
				void flush(FlushMode = FlushMode::Default) const { this->ensure_mirror(); }

				// -------------------- The distribution (for the skeletons) --------------------

				// The elements of this process, and where they are in the whole container
				size_type local_size() const { return this->m_dist.elements(mpi::rank()); }
				size_type local_offset() const { return this->m_dist.element_offset(mpi::rank()); }
				// The indices of the first dimension of this process (rows of a matrix)
				size_type local_first_dimension() const { return this->m_dist.count(mpi::rank()); }
				size_type local_first_offset() const { return this->m_dist.offset(mpi::rank()); }
				size_type first_dimension() const { return this->m_dist.n; }
				const mpi::BlockDistribution &distribution() const { return this->m_dist; }

				// The tile, up to date, to be read by a skeleton
				Tile &tile_for_read()
				{
					this->ensure_tile();
					return *this->m_tile;
				}

				// The tile, to be written by a skeleton. Call tile_written() when it is.
				Tile &tile_for_write() { return *this->m_tile; }

				void tile_written()
				{
					this->m_fill_pending = false;
					this->m_root_only = false;
					this->m_mirror_at_root = false;
					this->m_tile_valid = true;
					this->m_mirror_valid = false;
					this->m_mirror.reset();
				}

				// The whole container on every process, to be read (a random access argument of a skeleton)
				const T *replicated() const
				{
					this->ensure_mirror("random access argument: the container is gathered");
					return this->m_mirror.get();
				}

				// -------------------- For skepu::external --------------------

				struct ExternalState { bool tile_valid; };

				ExternalState external_begin() const
				{
					// (the code runs on the first process, which is the only one that needs the container)
					if (!this->m_root_only && !this->m_mirror_valid && !this->m_mirror_at_root)
						this->gather_for_external();
					this->m_external_listed = true;
					this->m_extents_at_begin = this->m_extents;
					return ExternalState{this->m_tile_valid};
				}

				void external_end_read(ExternalState state) const
				{
					this->m_tile_valid = state.tile_valid;
					this->m_external_listed = false;
				}

				// The first process wrote the container: it gives the others their parts. The code may have changed the shape of
				// the container (init: a file that is read has a size that only the first process knows): the others then
				// get the new shape first.
				void external_end_write()
				{
					const bool changed_here = this->m_extents != this->m_extents_at_begin;
					int changed = changed_here ? 1 : 0;
					std::array<unsigned long long, 4> extents{{this->m_extents[0], this->m_extents[1], this->m_extents[2], this->m_extents[3]}};
					MPI_Bcast(&changed, 1, MPI_INT, 0, mpi::comm());
					MPI_Bcast(extents.data(), 4, MPI_UNSIGNED_LONG_LONG, 0, mpi::comm());
					this->m_external_listed = false;

					if (changed)
					{
						// (the first process has the new data in its gathered copy: it keeps it)
						std::unique_ptr<T[]> kept = std::move(this->m_mirror);
						this->reshape_from_extents({{(size_t)extents[0], (size_t)extents[1], (size_t)extents[2], (size_t)extents[3]}});
						this->m_mirror = std::move(kept);
					}

					if (this->size() > 0)
					{
						// Nothing moves now: the other processes get the container when something needs it (see m_root_only)
						if (this->local_size() > 0)
							this->m_tile->invalidateDeviceData();
						this->m_fill_pending = false;
						this->m_root_only = true;
						this->m_mirror_at_root = false;
						this->m_tile_valid = false;
						if (mpi::rank() != 0)
						{
							this->m_mirror.reset();
							this->m_mirror_valid = false;
						}
						else
							this->m_mirror_valid = true;
					}
					else
						this->tile_written();
				}

			protected:
				typedef std::array<size_t, 4> Extents;

				// What the traces call the container: Vector, Matrix, Tensor3 or Tensor4
				virtual const char *kind() const = 0;

				Distributed() : m_tile(new Tile()) {}
				virtual ~Distributed() { this->trace_deallocation(); }

				// The container gets the shape: the sizes of its dimensions (init of the derived class). Called on all processes, or in an
				// skepu::external call on the first process only
				virtual void reshape_from_extents(const Extents &extents) = 0;

				Distributed(const Distributed &other): m_label(other.m_label)
				{
					other.ensure_tile();
					this->m_dist = other.m_dist;
					if (other.m_identity.live)
						this->trace_allocation(other.m_identity.kind); // (the copy is a container of its own)
					// (a node-level container cannot be empty, so a process without elements has an empty tile of no shape)
					this->m_tile.reset(other.local_size() > 0 ? new Tile(*other.m_tile) : new Tile());
					this->m_tile->setIndexOrigin(this->local_first_offset());
					this->m_fill_pending = other.m_fill_pending;
					this->m_fill_value = other.m_fill_value;
				}

				Distributed(Distributed &&other) noexcept = default;

				Distributed &operator=(const Distributed &other)
				{
					if (this != &other)
					{
						Distributed copy(other);
						*this = std::move(copy);
					}
					return *this;
				}

				Distributed &operator=(Distributed &&other) noexcept
				{
					if (this != &other)
					{
						this->trace_deallocation();
						this->m_dist = std::move(other.m_dist);
						this->m_tile = std::move(other.m_tile);
						this->m_label = std::move(other.m_label);
						this->m_mirror = std::move(other.m_mirror);
						this->m_mirror_valid = other.m_mirror_valid;
						this->m_tile_valid = other.m_tile_valid;
						this->m_fill_pending = other.m_fill_pending;
						this->m_fill_value = other.m_fill_value;
						this->m_root_only = other.m_root_only;
						this->m_mirror_at_root = other.m_mirror_at_root;
						this->m_extents = other.m_extents;
						this->m_extents_at_begin = other.m_extents_at_begin;
						this->m_external_listed = other.m_external_listed;
						this->m_identity = std::move(other.m_identity);
					}
					return *this;
				}

				// A new shape: n indices of the first dimension, each of unit elements. The tile is then empty
				// (the derived class gives it its shape); the container is valid.
				void setup(size_t n, size_t unit, const Extents &extents)
				{
					this->trace_deallocation();
					this->m_extents = extents;
					this->m_dist = mpi::BlockDistribution{n, mpi::size(), unit};
					this->m_mirror.reset();
					this->m_mirror_valid = false;
					this->m_tile_valid = true;
					this->m_fill_pending = false;
					this->m_root_only = false;
					this->m_mirror_at_root = false;
					this->m_tile.reset(new Tile());
					this->trace_allocation(this->kind()); // (called from the constructor or init of the derived class)
				}

				void tile_ready()
				{
					this->m_tile->setIndexOrigin(this->local_first_offset());
				}

				// The elements were all set to the value (on every process: see m_fill_pending)
				void known_fill(const T &value)
				{
					this->m_fill_pending = true;
					this->m_fill_value = value;
				}

				// In an skepu::external call (on the first process) the container is initialized in its gathered copy: the
				// processes get their parts at the end of the call.
				bool in_external_call() const { return mpi::external_depth() > 0; }

				void init_in_external_call(const T &value)
				{
					if (!this->m_external_listed)
						mpi::fatal("A distributed container is initialized in skepu::external without being listed in skepu::write(...)");
					this->m_mirror.reset(this->size() > 0 ? new T[this->size()] : nullptr);
					std::fill(this->m_mirror.get(), this->m_mirror.get() + this->size(), value);
					this->m_mirror_valid = true;
					this->m_tile_valid = false;
					this->m_fill_pending = false;
					this->m_root_only = false;
					this->m_mirror_at_root = false;
				}

				Tile &tile() { return *this->m_tile; }

				// What the randomize functions of the node-level containers do: the elements get a value from rand(), one call for
				// each, in the order of the memory. Every process makes the calls for all the elements (so that the state of
				// rand() is that of a single node, on every process) and keeps the values of its own.
				template<typename F>
				void generate_from_rand(F value)
				{
					// (a call for the trace, as in a single node, where randomize is an external of the library)
					mpi::trace::CallScope trace_scope("external");
					trace_scope.work_begin();
					const size_t first = this->local_offset(), last = first + this->local_size();
					T *tile = this->local_size() > 0 ? this->m_tile->getAddress() : nullptr;
					const size_t total = this->size();
					for (size_t i = 0; i < total; ++i)
					{
						const int r = std::rand();
						if (i >= first && i < last)
							tile[i - first] = value(r);
					}
					if (this->local_size() > 0)
						this->m_tile->invalidateDeviceData();
					this->tile_written();
					trace_scope.work_finish();
					trace_scope.finish_external("randomize", "", 0, {this->m_identity.id}, {});
				}

				const T *read_access() const
				{
					this->ensure_mirror();
					return this->m_mirror.get();
				}

				// The host may write: the tile is then out of date
				T *write_access()
				{
					this->ensure_mirror();
					this->m_tile_valid = false;
					this->m_fill_pending = false;
					return this->m_mirror.get();
				}

			private:
				// The identity of the container in the traces: an id, and whether its allocation was recorded
				struct Identity
				{
					int id = ::skepu_node::UniqueIdentifier::generate();
					std::string kind, name;
					bool live = false;

					Identity() = default;
					Identity(Identity &&other) noexcept: id(other.id), kind(std::move(other.kind)), name(std::move(other.name)), live(other.live) { other.live = false; }
					Identity &operator=(Identity &&other) noexcept
					{
						this->id = other.id;
						this->kind = std::move(other.kind);
						this->name = std::move(other.name);
						this->live = other.live;
						other.live = false;
						return *this;
					}
				};

				void trace_allocation(const std::string &kind)
				{
					this->m_identity.kind = kind;
					this->m_identity.name = this->m_label.empty() ? kind + " #" + std::to_string(this->m_identity.id) : this->m_label;
					this->m_identity.live = true;
					mpi::trace::allocation(this->m_identity.id, this->m_identity.name, true);
				}

				void trace_deallocation()
				{
					if (!this->m_identity.live)
						return;
					this->m_identity.live = false;
					mpi::trace::allocation(this->m_identity.id, this->m_identity.name, false);
				}

				Identity m_identity;
				mpi::BlockDistribution m_dist;
				std::unique_ptr<Tile> m_tile;
				std::string m_label;

				// The container gathered on every process, for the host access
				mutable std::unique_ptr<T[]> m_mirror;
				mutable bool m_mirror_valid = false;
				mutable bool m_tile_valid = true;

				// Every element still has the value that the container was created with, and every process knows it: the gathered
				// copy is made on each process without communication (until something writes the container)
				bool m_fill_pending = false;
				T m_fill_value{};

				// The container was written by an skepu::external call, on the first process: only that process has it (in its gathered
				// copy). The others get it when it is needed: the tiles when a skeleton needs them, a copy of the whole when the host reads it.
				mutable bool m_root_only = false;

				// The gathered copy is on the first process only (the code of an skepu::external call is run there, and nothing else needs the
				// copy): m_mirror_valid is then true there and false on the others. A host read on all the processes gets it to them (a broadcast).
				mutable bool m_mirror_at_root = false;

				Extents m_extents{{0, 0, 0, 0}};
				mutable Extents m_extents_at_begin{{0, 0, 0, 0}};
				mutable bool m_external_listed = false;

				void ensure_mirror(const char *reason = "host access: the container is gathered") const
				{
					if (this->m_root_only && mpi::external_depth() == 0) // (the code of an external call is on the first process, which has it)
					{
						// The first process gives every process a copy; the tiles are then made from it, where they are needed
						if (mpi::rank() != 0)
							this->m_mirror.reset(new T[this->size()]);
						mpi::trace::CommScope comm("bcast", reason, {this->m_identity.id});
						const unsigned long long total = (unsigned long long)this->size() * sizeof(T) * (unsigned long long)(mpi::size() - 1);
						comm.bytes(mpi::rank() == 0 ? total : 0, mpi::rank() == 0 ? 0 : (unsigned long long)this->size() * sizeof(T));
						mpi::broadcast_from_root(this->m_mirror.get(), this->size());
						this->m_mirror_valid = true;
						this->m_root_only = false;
						return;
					}
					if (this->m_mirror_at_root && mpi::external_depth() == 0)
					{
						// The first process has the gathered copy: it gives it to the others
						if (mpi::rank() != 0)
							this->m_mirror.reset(new T[this->size()]);
						mpi::trace::CommScope comm("bcast", reason, {this->m_identity.id});
						const unsigned long long total = (unsigned long long)this->size() * sizeof(T) * (unsigned long long)(mpi::size() - 1);
						comm.bytes(mpi::rank() == 0 ? total : 0, mpi::rank() == 0 ? 0 : (unsigned long long)this->size() * sizeof(T));
						mpi::broadcast_from_root(this->m_mirror.get(), this->size());
						this->m_mirror_valid = true;
						this->m_mirror_at_root = false;
						return;
					}
					if (this->m_mirror_valid)
						return;
					if (mpi::external_depth() > 0)
						mpi::fatal("A distributed container is used in skepu::external without being listed in skepu::read(...) or skepu::write(...)");

					if (this->m_fill_pending && this->size() > 0)
					{
						// (nothing has been written: every process has the value, and makes its copy without communication)
						this->m_mirror.reset(new T[this->size()]);
						std::fill(this->m_mirror.get(), this->m_mirror.get() + this->size(), this->m_fill_value);
					}
					else if (this->size() > 0)
					{
						this->m_mirror.reset(new T[this->size()]);
						const T *mine = nullptr;
						if (this->local_size() > 0)
						{
							this->m_tile->flush();
							mine = this->m_tile->getAddress();
						}
						mpi::trace::CommScope comm("allgather", reason, {this->m_identity.id});
						comm.bytes((unsigned long long)this->local_size() * sizeof(T) * (unsigned long long)(mpi::size() - 1),
							(unsigned long long)(this->size() - this->local_size()) * sizeof(T));
						mpi::allgather(mine, this->m_mirror.get(), this->m_dist);
					}
					this->m_mirror_valid = true;
				}

				// What an skepu::external call begins with: the first process, which runs the code, gets the container (the others do not need it)
				void gather_for_external() const
				{
					if (this->size() == 0)
					{
						this->m_mirror_valid = true;
						return;
					}
					if (this->m_fill_pending)
					{
						this->ensure_mirror(); // (every process knows it: no communication)
						return;
					}
					if (mpi::rank() == 0)
						this->m_mirror.reset(new T[this->size()]);
					const T *mine = nullptr;
					if (this->local_size() > 0)
					{
						this->m_tile->flush();
						mine = this->m_tile->getAddress();
					}
					mpi::trace::CommScope comm("gather", "skepu::external: the container is gathered to the process that runs the code", {this->m_identity.id});
					const unsigned long long total = (unsigned long long)this->size() * sizeof(T), mine_bytes = (unsigned long long)this->local_size() * sizeof(T);
					comm.bytes(mpi::rank() == 0 ? 0 : mine_bytes, mpi::rank() == 0 ? total - mine_bytes : 0);
					mpi::gather_to_root(mine, this->m_mirror.get(), this->m_dist);
					this->m_mirror_at_root = true;
					this->m_mirror_valid = (mpi::rank() == 0);
				}

				// The tile from the gathered container, if the host has changed it
				void ensure_tile() const
				{
					if (this->m_root_only)
					{
						// The first process gives each process its part
						if (this->local_size() > 0)
							this->m_tile->invalidateDeviceData();
						mpi::trace::CommScope comm("scatter", "the container is given to its owners", {this->m_identity.id});
						const unsigned long long total = (unsigned long long)this->size() * sizeof(T), mine = (unsigned long long)this->local_size() * sizeof(T);
						comm.bytes(mpi::rank() == 0 ? total - mine : 0, mpi::rank() == 0 ? 0 : mine);
						mpi::scatter_from_root(this->m_mirror.get(), this->local_size() > 0 ? this->m_tile->getAddress() : nullptr, this->m_dist);
						this->m_root_only = false;
						this->m_tile_valid = true;
						// (the gathered copy of the first process goes: all the processes must agree on whether there is one, as a gather is collective)
						this->m_mirror.reset();
						this->m_mirror_valid = false;
						this->m_mirror_at_root = false;
						return;
					}
					if (this->m_tile_valid)
						return;
					if (this->local_size() > 0)
					{
						T *tile = this->m_tile->getAddress();
						std::copy(this->m_mirror.get() + this->local_offset(), this->m_mirror.get() + this->local_offset() + this->local_size(), tile);
						this->m_tile->invalidateDeviceData();
					}
					this->m_tile_valid = true;
				}
			};
		}
	}

	template<typename T>
	struct is_distributed_container : std::is_base_of<mpi::detail::DistributedTag, std::decay_t<T>> {};
}

// The stream operators of skepu-lib/io.hpp read a distributed container in an skepu::external call (a node-level trait: see common.hpp)
namespace skepu_node
{
	template<typename T>
	struct is_skepu_io_container<T, std::enable_if_t<::skepu::is_distributed_container<T>::value>> : std::true_type {};
}
