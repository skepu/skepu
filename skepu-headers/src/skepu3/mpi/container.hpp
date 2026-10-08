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

				void setLabel(std::string label) { this->m_label = label; }
				std::string getLabel() const { return this->m_label; }

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
					this->m_tile_valid = true;
					this->m_mirror_valid = false;
					this->m_mirror.reset();
				}

				// The whole container on every process, to be read (a random access argument of a skeleton)
				const T *replicated() const { return this->read_access(); }

				// -------------------- For skepu::external --------------------

				struct ExternalState { bool tile_valid; };

				ExternalState external_begin() const
				{
					this->ensure_mirror();
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
						if (this->local_size() > 0)
							this->m_tile->invalidateDeviceData();
						mpi::scatter_from_root(this->m_mirror.get(), this->local_size() > 0 ? this->m_tile->getAddress() : nullptr, this->m_dist);
					}
					this->tile_written();
				}

			protected:
				typedef std::array<size_t, 4> Extents;

				Distributed() : m_tile(new Tile()) {}
				virtual ~Distributed() = default;

				// The container gets the shape: the sizes of its dimensions (init of the derived class). Called on all processes, or in an
				// skepu::external call on the first process only
				virtual void reshape_from_extents(const Extents &extents) = 0;

				Distributed(const Distributed &other): m_label(other.m_label)
				{
					other.ensure_tile();
					this->m_dist = other.m_dist;
					// (a node-level container cannot be empty, so a process without elements has an empty tile of no shape)
					this->m_tile.reset(other.local_size() > 0 ? new Tile(*other.m_tile) : new Tile());
					this->m_tile->setIndexOrigin(this->local_first_offset());
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

				Distributed &operator=(Distributed &&other) noexcept = default;

				// A new shape: n indices of the first dimension, each of unit elements. The tile is then empty
				// (the derived class gives it its shape); the container is valid.
				void setup(size_t n, size_t unit, const Extents &extents)
				{
					this->m_extents = extents;
					this->m_dist = mpi::BlockDistribution{n, mpi::size(), unit};
					this->m_mirror.reset();
					this->m_mirror_valid = false;
					this->m_tile_valid = true;
					this->m_tile.reset(new Tile());
				}

				void tile_ready()
				{
					this->m_tile->setIndexOrigin(this->local_first_offset());
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
				}

				Tile &tile() { return *this->m_tile; }

				// What the randomize functions of the node-level containers do: the elements get a value from rand(), one call for
				// each, in the order of the memory. Every process makes the calls for all the elements (so that the state of
				// rand() is that of a single node, on every process) and keeps the values of its own.
				template<typename F>
				void generate_from_rand(F value)
				{
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
					return this->m_mirror.get();
				}

			private:
				mpi::BlockDistribution m_dist;
				std::unique_ptr<Tile> m_tile;
				std::string m_label;

				// The container gathered on every process, for the host access
				mutable std::unique_ptr<T[]> m_mirror;
				mutable bool m_mirror_valid = false;
				mutable bool m_tile_valid = true;

				Extents m_extents{{0, 0, 0, 0}};
				mutable Extents m_extents_at_begin{{0, 0, 0, 0}};
				mutable bool m_external_listed = false;

				void ensure_mirror() const
				{
					if (this->m_mirror_valid)
						return;
					if (mpi::external_depth() > 0)
						mpi::fatal("A distributed container is used in skepu::external without being listed in skepu::read(...) or skepu::write(...)");

					if (this->size() > 0)
					{
						this->m_mirror.reset(new T[this->size()]);
						const T *mine = nullptr;
						if (this->local_size() > 0)
						{
							this->m_tile->flush();
							mine = this->m_tile->getAddress();
						}
						mpi::allgather(mine, this->m_mirror.get(), this->m_dist);
					}
					this->m_mirror_valid = true;
				}

				// The tile from the gathered container, if the host has changed it
				void ensure_tile() const
				{
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
