#pragma once

#include <initializer_list>
#include <iostream>

#include "skepu3/mpi/container.hpp"

namespace skepu
{
	/*!
	 *  A vector distributed over the processes of the program (see mpi::detail::Distributed). The interface is that of
	 *  the single-node vector.
	 */
	template<typename T>
	class Vector : public mpi::detail::Distributed<T, ::skepu_node::Vector<T>>
	{
		using Core = mpi::detail::Distributed<T, ::skepu_node::Vector<T>>;

	public:
		using typename Core::size_type;

		Vector() = default;

		explicit Vector(size_type num, const T &val = T()) { this->init(num, val); }
		Vector(size_type num, std::string label, const T &val = T()) { this->setLabel(label); this->init(num, val); }
		Vector(std::string label) { this->setLabel(label); }

		Vector(std::initializer_list<T> l)
		{
			this->init(l.size());
			std::copy(l.begin() + this->local_offset(), l.begin() + this->local_offset() + this->local_size(), this->tile().getAddress());
		}

		// The host data is the same on all processes. (The vector does not take over the ownership of the data.)
		Vector(T *const ptr, size_type num, bool /*deallocEnabled*/ = true)
		{
			this->init(num);
			std::copy(ptr + this->local_offset(), ptr + this->local_offset() + this->local_size(), this->tile().getAddress());
		}

		Vector(const Vector &) = default;
		Vector(Vector &&) noexcept = default;
		Vector &operator=(const Vector &) = default;
		Vector &operator=(Vector &&) noexcept = default;

		void init(size_type num) { this->init(num, T()); }

		size_type size_i() const { return this->size(); }

		// An element as a value to pass to a skeleton (a uniform argument): the element is gathered to every process if it is
		// not there
		const Scalar<T> elem(size_type index SKEPU_EXTERNAL_SOURCE_LOCATION_PARAM) const
		{
			return Scalar<T>(this->read_access()[index]);
		}

		void init(size_type num, const T &val)
		{
			this->setup(num, 1, {{num, 0, 0, 0}});
			if (this->in_external_call())
				return this->init_in_external_call(val);
			if (this->local_size() > 0)
				this->tile().init(this->local_size(), val);
			this->tile_ready();
		}

	protected:
		void reshape_from_extents(const typename Core::Extents &e) override { this->init(e[0]); }
		const char *kind() const override { return "Vector"; }

	public:
		T &operator()(size_type index) { return this->write_access()[index]; }
		const T &operator()(size_type index) const { return this->read_access()[index]; }
		T &operator[](size_type index) { return this->write_access()[index]; }
		const T &operator[](size_type index) const { return this->read_access()[index]; }

		void randomize(int min = 0, int max = RAND_MAX)
		{
			// as the node-level vector: rand() % max + min
			this->generate_from_rand([&](int r) { return (T)(r % max + min); });
		}

		// The whole vector, to be read, on every process (not a copy)
		::skepu_node::Vector<T> replicated_view() const
		{
			return ::skepu_node::Vector<T>(const_cast<T *>(this->replicated()), this->size(), false);
		}

		friend std::ostream &operator<<(std::ostream &output, const Vector &vec)
		{
			const T *data = vec.read_access();
			output << vec.size() << ".... ";
			for (size_t i = 0; i < vec.size(); ++i)
				output << data[i] << " ";
			return output;
		}
	};
}
