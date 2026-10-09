#pragma once

#include <iostream>

#include "skepu3/mpi/container.hpp"

namespace skepu
{
	/*!
	 *  A three-dimensional tensor distributed over the processes of the program, by blocks of the first dimension
	 *  (see mpi::detail::Distributed). The interface is that of the single-node tensor.
	 */
	template<typename T>
	class Tensor3 : public mpi::detail::Distributed<T, ::skepu_node::Tensor3<T>>
	{
		using Core = mpi::detail::Distributed<T, ::skepu_node::Tensor3<T>>;

	public:
		using typename Core::size_type;

		Tensor3() = default;

		explicit Tensor3(size_type si, size_type sj, size_type sk, std::string label = "", const T &val = T())
		{
			this->setLabel(label);
			this->init(si, sj, sk, val);
		}

		// The host data is the same on all processes. (The tensor does not take over the ownership of the data.)
		Tensor3(T *const ptr, size_type si, size_type sj, size_type sk, bool /*deallocEnabled*/ = true)
		{
			this->init(si, sj, sk);
			std::copy(ptr + this->local_offset(), ptr + this->local_offset() + this->local_size(), this->tile().getAddress());
		}

		Tensor3(const Tensor3 &) = default;
		Tensor3(Tensor3 &&) noexcept = default;
		Tensor3 &operator=(const Tensor3 &) = default;
		Tensor3 &operator=(Tensor3 &&) noexcept = default;

		void init(size_type si, size_type sj, size_type sk) { this->init(si, sj, sk, T()); }

		void init(size_type si, size_type sj, size_type sk, const T &val)
		{
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->setup(si, sj * sk, {{si, sj, sk, 0}});
			if (this->in_external_call())
				return this->init_in_external_call(val);
			if (this->local_size() > 0)
				this->tile().init(this->local_first_dimension(), sj, sk, val);
			this->tile_ready();
		}

	protected:
		void reshape_from_extents(const typename Core::Extents &e) override { this->init(e[0], e[1], e[2]); }
		const char *kind() const override { return "Tensor3"; }

	public:
		size_type size_i() const { return this->distribution().n; }
		size_type size_j() const { return this->m_size_j; }
		size_type size_k() const { return this->m_size_k; }

		T &operator()(size_type i, size_type j, size_type k) { return this->write_access()[(i * this->m_size_j + j) * this->m_size_k + k]; }
		const T &operator()(size_type i, size_type j, size_type k) const { return this->read_access()[(i * this->m_size_j + j) * this->m_size_k + k]; }
		T &operator()(size_type index) { return this->write_access()[index]; }
		const T &operator()(size_type index) const { return this->read_access()[index]; }
		T &operator[](size_type index) { return this->write_access()[index]; }
		const T &operator[](size_type index) const { return this->read_access()[index]; }

		void randomize(int min = 0, int max = RAND_MAX)
		{
			// as the node-level vector, which the tensors are
			this->generate_from_rand([&](int r) { return (T)(r % max + min); });
		}

		::skepu_node::Tensor3<T> replicated_view() const
		{
			return ::skepu_node::Tensor3<T>(const_cast<T *>(this->replicated()), this->size_i(), this->size_j(), this->size_k(), false);
		}

		friend std::ostream &operator<<(std::ostream &os, const Tensor3 &ten)
		{
			const T *data = ten.read_access();
			os << "Tensor3: (" << ten.size_i() << " x " << ten.size_j() << " x " << ten.size_k() << ")\n";
			for (size_t i = 0; i < ten.size(); ++i)
				os << data[i] << ((i + 1) % ten.size_k() == 0 ? "\n" : " ");
			return os;
		}

	private:
		size_type m_size_j = 0, m_size_k = 0;
	};

	/*!
	 *  A four-dimensional tensor, distributed by blocks of the first dimension.
	 */
	template<typename T>
	class Tensor4 : public mpi::detail::Distributed<T, ::skepu_node::Tensor4<T>>
	{
		using Core = mpi::detail::Distributed<T, ::skepu_node::Tensor4<T>>;

	public:
		using typename Core::size_type;

		Tensor4() = default;

		explicit Tensor4(size_type si, size_type sj, size_type sk, size_type sl, std::string label = "", const T &val = T())
		{
			this->setLabel(label);
			this->init(si, sj, sk, sl, val);
		}

		Tensor4(T *const ptr, size_type si, size_type sj, size_type sk, size_type sl, bool /*deallocEnabled*/ = true)
		{
			this->init(si, sj, sk, sl);
			std::copy(ptr + this->local_offset(), ptr + this->local_offset() + this->local_size(), this->tile().getAddress());
		}

		Tensor4(const Tensor4 &) = default;
		Tensor4(Tensor4 &&) noexcept = default;
		Tensor4 &operator=(const Tensor4 &) = default;
		Tensor4 &operator=(Tensor4 &&) noexcept = default;

		void init(size_type si, size_type sj, size_type sk, size_type sl) { this->init(si, sj, sk, sl, T()); }

		void init(size_type si, size_type sj, size_type sk, size_type sl, const T &val)
		{
			this->m_size_j = sj;
			this->m_size_k = sk;
			this->m_size_l = sl;
			this->setup(si, sj * sk * sl, {{si, sj, sk, sl}});
			if (this->in_external_call())
				return this->init_in_external_call(val);
			if (this->local_size() > 0)
				this->tile().init(this->local_first_dimension(), sj, sk, sl, val);
			this->tile_ready();
		}

	protected:
		void reshape_from_extents(const typename Core::Extents &e) override { this->init(e[0], e[1], e[2], e[3]); }
		const char *kind() const override { return "Tensor4"; }

	public:
		size_type size_i() const { return this->distribution().n; }
		size_type size_j() const { return this->m_size_j; }
		size_type size_k() const { return this->m_size_k; }
		size_type size_l() const { return this->m_size_l; }

		T &operator()(size_type i, size_type j, size_type k, size_type l)
		{
			return this->write_access()[((i * this->m_size_j + j) * this->m_size_k + k) * this->m_size_l + l];
		}
		const T &operator()(size_type i, size_type j, size_type k, size_type l) const
		{
			return this->read_access()[((i * this->m_size_j + j) * this->m_size_k + k) * this->m_size_l + l];
		}
		T &operator()(size_type index) { return this->write_access()[index]; }
		const T &operator()(size_type index) const { return this->read_access()[index]; }
		T &operator[](size_type index) { return this->write_access()[index]; }
		const T &operator[](size_type index) const { return this->read_access()[index]; }

		void randomize(int min = 0, int max = RAND_MAX)
		{
			// as the node-level vector, which the tensors are
			this->generate_from_rand([&](int r) { return (T)(r % max + min); });
		}

		::skepu_node::Tensor4<T> replicated_view() const
		{
			return ::skepu_node::Tensor4<T>(const_cast<T *>(this->replicated()), this->size_i(), this->size_j(), this->size_k(), this->size_l(), false);
		}

		friend std::ostream &operator<<(std::ostream &os, const Tensor4 &ten)
		{
			const T *data = ten.read_access();
			os << "Tensor4: (" << ten.size_i() << " x " << ten.size_j() << " x " << ten.size_k() << " x " << ten.size_l() << ")\n";
			for (size_t i = 0; i < ten.size(); ++i)
				os << data[i] << ((i + 1) % ten.size_l() == 0 ? "\n" : " ");
			return os;
		}

	private:
		size_type m_size_j = 0, m_size_k = 0, m_size_l = 0;
	};
}
