#pragma once

#include <initializer_list>
#include <iostream>

#include "skepu3/mpi/container.hpp"

namespace skepu
{
	/*!
	 *  A matrix distributed over the processes of the program, by blocks of rows (see mpi::detail::Distributed). The
	 *  interface is that of the single-node matrix.
	 */
	template<typename T>
	class Matrix : public mpi::detail::Distributed<T, ::skepu_node::Matrix<T>>
	{
		using Core = mpi::detail::Distributed<T, ::skepu_node::Matrix<T>>;

	public:
		using typename Core::size_type;

		Matrix() = default;
		Matrix(std::string label) { this->setLabel(label); }

		Matrix(size_type rows, size_type cols, const T &val = T()) { this->init(rows, cols, val); }
		Matrix(size_type rows, size_type cols, std::string label, const T &val = T()) { this->setLabel(label); this->init(rows, cols, val); }

		// The host data is the same on all processes. (The matrix does not take over the ownership of the data.)
		Matrix(T *const ptr, size_type rows, size_type cols, bool /*deallocEnabled*/ = true)
		{
			this->init(rows, cols);
			std::copy(ptr + this->local_offset(), ptr + this->local_offset() + this->local_size(), this->tile().getAddress());
		}

		Matrix(const Matrix &) = default;
		Matrix(Matrix &&) noexcept = default;
		Matrix &operator=(const Matrix &) = default;
		Matrix &operator=(Matrix &&) noexcept = default;

		void init(size_type rows, size_type cols) { this->init(rows, cols, T()); }

		void init(size_type rows, size_type cols, const T &val)
		{
			this->setup(rows, cols, {{rows, cols, 0, 0}});
			if (this->in_external_call())
				return this->init_in_external_call(val);
			if (this->local_size() > 0)
				this->tile().init(this->local_first_dimension(), cols, val);
			this->tile_ready();
		}

	protected:
		void reshape_from_extents(const typename Core::Extents &e) override { this->init(e[0], e[1]); }
		const char *kind() const override { return "Matrix"; }

	public:
		size_type total_rows() const { return this->distribution().n; }
		size_type total_cols() const { return this->distribution().unit; }
		size_type size_i() const { return this->total_rows(); }
		size_type size_j() const { return this->total_cols(); }

		T &operator()(size_type row, size_type col) { return this->write_access()[row * this->total_cols() + col]; }
		const T &operator()(size_type row, size_type col) const { return this->read_access()[row * this->total_cols() + col]; }
		T &operator()(size_type index) { return this->write_access()[index]; }
		const T &operator()(size_type index) const { return this->read_access()[index]; }
		T &operator()(const Index2D index) { return (*this)(index.row, index.col); }
		T &operator[](size_type index) { return this->write_access()[index]; }
		const T &operator[](size_type index) const { return this->read_access()[index]; }

		void randomize(int min = 0, int max = RAND_MAX)
		{
			// as the node-level matrix
			this->generate_from_rand([&](int r) { return (T)(r % (int)(max - min + 1) + min); });
		}

		void randomizeReal(double min = 0, double max = 1)
		{
			// as the node-level matrix
			this->generate_from_rand([&](int r) { return min + (T)((r % RAND_MAX) / (double)RAND_MAX) * (max - min); });
		}

		// The whole matrix, to be read, on every process (not a copy)
		::skepu_node::Matrix<T> replicated_view() const
		{
			return ::skepu_node::Matrix<T>(const_cast<T *>(this->replicated()), this->total_rows(), this->total_cols(), false);
		}

		friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix)
		{
			const T *data = matrix.read_access();
			os << "Matrix: (" << matrix.total_rows() << " X " << matrix.total_cols() << ")\n";
			for (size_type i = 0; i < matrix.size(); i++)
			{
				os << data[i] << " ";
				if ((i + 1) % matrix.total_cols() == 0)
					os << "\n";
			}
			return os;
		}
	};

	template<typename T>
	struct is_distributed_matrix : std::false_type {};

	template<typename T>
	struct is_distributed_matrix<Matrix<T>> : std::true_type {};
}
