#pragma once
#ifndef SKEPU_STARPU_MATRIX_HPP
#define SKEPU_STARPU_MATRIX_HPP 1

#include <iomanip>
#include <iostream>

#include "../../common.hpp"
#include "iterator.hpp"
#include "partition.hpp"

namespace skepu
{

template<typename T>
class Matrix
{
public:
	typedef T value_type;
	typedef value_type & referenece;
	typedef value_type const & const_reference;
	typedef value_type * pointer;
	typedef size_t size_type;

	typedef MatrixIterator<T> iterator;
	typedef MatrixIterator<const T> const_iterator;

	typedef util::matrix_partition<T> partition_type;

private:
	partition_type m_data;

public:
	Matrix() noexcept : m_data() {}
	Matrix(Matrix const & other) noexcept : m_data(other.m_data) {}
	Matrix(Matrix && other) noexcept : m_data(std::move(other.m_data)) {}

	Matrix(size_type rows, size_type cols) noexcept
	{
		if(rows && cols)
			init(rows, cols, T());
	}

	Matrix(size_type rows, size_type cols, const_reference val) noexcept
	{
		if(rows && cols)
			init(rows, cols, val);
	}

	Matrix(std::initializer_list<T> const & l)
	{
		auto l_it = l.begin();
		m_data = partition_type(*l_it, *(l_it +1));
		l_it += 2;
		if((l.end() - l_it) != m_data.size())
		{
			std::cerr << "[skepu::Matrix] Initializer list not of the correct size\n";
			std::abort();
		}
		m_data.set(l_it, l.end());
	}

	Matrix(pointer ptr, size_t rows, size_t cols, bool deallocEnabled = true)
	{
		if(m_data.size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can only be initialized once!\n";
			std::abort();
		}
		if(!(rows && cols))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can not initialize without size\n";
			std::abort();
		}

		m_data.init(ptr, rows, cols, deallocEnabled);
	}

	~Matrix() noexcept = default;

	auto
	init(size_type rows, size_type cols) noexcept
	-> void
	{
		init(rows, cols, T());
	}

	auto
	init(size_type rows, size_type cols, const_reference val) noexcept
	-> void
	{
		auto size = rows * cols;
		if(m_data.size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can only be initialized once!\n";
			std::abort();
		}
		if(!(size))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can not initialize without size\n";
			std::abort();
		}

		m_data.init(rows, cols);
		m_data.fill(size, val);
	}

	auto
	getAddress() noexcept
	-> pointer
	{
		return m_data.data();
	}
	
	auto
	data() noexcept
	-> pointer
	{
		return m_data.data();
	}
	
	auto
	local_storage_handle() noexcept
	-> starpu_data_handle_t
	{
		return m_data.local_storage_handle();
	}

	auto
	size() const noexcept
	-> size_type
	{
		return m_data.size();
	}

	auto size_i() const noexcept -> size_type { return m_data.rows(); }
	auto size_j() const noexcept -> size_type { return m_data.cols(); }
	constexpr auto size_k() const noexcept -> size_type { return 0; }
	constexpr auto size_l() const noexcept -> size_type { return 0; }
	
	auto total_rows() const noexcept -> size_type { return m_data.rows(); }
	auto total_cols() const noexcept -> size_type { return m_data.cols(); }

	void change_layout()
	{
		m_data.flip_dim();
	}

	auto
	resize(size_type rows, size_type cols)
	-> void
	{
		partition_type tmp(rows,cols);
		for(size_t i(0); i < m_data.size(); ++i)
			tmp(i) = m_data(i);
		m_data = std::move(tmp);
	}

	auto
	resize(size_type const rows, size_type const cols, T val)
	-> void;

	auto
	operator=(Matrix<T> const & other)
	-> Matrix &
	{
		m_data = other.m_data;
		return *this;
	}

	auto
	operator=(Matrix && other)
	-> Matrix &
	{
		m_data = std::move(other.m_data);
		return *this;
	}

	auto
	set(size_type const row, size_type const  col, const_reference val)
	-> void
	{
		m_data.set(row, col, val);
	}

	Matrix<T>&
	subsection(
		size_type const row,
		size_type const col,
		size_type const rowWidth,
		size_type const colWidth);

	auto
	begin()
	-> iterator
	{
		return iterator(m_data);
	}

	auto
	begin() const
	-> const_iterator
	{
		return const_iterator(m_data);
	}

	auto
	begin(size_type row)
	-> iterator
	{
		return iterator(m_data.data() + (row * m_data.cols()), m_data);
	}

	auto
	begin(size_type row) const
	-> const_iterator
	{
		return iterator(m_data.data() + (row * m_data.cols()), m_data);
	}

	auto
	end()
	-> iterator
	{
		return iterator(m_data.data() + m_data.size(), m_data);
	}

	auto
	end() const
	-> const_iterator
	{
		return iterator(m_data.data() + m_data.size(), m_data);
	}

	auto
	end(size_type row)
	-> iterator
	{
		return iterator(m_data.data() + ((row +1)* m_data.cols()), m_data);
	}

	auto
	end(size_type row) const
	-> const_iterator
	{
		return iterator(m_data.data() + ((row +1)* m_data.cols()), m_data);
	}

	auto
	flush(FlushMode = FlushMode::Default)
	-> void
	{
		m_data.allgather();
	}

	auto
	empty() const
	-> bool;

	auto
	at(size_type const row, size_type const col)
	-> T&;

	auto
	row_back(size_type const row)
	-> size_type;

	auto
	row_back(size_type row) const
	-> const T&;

	auto
	row_front(size_type row)
	-> size_type;

	auto
	row_front(size_type row) const
	-> const T&;

	auto
	col_back(size_type col)
	-> T&;

	auto
	col_front(size_type col)
	-> T&;

	auto
	clear()
	-> void;

	auto
	erase( iterator loc )
	-> iterator;

	auto
	erase( iterator start, iterator end )
	-> iterator;

	auto
	swap(Matrix<T>& from)
	-> void;

	auto
	operator()(size_type const row, size_type const col)
	-> referenece
	{
		return m_data(row,col);
	}

	auto
	operator()(const size_type row, const size_type col) const
	-> const T&
	{
		return m_data(row,col);
	}

	auto
	transpose(BackendSpec const & = BackendSpec("CPU"))
	-> Matrix &
	{
		partition_type tmp(m_data.cols(), m_data.rows());
		m_data.transpose_to(tmp);
		m_data = std::move(tmp);

		return *this;
	}

	auto
	randomize(
		T const & min = 0,
		T const & max = std::numeric_limits<T>::max()) noexcept
	-> void
	{
		m_data.randomize(min, max);
	}

	void save(const std::string& filename);
	void load(
		const std::string& filename,
		size_type rowWidth,
		size_type numRows = 0);

private:
	friend cont;

	auto
	getParent()
	-> partition_type &
	{
		return m_data;
	}
};

template<typename T>
struct is_skepu_matrix<Matrix<T>> : std::true_type
{};

template<typename T>
struct is_skepu_matrix<Matrix<T>&> : std::true_type
{};

template<typename T>
struct is_skepu_matrix<Matrix<T> const&> : std::true_type
{};

template<typename T>
auto inline
operator<<(std::ostream & os, Matrix<T> const & m)
-> std::ostream &
{
	os << "skepu::Matrix (" << m.size_i() << "," << m.size_j() << ")\n";
	std::vector<int> w(m.size_j(), 0);
	for(int i(0); i < m.size_i(); ++i)
	{
		for(int j(0); j < m.size_j(); ++j)
		{
			std::stringstream ss;
			ss << m(i,j);
			w[j] =
				std::max<int>(ss.str().size(), w[j]);
		}
	}

	for(size_t i(0); i < m.size_i(); ++i)
	{
		os << std::setw(w[0]) << m(i,0);
		for(size_t j(1); j < m.size_j(); ++j)
			os << ", " << std::setw(w[j]) << m(i, j);
		os << "\n";
	}

	return os;
}

} // end namespace skepu

#endif // SKEPU_STARPU_MATRIX_HPP
