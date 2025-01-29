#pragma once
#ifndef SKEPU_STARPU_MATRIX_ITERATOR_HPP
#define SKEPU_STARPU_MATRIX_ITERATOR_HPP 1

#include <iterator>

#include "../../common.hpp"
#include "partition.hpp"

namespace skepu {

template<typename T>
class MatrixIterator
{
public:
	typedef std::random_access_iterator_tag iterator_category;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	typedef T value_type;
	typedef T & reference;
	typedef T * pointer;
	typedef skepu::util::matrix_partition<T> matrix;

private:
	pointer m_begin;
	pointer m_data;
	matrix * m_matrix;

public:
	MatrixIterator() noexcept
	: m_begin{0}, m_data{0}, m_matrix{0}
	{}

	MatrixIterator(MatrixIterator const & other) noexcept
	: m_begin{other.m_begin}, m_data(other.m_data), m_matrix(other.m_matrix)
	{}

	MatrixIterator(MatrixIterator && other) noexcept
	: m_begin(std::move(other.m_begin)),
		m_data(std::move(other.m_data)),
		m_matrix(std::move(other.m_matrix))
	{}

	MatrixIterator(matrix & c) noexcept
	: m_begin(c.data()), m_data(c.data()), m_matrix(&c)
	{}

	MatrixIterator(pointer ptr, matrix & c) noexcept
	: m_begin(c.data()), m_data(ptr), m_matrix(&c)
	{}

	~MatrixIterator() noexcept
	{}

	auto inline
	operator=(MatrixIterator const & other) noexcept
	-> MatrixIterator &
	{
		m_begin = other.m_begin;
		m_data = other.m_data;
		m_matrix = other.m_matrix;

		return *this;
	}

	auto inline
	operator=(MatrixIterator && other) noexcept
	-> MatrixIterator &
	{
		m_begin = std::move(other.m_begin);
		m_data = std::move(other.m_data);
		m_matrix = std::move(other.m_matrix);

		return *this;
	}

	auto inline
	operator==(MatrixIterator const & other) const noexcept
	-> bool
	{
		return m_matrix == other.m_matrix
			&& m_data == other.m_data;
	}

	auto inline
	operator!=(MatrixIterator const & other) const noexcept
	-> bool
	{
		return !(*this == other);
	}

	auto inline
	operator<(MatrixIterator const & other) const noexcept
	-> bool
	{
		return m_data < other.m_data;
	}

	auto inline
	operator<=(MatrixIterator const & other) const noexcept
	-> bool
	{
		return m_data <= other.m_data;
	}

	auto inline
	operator>(MatrixIterator const & other) const noexcept
	-> bool
	{
		return m_data > other.m_data;
	}

	auto inline
	operator>=(MatrixIterator const & other) const noexcept
	-> bool
	{
		return m_data >= other.m_data;
	}

	auto inline
	operator[](difference_type n) noexcept
	-> reference
	{
		return *(m_data + n);
	}

	auto inline
	operator*() noexcept
	-> reference
	{
		return *m_data;
	}

	auto inline
	operator->() noexcept
	-> pointer
	{
		return m_data;
	}

	auto inline
	operator++() noexcept
	-> MatrixIterator &
	{
		++m_data;
		return *this;
	}

	auto inline
	operator++(int) noexcept
	-> MatrixIterator
	{
		auto tmp = *this;
		++m_data;
		return tmp;
	}

	auto inline
	operator--() noexcept
	-> MatrixIterator &
	{
		--m_data;
	}

	auto inline
	operator--(int) noexcept
	-> MatrixIterator
	{
		auto tmp = *this;
		--m_data;
		return tmp;
	}

	auto inline
	operator+(difference_type n) const noexcept
	-> MatrixIterator
	{
		return MatrixIterator(*this) += n;
	}

	auto inline
	operator+=(difference_type n) noexcept
	-> MatrixIterator &
	{
		m_data += n;
		return *this;
	}

	auto inline
	operator-(difference_type n) const noexcept
	-> MatrixIterator
	{
		return MatrixIterator(*this) -= n;
	}

	auto inline
	operator-=(difference_type n) noexcept
	-> MatrixIterator &
	{
		m_data -= n;
		return *this;
	}

	auto inline
	operator-(MatrixIterator const & other) const noexcept
	-> difference_type
	{
		return m_data - other.m_data;
	}

	auto inline
	getParent() noexcept
	-> matrix &
	{
		return *m_matrix;
	}

	auto inline
	index() const noexcept
	-> Index2D
	{
		auto pos = m_data - m_begin;
		Index2D idx;
		idx.row = pos / m_matrix->cols();
		idx.col = pos % m_matrix->cols();
		return idx;
	}

	auto inline
	offset() const noexcept
	-> size_type
	{
		return m_data - m_begin;
	}
};

template<typename T>
auto inline
operator+(
	typename MatrixIterator<T>::difference_type n,
	MatrixIterator<T> const & iterator) noexcept
-> MatrixIterator<T>
{
	return iterator + n;
}

} // namespace skepu

#endif // SKEPU_STARPU_MATRIX_ITERATOR_HPP
