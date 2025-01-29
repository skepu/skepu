#pragma once
#ifndef SKEPU_STARPU_VECTOR_ITERATOR_HPP
#define SKEPU_STARPU_VECTOR_ITERATOR_HPP 1

#include <iterator>

#include "../../common.hpp"
#include "partition.hpp"

namespace skepu {

template<typename T>
class VectorIterator
{
public:
	typedef std::random_access_iterator_tag iterator_category;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	typedef T value_type;
	typedef T & reference;
	typedef T * pointer;
	typedef skepu::util::vector_partition<T> vector;

private:
	pointer m_begin;
	pointer m_data;
	vector * m_vector;

public:
	VectorIterator() noexcept
	: m_begin{0}, m_data{0}, m_vector{0}
	{}

	VectorIterator(VectorIterator const & other) noexcept
	: m_begin{other.m_begin}, m_data(other.m_data), m_vector(other.m_vector)
	{}

	VectorIterator(VectorIterator && other) noexcept
	: m_begin(std::move(other.m_begin)),
		m_data(std::move(other.m_data)),
		m_vector(std::move(other.m_vector))
	{}

	VectorIterator(vector & c) noexcept
	: m_begin(c.data()), m_data(c.data()), m_vector(&c)
	{}

	VectorIterator(pointer ptr, vector & c) noexcept
	: m_begin(c.data()), m_data(ptr), m_vector(&c)
	{}

	~VectorIterator() noexcept
	{}

	auto inline
	operator=(VectorIterator const & other) noexcept
	-> VectorIterator &
	{
		m_begin = other.m_begin;
		m_data = other.m_data;
		m_vector = other.m_vector;

		return *this;
	}

	auto inline
	operator=(VectorIterator && other) noexcept
	-> VectorIterator &
	{
		m_begin = std::move(other.m_begin);
		m_data = std::move(other.m_data);
		m_vector = std::move(other.m_vector);

		return *this;
	}

	auto inline
	operator==(VectorIterator const & other) const noexcept
	-> bool
	{
		return m_vector == other.m_vector
			&& m_data == other.m_data;
	}

	auto inline
	operator!=(VectorIterator const & other) const noexcept
	-> bool
	{
		return !(*this == other);
	}

	auto inline
	operator<(VectorIterator const & other) const noexcept
	-> bool
	{
		return m_data < other.m_data;
	}

	auto inline
	operator<=(VectorIterator const & other) const noexcept
	-> bool
	{
		return m_data <= other.m_data;
	}

	auto inline
	operator>(VectorIterator const & other) const noexcept
	-> bool
	{
		return m_data > other.m_data;
	}

	auto inline
	operator>=(VectorIterator const & other) const noexcept
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
	-> VectorIterator &
	{
		++m_data;
		return *this;
	}

	auto inline
	operator++(int) noexcept
	-> VectorIterator
	{
		auto tmp = *this;
		++m_data;
		return tmp;
	}

	auto inline
	operator--() noexcept
	-> VectorIterator &
	{
		--m_data;
	}

	auto inline
	operator--(int) noexcept
	-> VectorIterator
	{
		auto tmp = *this;
		--m_data;
		return tmp;
	}

	auto inline
	operator+(difference_type n) const noexcept
	-> VectorIterator
	{
		return VectorIterator(*this) += n;
	}

	auto inline
	operator+=(difference_type n) noexcept
	-> VectorIterator &
	{
		m_data += n;
		return *this;
	}

	auto inline
	operator-(difference_type n) const noexcept
	-> VectorIterator
	{
		return VectorIterator(*this) -= n;
	}

	auto inline
	operator-=(difference_type n) noexcept
	-> VectorIterator &
	{
		m_data -= n;
		return *this;
	}

	auto inline
	operator-(VectorIterator const & other) const noexcept
	-> difference_type
	{
		return m_data - other.m_data;
	}

	auto inline
	getParent() noexcept
	-> vector &
	{
		return *m_vector;
	}

	auto inline
	index() const noexcept
	-> Index1D
	{
		Index1D idx;
		idx.i = offset();
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
	typename VectorIterator<T>::difference_type n,
	VectorIterator<T> const & iterator) noexcept
-> VectorIterator<T>
{
	return iterator + n;
}

template<typename T>
struct is_skepu_iterator<VectorIterator<T>, T> : public std::true_type {};
} // namespace skepu

#endif // SKEPU_STARPU_VECTOR_ITERATOR_HPP
