#pragma once
#ifndef SKEPU_STARPU_TENSOR3_ITERATOR_HPP
#define SKEPU_STARPU_TENSOR3_ITERATOR_HPP 1

#include <iterator>

#include "../../common.hpp"

namespace skepu {

template<typename T, template<typename>class P>
class Iterator
{
public:
	typedef std::random_access_iterator_tag iterator_category;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	typedef T value_type;
	typedef T & reference;
	typedef T * pointer;
	typedef typename std::conditional<std::is_const<T>::value,
			P<typename std::remove_const<T>::type> const,
			P<T>>::type
		partition_type;
	typedef typename partition_type::index_type index_type;

private:
	pointer m_begin;
	pointer m_data;
	partition_type * m_partition;

public:
	Iterator() noexcept
	: m_begin{0}, m_data{0}, m_partition{0}
	{}

	Iterator(Iterator const & other) noexcept
	: m_begin{other.m_begin}, m_data(other.m_data), m_partition(other.m_partition)
	{}

	Iterator(Iterator && other) noexcept
	: m_begin(std::move(other.m_begin)),
		m_data(std::move(other.m_data)),
		m_partition(std::move(other.m_partition))
	{}

	Iterator(partition_type & c) noexcept
	: m_begin(c.data()), m_data(c.data()), m_partition(&c)
	{}

	Iterator(pointer ptr, partition_type & c) noexcept
	: m_begin(c.data()), m_data(ptr), m_partition(&c)
	{}

	~Iterator() noexcept
	{}

	auto
	operator=(Iterator const & other) noexcept
	-> Iterator &
	{
		m_begin = other.m_begin;
		m_data = other.m_data;
		m_partition = other.m_partition;

		return *this;
	}

	auto
	operator=(Iterator && other) noexcept
	-> Iterator &
	{
		m_begin = std::move(other.m_begin);
		m_data = std::move(other.m_data);
		m_partition = std::move(other.m_partition);

		return *this;
	}

	auto
	operator==(Iterator const & other) const noexcept
	-> bool
	{
		return m_partition == other.m_partition
			&& m_data == other.m_data;
	}

	auto
	operator!=(Iterator const & other) const noexcept
	-> bool
	{
		return !(*this == other);
	}

	auto
	operator<(Iterator const & other) const noexcept
	-> bool
	{
		return m_data < other.m_data;
	}

	auto
	operator<=(Iterator const & other) const noexcept
	-> bool
	{
		return m_data <= other.m_data;
	}

	auto
	operator>(Iterator const & other) const noexcept
	-> bool
	{
		return m_data > other.m_data;
	}

	auto
	operator>=(Iterator const & other) const noexcept
	-> bool
	{
		return m_data >= other.m_data;
	}

	auto
	operator[](difference_type n) noexcept
	-> reference
	{
		return *(m_data + n);
	}

	auto
	operator*() noexcept
	-> reference
	{
		return *m_data;
	}

	auto
	operator->() noexcept
	-> pointer
	{
		return m_data;
	}

	auto
	operator++() noexcept
	-> Iterator &
	{
		++m_data;
		return *this;
	}

	auto
	operator++(int) noexcept
	-> Iterator
	{
		auto tmp = *this;
		++m_data;
		return tmp;
	}

	auto
	operator--() noexcept
	-> Iterator &
	{
		--m_data;
	}

	auto
	operator--(int) noexcept
	-> Iterator
	{
		auto tmp = *this;
		--m_data;
		return tmp;
	}

	auto
	operator+(difference_type n) const noexcept
	-> Iterator
	{
		return Iterator(*this) += n;
	}

	auto
	operator+=(difference_type n) noexcept
	-> Iterator &
	{
		m_data += n;
		return *this;
	}

	auto
	operator-(difference_type n) const noexcept
	-> Iterator
	{
		return Iterator(*this) -= n;
	}

	auto
	operator-=(difference_type n) noexcept
	-> Iterator &
	{
		m_data -= n;
		return *this;
	}

	auto
	operator-(Iterator const & other) const noexcept
	-> difference_type
	{
		return m_data - other.m_data;
	}

	auto
	getParent() noexcept
	-> partition_type &
	{
		return *m_partition;
	}

	auto
	index() const noexcept
	-> index_type
	{
		return m_partition->index(offset());
	}

	auto
	offset() const noexcept
	-> size_type
	{
		return m_data - m_begin;
	}
};

template<typename Iterator>
auto inline
operator+(
	typename Iterator::difference_type n,
	Iterator const & iterator) noexcept
-> Iterator
{
	return iterator + n;
}

} // namespace skepu

#endif // SKEPU_STARPU_TENSOR3_ITERATOR_HPP
