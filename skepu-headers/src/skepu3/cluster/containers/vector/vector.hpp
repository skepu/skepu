#pragma once
#ifndef SKEPU_STARPU_VECTOR_HPP
#define SKEPU_STARPU_VECTOR_HPP 1

#include <initializer_list>
#include <iostream>
#include <string>
#include <type_traits>

#include "../../common.hpp"
#include "iterator.hpp"
#include "partition.hpp"
#include "proxy.hpp"

namespace skepu {

template <typename T>
class Vector
{
public:
	typedef T value_type;
	typedef T * pointer;
	typedef T & reference;
	typedef T const & const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;

	typedef VectorIterator<value_type> iterator;
	typedef VectorIterator<const value_type> const_iterator;

	typedef Vec<value_type> proxy_type;
	typedef util::vector_partition<T> partition_type;

private:
	partition_type m_data;

public:
	Vector() noexcept
	: m_data{}
	{}

	Vector(Vector const & v) noexcept
	: m_data(v.m_data)
	{}

	Vector(Vector && v) noexcept
	: m_data(std::move(v.m_data))
	{}

	Vector(std::initializer_list<value_type> l) noexcept
	: m_data(l.size())
	{
		m_data.set(l.begin(), l.end());
	}

	Vector(size_type count) noexcept
	{
		if(count)
			init(count, T());
	}

	Vector(size_type count, const_reference val) noexcept
	{
		if(count)
			init(count, val);
	}

	Vector(pointer p, size_type count, bool deallocEnabled = true)
	{
		init(p, count, deallocEnabled);
	}

	~Vector() noexcept
	{}

	auto
	init(size_type count) noexcept
	-> void
	{
		init(count, T());
	}

	auto
	init(size_type count, const_reference val) noexcept
	-> void
	{
		if(m_data.size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can only be initialized once!\n";
			std::abort();
		}
		if(!count)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can not initialize without size\n";
			std::abort();
		}

		m_data.init(count);
		m_data.fill(count, val);
	}

	auto
	init(pointer p, size_type count, bool deallocEnabled) noexcept
	-> void
	{
		if(m_data.size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can only be initialized once!\n";
			std::abort();
		}
		if(!count)
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can not initialize without size\n";
			std::abort();
		}

		m_data.init(p, count, deallocEnabled);
	}

	auto
	operator=(Vector const & v) noexcept
	-> Vector<value_type> &
	{
		m_data = v.m_data;

		return *this;
	}

	auto
	operator=(Vector && v) noexcept
	-> Vector<value_type> &
	{
		m_data = std::move(v.m_data);

		return *this;
	}

	auto
	getAddress() noexcept
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

	/* Element access */
	auto
	operator()(size_type const pos) noexcept
	-> reference
	{
		return m_data(pos);
	}

	auto
	operator()(size_type const pos) const noexcept
	-> const_reference
	{
		return m_data(pos);
	}

	auto
	front() noexcept
	-> reference
	{
		return m_data(0);
	}

	auto
	front() const noexcept
	-> const_reference
	{
		return m_data(0);
	}

	auto
	back() noexcept
	-> reference
	{
		return m_data(m_data.size() -1);
	}

	auto
	back() const noexcept
	-> const_reference
	{
		return m_data(m_data.size() -1);
	}

	auto
	begin() noexcept
	-> iterator
	{
		return iterator(m_data);
	}

	auto
	begin() const noexcept
	-> const_iterator
	{
		return const_iterator(m_data);
	}

	auto
	end() noexcept
	-> iterator
	{
		return iterator(m_data.data() + m_data.size(), m_data);
	}

	auto
	end() const noexcept
	-> const_iterator
	{
		return const_iterator(m_data.data() + m_data.size(), m_data);
	}

	auto
	set(size_type pos, const_reference value) noexcept
	-> void
	{
		m_data.set(pos, value);
	}

	auto
	insert(iterator /* c */, size_type /* m */, const T& /* l */)
	-> void
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

	auto
	insert(iterator /* c */, const_reference /* l */)
	-> iterator
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

	auto
	erase(iterator /* c */)
	-> iterator
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

	auto
	erase(iterator /* t */, iterator /* d */)
	-> iterator
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

	auto
	clear()
	-> void
	{
		m_data.clear();
	}

	auto
	swap(Vector & other) noexcept
	-> void
	{
		auto tmp = std::move(m_data);
		m_data = std::move(other.m_data);
		other.m_data = std::move(tmp);
	}

	/* Capacity */
	auto
	size() const noexcept
	-> size_type
	{
		return m_data.size();
	}

	auto
	size_i() const noexcept
	-> size_type
	{
		return size();
	}

	auto constexpr
	size_j() const noexcept
	-> size_type
	{
		return 0;
	}

	auto constexpr
	size_k() const noexcept
	-> size_type
	{
		return 0;
	}

	auto constexpr
	size_l() const noexcept
	-> size_type
	{
		return 0;
	}

	auto
	empty() const noexcept
	-> bool
	{
		return !m_data.size();
	}

	auto
	capacity() const noexcept
	-> size_type
	{
		return m_data.capacity();
	}

	auto
	resize(size_type count) noexcept
	-> void
	{
		resize(count, T());
	}

	auto
	resize(size_type count, const_reference val) noexcept
	-> void
	{
		partition_type tmp(count);
		m_data.allgather();
		size_t end_pos = std::min(count, m_data.size());
		for(size_t i(0); i < end_pos; ++i)
			tmp.set(i, m_data(i));
		if(count < m_data.size())
			for(size_t i(m_data.size()); i < count; ++i)
				tmp.set(i, val);
		m_data = std::move(tmp);
	}

	/* Consistency */
	auto
	flush(FlushMode /* e */ = FlushMode::Default) noexcept
	-> void
	{
		m_data.allgather();
	}

	/* Utility functions */

	auto
	randomize(
		T const & min = 0,
		T const & max = std::numeric_limits<T>::max()) noexcept
	-> void
	{
		m_data.randomize(min, max);
	}

	auto
	save(const std::string& /* filename */, const std::string& /* delimiter */)
	-> void
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

	auto
	load(const std::string& /* e */, size_type /* s */)
	-> void
	{
		throw std::logic_error{std::string(__PRETTY_FUNCTION__)
			+ " Not implemented yet!\n"};
	}

private:
	//TODO: skepu::util should really be struct cont_helpers{...};
	friend cont;

	auto
	getParent() noexcept
	-> util::vector_partition<T> &
	{
		return m_data;
	}
};

template<typename T>
struct is_skepu_vector<Vector<T>> : std::true_type
{};

template<typename T>
struct is_skepu_vector<Vector<T>&> : std::true_type
{};

template<typename T>
struct is_skepu_vector<Vector<T> const&> : std::true_type
{};

template<typename T>
auto inline
operator<<(std::ostream & os, Vector<T> const & v)
-> std::ostream &
{
	os << "skepu::Vector (" << v.size() << ")\n";
	os << v(0);
	for(int i(1); i < v.size(); ++i)
		os << ", " << v(i);
	os << "\n";

	return os;
}

} // namespace skepu

namespace std {

template<typename T>
inline void
swap(skepu::Vector<T> & a, skepu::Vector<T> & b) noexcept
{
	a.swap(b);
}

} // namespace std

#endif // SKEPU_STARPU_VECTOR_HPP
