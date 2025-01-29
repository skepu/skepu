#pragma once
#ifndef SKEPU_STARPU_TENSOR3_HPP
#define SKEPU_STARPU_TENSOR3_HPP 1

#include <iomanip>

#include "iterator.hpp"
#include "proxy.hpp"
#include "partition.hpp"

namespace skepu
{

/* TODO: Original container inherrits from Vector. */
template<typename T>
class Tensor3
{
public:
	typedef size_t size_type;
	typedef T value_type;
	typedef T & reference;
	typedef T const & const_reference;
	typedef T * pointer;
	typedef ptrdiff_t difference_type;

	typedef Iterator<T, util::tensor3_partition> iterator;
	typedef Iterator<const T, util::tensor3_partition> const_iterator;

	typedef util::tensor3_partition<T> partition_type;

private:
	partition_type m_partition;

public:
	Tensor3() noexcept : m_partition() {}

	Tensor3(Tensor3 const & other) noexcept
	: m_partition(other.m_partition)
	{}

	Tensor3(Tensor3 && other) noexcept
	: m_partition(std::move(other.m_partition))
	{}

	explicit
	Tensor3(size_type i, size_type j, size_type k)
	{
		if(i && j && k)
			init(i, j, k, T());
	}

	explicit
	Tensor3(size_type i, size_type j, size_type k, const_reference val)
	{
		if(i && j && k)
			init(i, j, k, val);
	}

	explicit
	Tensor3(
		pointer ptr,
		size_type i,
		size_type j,
		size_type k,
		bool deallocEnabled = true)
	{
		if(m_partition.size())
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can only be initialized once!\n";
			std::abort();
		}
		if(!(i && j && k))
		{
			if(!cluster::mpi_rank())
				std::cerr << "[SkePU][Vector] Error: "
					"Can not initialize without size\n";
			std::abort();
		}
		m_partition.init(ptr, i, j, k, deallocEnabled);
	}

	~Tensor3() noexcept = default;

	auto
	init(size_type i, size_type j, size_type k) noexcept
	-> void
	{
		init(i, j, k, T());
	}

	auto
	init(size_type i, size_type j, size_type k, const_reference val) noexcept
	-> void
	{
		auto size = i * j * k;
		if(m_partition.size())
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

		m_partition.init(i, j, k);
		m_partition.fill(size, val);
	}

	auto
	operator=(Tensor3 const & other) noexcept
	-> Tensor3 &
	{
		m_partition = other.m_partition;
		return *this;
	}

	auto
	operator=(Tensor3 && other) noexcept
	-> Tensor3 &
	{
		m_partition = std::move(other.m_partition);
		return *this;
	}

	auto
	operator()(size_type pos) noexcept
	-> reference
	{
		return m_partition(pos);
	}

	auto
	operator()(size_type pos) const noexcept
	-> const_reference
	{
		return m_partition(pos);
	}

	auto
	operator()(size_type i, size_type j, size_type k) noexcept
	-> reference
	{
		return m_partition(i, j, k);
	}

	auto
	operator()(size_type i, size_type j, size_type k) const noexcept
	-> const_reference
	{
		return m_partition(i, j, k);
	}

	auto
	getAddress() noexcept
	-> pointer
	{
		return m_partition.data();
	}

	auto
	front() noexcept
	-> reference
	{
		return m_partition(0);
	}

	auto
	front() const noexcept
	-> const_reference
	{
		return m_partition(0);
	}

	auto
	back() noexcept
	-> reference
	{
		return m_partition(m_partition.size() -1);
	}

	auto
	back() const noexcept
	-> const_reference
	{
		return m_partition(m_partition.size() -1);
	}

	auto
	begin() noexcept
	-> iterator
	{
		return iterator(m_partition);
	}

	auto
	begin() const noexcept
	-> const_iterator
	{
		return const_iterator(m_partition);
	}

	auto
	end() noexcept
	-> iterator
	{
		return iterator(m_partition.data() + m_partition.size(), m_partition);
	}

	auto
	end() const noexcept
	-> const_iterator
	{
		return const_iterator(m_partition.data + m_partition.size(), m_partition);
	}

	auto
	swap(Tensor3 & other) noexcept
	-> void
	{
		auto tmp = std::move(m_partition);
		m_partition = std::move(other.m_partition);
		other.m_partition = std::move(tmp);
	}

	/* Capacity */
	auto
	size() const noexcept
	-> size_type
	{
		return m_partition.size();
	}

	auto
	size_i() const noexcept
	-> size_type
	{
		return m_partition.size_i();
	}

	auto
	size_j() const noexcept
	-> size_type
	{
		return m_partition.size_j();
	}

	auto
	size_k() const noexcept
	-> size_type
	{
		return m_partition.size_k();
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
		return !m_partition.size();
	}

	auto
	resize(size_type i, size_type j, size_type k) noexcept
	-> void
	{
		resize(i, j, k, T());
	}

	auto
	resize(size_type i, size_type j, size_type k, const_reference val) noexcept
	-> void
	{
		auto tmp = partition_type(i, j, k);
		auto cpy_i = std::min(i, m_partition.size_i());
		auto cpy_j = std::min(j, m_partition.size_j());
		auto cpy_k = std::min(k, m_partition.size_k());

		for(size_t ii(0); ii < cpy_i; ++ii)
		{
			for(size_t ij(0); ij < cpy_j; ++ij)
			{
				for(size_t ik(0); ik < cpy_k; ++ik)
					tmp(ii, ij, ik) = m_partition(ii, ij, ik);
				for(size_t ik(cpy_k); ik < k; ++ik)
					tmp(ii,ij,ik) = val;
			}
			for(size_t ij(cpy_j); ij < j; ++ij)
				for(size_t ik(0); ik < k; ++ik)
					tmp(ii, ij, ik) = val;
		}
		for(size_t ii(cpy_i); ii < i; ++ii)
			for(size_t ij(cpy_j); ij < j; ++ij)
				for(size_t ik(0); ik < k; ++ik)
					tmp(ii, ij, ik) = val;

		m_partition = std::move(tmp);
	}

	/* Consistency */
	auto
	flush(FlushMode /* e */ = FlushMode::Default) noexcept
	-> void
	{
		m_partition.allgather();
	}

	auto
	randomize(
		int const & min = 0,
		int const & max = std::numeric_limits<T>::max()) noexcept
	-> void
	{
		m_partition.randomize(min, max);
	}

	auto
	randomize(
		float const & min = 0,
		float const & max = std::numeric_limits<T>::max()) noexcept
	-> void
	{
		m_partition.randomize(min, max);
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
	friend cont;

	auto
	getParent() noexcept
	-> partition_type &
	{
		return m_partition;
	}

	auto
	getParent() const noexcept
	-> partition_type const &
	{
		return m_partition;
	}
};

template<typename T>
struct is_skepu_tensor3<Tensor3<T>> : public std::true_type {};

template<typename T>
auto inline
operator<<(std::ostream & os, skepu::Tensor3<T> const & t) noexcept
-> std::ostream &
{
	std::cout << "skepu::Tensor3 "
		<< "(" << t.size_i()
		<< "," << t.size_j()
		<< "," << t.size_k()
		<< ") i(j,k)\n";

	std::vector<int> w(t.size_i() * t.size_k(), 0);
	for(int i(0); i < t.size_i(); ++i)
	{
		auto offset = i * t.size_k();
		for(int j(0); j < t.size_j(); ++j)
			for(int k(0); k < t.size_k(); ++k)
			{
				std::stringstream ss;
				ss << t(i,j,k);
				w[offset +k] =
					std::max<int>(ss.str().size(), w[offset +k]);
			}
	}

	for(size_t j(0); j < t.size_j(); ++j)
	{
		auto last_i = t.size_i() -1;
		for(size_t i(0); i < last_i; ++i)
		{
			auto offset = i * t.size_k();
			std::cout << std::setw(w[offset]) << t(i,j,0);
			for(size_t k(1); k < t.size_k(); ++k)
			{
				std::cout << ", " << std::setw(w[offset +k]) << t(i,j,k);
			}
			std::cout << "    ";
		}

		auto offset = last_i * t.size_k();
		std::cout << std::setw(w[offset]) << t(last_i,j,0);
		for(size_t k(1); k < t.size_k(); ++k)
		{
			std::cout << ", " << std::setw(w[offset +k]) << t(last_i,j,k);
		}
		std::cout << "\n";
	}

	return os;
}

} // namespace skepu

namespace std {

template<typename T>
auto
swap(skepu::Tensor3<T> & a, skepu::Tensor3<T> & b) noexcept
-> void
{
	return a.swap(b);
}

} // namespace std

#endif // SKEPU_STARPU_TENSOR3_HPP
