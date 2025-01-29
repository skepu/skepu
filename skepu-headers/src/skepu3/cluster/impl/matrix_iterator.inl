#ifndef MATRIX_ITERATOR_INL
#define MATRIX_ITERATOR_INL

#include <skepu3/cluster/matrix_iterator.hpp>

#include <skepu3/cluster/matrix.hpp>
#include <skepu3/cluster/starpu_matrix_container.hpp>

namespace skepu
{
	template<typename T>
	MatrixIterator<T>
	::MatrixIterator(const Size2D & index, Matrix<T> & parent)
		: m_index { index },
		  m_parent { & parent }
	{}


	template<typename T>
	cluster::starpu_matrix_container<T> & MatrixIterator<T>
	::getData() const
	{
		return getParent().data();
	}


	template<typename T>
	size_t MatrixIterator<T>
	::getIndex() const
	{
		return m_index.row * getParent().total_cols() + m_index.col;
	}


	template<typename T>
	Index2D MatrixIterator<T>
	::getIndex2D() const
	{
		return m_index;
	}

	template<typename T>
	size_t MatrixIterator<T>
	::size() const
	{
		return getParent().size() - getIndex();
	}


	template<typename T>
	Matrix<T> & MatrixIterator<T>
	::getParent()
	{
		return *m_parent;
	}

	template<typename T>
	const Matrix<T> & MatrixIterator<T>
	::getParent() const
	{
		return *m_parent;
	}

	template<typename T>
	MatrixIterator<T> MatrixIterator<T>
	::begin() const
	{
		return *this;
	}


	template<typename T>
	T MatrixIterator<T>
	::operator()(const size_t index)
	{
		return m_parent->at(getIndex() + index);
	}



	template<typename T>
	T MatrixIterator<T>
	::operator()(const size_t row, const size_t col)
	{
		return m_parent->at(m_parent->total_cols()*row + col + getIndex());
	}


	template<typename T>
	T MatrixIterator<T>
	::operator*()
	{
		return m_parent->at(getIndex());
	}


	template<typename T>
	T MatrixIterator<T>
	::operator->()
	{
		return m_parent->at(getIndex());
	}


	// TODO: ändra till 2D index nedan!
	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>
	::operator+=(const ssize_t i)
	{
		m_index.col += i;
		return *this;
	}

	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>
	::operator+=(const Size2D & i)
	{
		m_index.col += i.col;
		m_index.row += i.row;
		return *this;
	}

	template<typename T>
	MatrixIterator<T> MatrixIterator<T>
	::operator+(const ssize_t i)
	{
		auto res = *this;
		res.m_index.col += i;
		return res;
	}

	template<typename T>
	MatrixIterator<T> MatrixIterator<T>
	::operator+(const Size2D & i)
	{
		auto res = *this;
		res.m_index.col += i.col;
		res.m_index.row += i.row;
		return res;
	}

	template<typename T>
	MatrixIterator<T> MatrixIterator<T>
	::operator-(const Size2D & i)
	{
		auto res = *this;
		res.m_index.col -= i.col;
		res.m_index.row -= i.row;
		return res;
	}

	template<typename T>
	MatrixIterator<T> MatrixIterator<T>
	::operator-(const ssize_t i)
	{
		auto res = *this;
		res.m_index.col -= i;
		return res;
	}

	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>
	::operator-=(const Size2D & i)
	{
		m_index.col -= i.col;
		m_index.row -= i.row;
		return *this;
	}



	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>
	::operator-=(const ssize_t i)
	{
		m_index.col -= i;
		return *this;
	}

	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>
	::operator++() // Prefix
	{
		m_index.row = m_index.row + ((m_index.col + 1) / m_parent->size2D().col);
		m_index.col = (m_index.col + 1) % m_parent->size2D().col;
		++m_index.col;
	}


	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>
	::operator--() // Prefix
	{
		m_index.row = m_index.row - (m_index.col == 0 ? 1 : 0);
		m_index.col = (m_index.col - 1) % m_parent->size2D().col;
	}


	template<typename T>
	MatrixIterator<T> MatrixIterator<T>
	::operator++(int) // Postfix
	{
		auto tmp = *this;
		++(*this);
		return tmp;
	}



	template<typename T>
	MatrixIterator<T> MatrixIterator<T>
	::operator--(int) // Postfix
	{
		auto tmp = *this;
		--(*this);
		return tmp;
	}


	/**
	 * @brief This operator takes two iterators into the same matrix and
	 * returns the 2D size difference between them.
	 *
	 * Why an operator? For fun! (also, sorry.)
	 *
	 * @param rhs
	 * @return Matrix
	 */
	template<typename T>
	Size2D MatrixIterator<T>
	::operator%(MatrixIterator<T>& rhs)
	{
		assert(m_parent == rhs.m_parent
		       && "These iterators have different parents");
		auto start = getIndex2D();
		auto end = rhs.getIndex2D();

		assert(start.row <= end.row &&
		       start.col < end.col &&
		       "Invalid iterators");

		Size2D res;
		res.row = end.row - start.row;
		res.col = end.col - start.col;
		res.i = res.col; // ┻━┻ ︵¯\(ツ)/ ︵ ┻━┻
		assert(res.row > 0 && res.col > 0);
		return res;
	}

	template<typename T>
	inline size_t MatrixIterator<T>
	::block_height_from(const Index2D & idx)
	{
		auto & t = *this;
		auto i = idx + getIndex2D();
		return getParent().block_height_from(i);
	}

	template<typename T>
	inline cluster::helpers::handle_cut MatrixIterator<T>
	::largest_cut_from(const Index2D & idx)
	{
		return getParent().largest_cut_from(idx + getIndex2D());
	}
} // skepu


#endif /* MATRIX_ITERATOR_INL */
