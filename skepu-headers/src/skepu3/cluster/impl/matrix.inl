#ifndef MATRIX_INL
#define MATRIX_INL

#include <skepu3/cluster/starpu_matrix_container.hpp>
#include <skepu3/cluster/matrix.hpp>
#include <cstddef>
#include <memory>

namespace skepu
{
	template<typename T>
	Matrix<T>
	::Matrix() {}

	template<typename T>
	Matrix<T>
	::Matrix(const Size2D & size)
		: m_size { size },
		  m_offset { },
		  m_data {
			  std::make_shared<cluster::starpu_matrix_container<T>>(size.row,
			                                                        size.col)}
	{}

	template<typename T>
	Matrix<T>
	::Matrix(const Size2D & size, const std::vector<T>& vals)
		: Matrix(size)
	{
		// TODO: Make this non-stupid
		for(size_t i {}; i < vals.size(); ++i) {
			m_data->set(i, vals[i]);
		}
	}

	template<typename T>
	Matrix<T>
	::Matrix(const Size2D & size, std::vector<T>&& vals)
		: Matrix(size)
	{
		// TODO: Make this non-stupid
		for(size_t i {}; i < vals.size(); ++i) {
			m_data->set(i, vals[i]);
		}
	}


	template<typename T>
	Matrix<T>
	::Matrix(const size_t size)
		: Matrix(Size2D{1, size}) {}

	template<typename T>
	Matrix<T>
	::Matrix(const size_t size, const std::vector<T>& vals)
		: Matrix({1, size}, vals) {}


	template<typename T>
	Matrix<T>
	::Matrix(const size_t size, std::vector<T>&& vals)
		: Matrix({1, size}, std::forward<std::vector<T>>(vals)) {}

template<typename T>
Matrix<T>
::Matrix(size_t const rows, size_t const cols)
: Matrix{Size2D{rows, cols}}
{}


	template<typename T>
	Matrix<T>
	::Matrix(Matrix<T> const & copy)
		: m_size{copy.m_size},
		  m_offset{},
			m_data{
				std::make_shared<cluster::starpu_matrix_container<T>>(
					copy.total_rows(),
					copy.total_cols())}
	{
		// TODO: Make this non-stupid
		// Is it possible to only update the local copy on every node?
		// That would make the copy O(N/p) in time, if there is one block per p.
		for (size_t row {}; row < m_size.row; ++row) {
			for (size_t col {}; col < m_size.col; ++col) {
				m_data->set(row, col, copy.at(row,col));
			}
		}
	}

	template<typename T>
	size_t Matrix<T>
	::size() const
	{
		return m_size.row*m_size.col;
	}


	template<typename T>
	size_t Matrix<T>
	::total_rows() const
	{
		return m_size.row;
	}


	template<typename T>
	size_t Matrix<T>
	::total_cols() const
	{
		return m_size.col;
	}


	template<typename T>
	Matrix<T> Matrix<T>
	::subsection(size_t row, size_t col,
	             size_t rowWidth, size_t colWidth)
	{
		Matrix<T> res;
		res.m_size.row = rowWidth - row;
		res.m_size.col = colWidth - col;
		res.m_offset.row = m_offset.row;
		res.m_offset.col = m_offset.col;

		res.m_size.i = res.m_size.col;
		res.m_offset.i = res.m_offset.i;

		res.m_data = m_data;
	}

	template<typename T>
	Matrix<T> Matrix<T>
	::subsection()
	{
		Matrix<T> res;
		res.m_size.row = m_size.row;
		res.m_size.col = m_size.col;
		res.m_offset.row = m_offset.row;
		res.m_offset.col = m_offset.col;

		res.m_size.i = m_size.i;
		res.m_offset.i = m_offset.i;

		res.m_data = m_data;
	}


	template<typename T>
	size_t Matrix<T>
	::capacity() const
	{
		return size();
	}


	template<typename T>
	bool Matrix<T>
	::empty() const
	{
		return size() == 0;
	}


	template<typename T>
	T Matrix<T>
	::at(size_t index) const
	{
		return at(index/total_cols(),index%total_cols());
	}


	template<typename T>
	T Matrix<T>
	::at(size_t row, size_t col) const
	{
		assert(row < m_size.row && "Index out of bounds");
		assert(col < m_size.col && "Index out of bounds");
		return m_data->operator()(m_offset.row+row,m_offset.col+col);
	}


	template<typename T>
	T Matrix<T>
	::operator[](const size_t index)
	{
		return at(index);
	}


	template<typename T>
	T Matrix<T>
	::operator()(const Index2D index)
	{
		return at(index.row, index.col);
	}


	template<typename T>
	T Matrix<T>
	::operator()(const size_t index)
	{
		return at(index);
	}

	template<typename T>
	T Matrix<T>
	::operator()(const size_t & row, const size_t & col)
	{
		return at(row, col);
	}

	template<typename T>
	Matrix<T> &
	Matrix<T>::
	operator=(Matrix<T> const & other)
	{
		m_size = other.m_size;
		m_offset = other.m_offset;
		m_data =
			std::make_shared<cluster::starpu_matrix_container<T>>(
				m_size.row,
				m_size.col);

		for (size_t row {}; row < m_size.row; ++row) {
			for (size_t col {}; col < m_size.col; ++col) {
				(*m_data).set(row, col, other.at(row,col));
			}
		}

		return *this;
	}

	template<typename T>
	void Matrix<T>
	::set(size_t index, const T & data)
	{
		set(index/total_cols(),index%total_cols(), data);
	}

	template<typename T>
	void Matrix<T>
	::set(size_t row, size_t col, const T & value)
	{
		assert(row < m_size.row && "Index out of bounds");
		assert(col < m_size.col && "Index out of bounds");
		m_data->set(m_offset.row+row,m_offset.col+col, value);
	}

	template<typename T>
	void
	Matrix<T>::flush()
	{
		m_data->allgather();
	}

	template<typename T>
	void Matrix<T>
	::clear()
	{
		m_size = {};
		m_offset = {};
		m_data.clear();
	}


	template<typename T>
	void Matrix<T>
	::swap(Matrix<T> & from)
	{
		std::swap(from.m_size, m_size);
		std::swap(from.m_offset, m_offset);
		std::swap(from.m_data, m_data);
	}


	template<typename T>
	Matrix<T>& Matrix<T>
	::operator~()
	{
		assert(false && "Not implemented yet, sorry :(");
		return *this;
	}


	template<typename T>
	cluster::starpu_matrix_container<T> & Matrix<T>
	::data()
	{
		return *m_data;
	}


	template<typename T>
	const Offset2D & Matrix<T>
	::offset() const
	{
		return m_offset;
	}


	template<typename T>
	const Offset2D & Matrix<T>
	::size2D() const
	{
		return m_size;
	}

	template<typename T>
	MatrixIterator<T> Matrix<T>
	::begin()
	{
		return MatrixIterator<T>({}, *this);
	}

	template<typename T>
	MatrixIterator<T> Matrix<T>
	::end()
	{
		return MatrixIterator<T>(m_size, *this);
	}


	/**
	 * @brief Return the number of rows between the given index and
	 * the next block boundary
	 *
	 * @param idx
	 * @return size_t
	 */
	template<typename T>
	size_t Matrix<T>
	::block_height_from(const Index2D & idx)
	{
		assert(idx.row < m_size.row && "row index out of bounds");
		assert(idx.col < m_size.col && "col index out of bounds");
		return m_data->row_block_height(idx.row + m_offset.row);
	}

	/**
	 * @brief Return the number of rows between the given index and
	 * the next block boundary
	 *
	 * @param idx
	 * @return size_t
	 */
	template<typename T>
	size_t Matrix<T>
	::block_width_from(const Index2D & idx)
	{
		assert(idx.row < m_size.row && "row index out of bounds");
		assert(idx.col < m_size.col && "col index out of bounds");
		return m_data->col_block_width(idx.col + m_offset.col);
	}

	/**
	 * @brief Get the largest possible cut from the given `idx` to
	 * the closest block boundaries > `idx`.
	 *
	 * @param idx
	 * @return handle_cut
	 */
	template<typename T>
	cluster::helpers::handle_cut Matrix<T>
	::largest_cut_from(const Index2D & idx)
	{
		assert(idx.row < m_size.row && "row index out of bounds");
		assert(idx.col < m_size.col && "col index out of bounds");
		return m_data->largest_cut(idx + m_offset);
	}
}

#endif /* MATRIX_INL */
