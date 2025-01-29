#ifndef MATRIX_ITERATOR_HPP
#define MATRIX_ITERATOR_HPP

#include <skepu3/cluster/matrix.hpp>
#include <skepu3/cluster/index.hpp>
#include <skepu3/cluster/starpu_matrix_container.hpp>

namespace skepu
{
	template<typename T>
	class MatrixIterator
	{
		// TODO: Implement the entire iterator interface, not just the
		// parts we need.

		// The matrix iterator will iterate row wise, but the area of
		// computation it defines is bound to the rectangle formed between
		// the current position and the lower right corner of the matrix.

		// That is, it works normally for vectors, but weirdly for
		// matrices.
	private:
		Size2D m_index;
		Matrix<T> * m_parent;
	public:
		MatrixIterator(const Size2D & index, Matrix<T> & parent);
		cluster::starpu_matrix_container<T> & getData() const;
		size_t getIndex() const;
		Index2D getIndex2D() const;
		size_t size() const;
		const Matrix<T> & getParent() const;
		Matrix<T> & getParent();
		MatrixIterator<T> begin() const;
		T operator()(const size_t index);
		T operator()(const size_t row, size_t col);
		T operator*();
		T operator->();

		const MatrixIterator<T>& operator+=(const ssize_t i);
		const MatrixIterator<T>& operator-=(const ssize_t i);

		const MatrixIterator<T>& operator+=(const Size2D & i);
		const MatrixIterator<T>& operator-=(const Size2D & i);

		MatrixIterator<T> operator+(const ssize_t i);
		MatrixIterator<T> operator-(const ssize_t i);

		MatrixIterator<T> operator+(const Size2D & i);
		MatrixIterator<T> operator-(const Size2D & i);

		const MatrixIterator& operator++();
		MatrixIterator operator++(int);
		const MatrixIterator& operator--();
		MatrixIterator operator--(int);
		Size2D operator%(MatrixIterator<T>& rhs);

		inline size_t block_height_from(const Index2D & idx);
		inline cluster::helpers::handle_cut largest_cut_from(const Index2D & idx);
	};
}

template<typename T>
struct is_skepu_matrix_iterator: std::false_type {};

template<typename T>
struct is_skepu_matrix_iterator<skepu::MatrixIterator<T>>: std::true_type {};

#include <skepu3/cluster/impl/matrix_iterator.inl>

#endif /* MATRIX_ITERATOR_HPP */
