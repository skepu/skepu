#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <skepu3/cluster/starpu_matrix_container.hpp>
#include <cstddef>
#include <memory>

namespace skepu
{
	template<typename T>
	class MatrixIterator;

	template<typename T>
	struct Mat;

	template<typename T>
	class Matrix
	{
	private:
		std::shared_ptr<cluster::starpu_matrix_container<T>> m_data;
		Size2D m_size;
		Offset2D m_offset;
	public:
		typedef MatrixIterator<T> iterator;
		typedef MatrixIterator<const T> const_iterator;
		typedef Mat<T> proxy_type;

		typedef std::vector<T> container_type;
		typedef typename std::vector<T>::iterator vector_iterator;
		typedef size_t size_type;
		typedef typename std::vector<T>::value_type value_type;
		typedef typename std::vector<T>::difference_type difference_type;
		typedef typename std::vector<T>::pointer pointer;
		typedef typename std::vector<T>::reference reference;
		typedef typename std::vector<T>::const_reference const_reference;

		Matrix();
		Matrix(const size_t size);
		//Matrix(const size_t size, const T& val);
		Matrix(const size_t size, const std::vector<T>& vals);
		Matrix(const size_t size, std::vector<T>&& vals);
		Matrix(const Size2D & size);
		Matrix(size_t const rows, size_t const cols);
		//Matrix(const Size2D & size, const T& val);
		Matrix(const Size2D & size, const std::vector<T>& vals);
		Matrix(const Size2D & size, std::vector<T>&& vals);
		Matrix(Matrix<T> const & copy);
		~Matrix() = default;

		const Matrix<T>& getParent() const;
		Matrix<T>& getParent();

		size_t size() const;
		size_t total_rows() const;
		size_t total_cols() const;
		Matrix<T> subsection();
		Matrix<T> subsection(size_t row, size_t col,
		                     size_t rowWidth, size_t colWidth);

		size_t block_height_from(const Index2D & idx);
		size_t block_width_from(const Index2D & idx);
		cluster::helpers::handle_cut largest_cut_from(const Index2D & idx);

		MatrixIterator<T> begin();
		MatrixIterator<T> end();

		size_t capacity() const;
		bool empty() const;
		T at(size_t index) const;
		T at(size_t row, size_t col) const;
		T operator[](const size_t index);
		T operator()(const Index2D index);
		T operator()(const size_t index);
		T operator()(const size_t & row, const size_t & col);
		Matrix & operator=(Matrix const & other);
		void set(size_t index, const T & data = {});
		void set(size_t row, size_t col, const T & data = {});

		void flush();
		void clear();

		void swap(Matrix<T>& from);
		Matrix<T>& operator~(); // transpose

		cluster::starpu_matrix_container<T> & data();
		const Offset2D & offset() const;
		const Size2D & size2D() const;
	};

template<typename T>
struct is_skepu_container<skepu::Matrix<T>> : std::true_type {};

}

#include <skepu3/cluster/impl/matrix.inl>

#endif /* MATRIX_HPP */
