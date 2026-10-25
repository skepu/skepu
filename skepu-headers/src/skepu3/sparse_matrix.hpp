/*! \file sparse_matrix.hpp
 *  \brief The SparseMatrix container: a matrix in compressed sparse row (CSR) format.
 *
 *  A SparseMatrix is a *pattern* (the size, the row offsets and the column indices of the stored entries, which is immutable and can be
 *  shared by several matrices) and a Vector of the *values* of the stored entries. The column indices of a row are sorted and unique.
 *
 *  A skeleton can use a SparseMatrix as
 *   - a random access argument (a SparseMat proxy in the user function, to read the rows),
 *   - an element-wise argument or the output of a Map, MapReduce or Reduce: the stored values, with the index (row, column) of each;
 *     the inputs and the output share a pattern (an output that has none gets the pattern of the inputs). A Map can combine inputs of
 *     different patterns, by the union or the intersection of the patterns, if the skeleton is given a sparse policy,
 *   - the input of a row-wise reduction (Reduce or MapReduce with the mode RowWise), which gives a Vector with an element for each row.
 *
 *  (CPU and OpenMP backends so far.)
 */

#ifndef _SPARSE_MATRIX_H
#define _SPARSE_MATRIX_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace skepu
{

	// The type of the indices (of the columns, and the offsets of the rows) of a SparseMatrix
	using SparseIndex = int32_t;

	/*!
	 *  How a Map combines sparse inputs of different patterns. With SamePattern (the default) the patterns must be equal.
	 */
	enum class SparsePolicy
	{
		SamePattern,  // the patterns must be equal
		Union,        // the output has an entry wherever any input has one (a missing entry has the implicit zero as value)
		Intersection  // the output has an entry wherever all the inputs have one
	};


	/*!
	 *  The pattern of a SparseMatrix: its size and where the entries are. It does not change after it is made, and it is shared by the
	 *  matrices that have it.
	 */
	class SparsePattern
	{
	public:
		SparsePattern(size_t rows, size_t cols, std::vector<SparseIndex> row_offsets, std::vector<SparseIndex> col_indices)
		: m_rows(rows), m_cols(cols), m_row_offsets(std::move(row_offsets)), m_col_indices(std::move(col_indices)), m_id(next_id())
		{
			this->validate();
		}

		size_t rows() const { return this->m_rows; }
		size_t cols() const { return this->m_cols; }
		size_t nnz() const { return this->m_col_indices.size(); }
		uint64_t id() const { return this->m_id; }
		const std::vector<SparseIndex> &row_offsets() const { return this->m_row_offsets; }
		const std::vector<SparseIndex> &col_indices() const { return this->m_col_indices; }

		// true iff the two patterns are the same (the same object, or the same size and entries)
		bool equals(const SparsePattern &other) const
		{
			if (this == &other)
				return true;
			return this->m_rows == other.m_rows && this->m_cols == other.m_cols && this->m_row_offsets == other.m_row_offsets
				&& this->m_col_indices == other.m_col_indices;
		}

		// The pattern of the transpose, and for each of its entries the position of the entry in this pattern (made once)
		std::shared_ptr<const SparsePattern> transposed(std::vector<SparseIndex> &permutation) const
		{
			std::lock_guard<std::mutex> lock(this->m_mutex);
			if (!this->m_transposed)
			{
				std::vector<SparseIndex> offsets(this->m_cols + 1, 0), cols(this->nnz()), perm(this->nnz());
				for (size_t k = 0; k < this->nnz(); ++k)
					++offsets[this->m_col_indices[k] + 1];
				for (size_t c = 0; c < this->m_cols; ++c)
					offsets[c + 1] += offsets[c];
				std::vector<SparseIndex> next(offsets.begin(), offsets.end() - 1);
				for (size_t r = 0; r < this->m_rows; ++r)
					for (SparseIndex k = this->m_row_offsets[r]; k < this->m_row_offsets[r + 1]; ++k)
					{
						const SparseIndex place = next[this->m_col_indices[k]]++;
						cols[place] = (SparseIndex)r;
						perm[place] = k;
					}
				this->m_transposed = std::make_shared<const SparsePattern>(this->m_cols, this->m_rows, std::move(offsets), std::move(cols));
				this->m_permutation = std::move(perm);
			}
			permutation = this->m_permutation;
			return this->m_transposed;
		}

	private:
		static uint64_t next_id()
		{
			static std::atomic<uint64_t> counter{1};
			return counter++;
		}

		void validate() const
		{
			if (this->m_row_offsets.size() != this->m_rows + 1)
				SKEPU_ERROR("SparseMatrix: the row offsets must have " << this->m_rows + 1 << " elements, not " << this->m_row_offsets.size());
			if (this->m_row_offsets[0] != 0 || (size_t)this->m_row_offsets[this->m_rows] != this->m_col_indices.size())
				SKEPU_ERROR("SparseMatrix: the row offsets must start with 0 and end with the number of entries");
			for (size_t r = 0; r < this->m_rows; ++r)
			{
				if (this->m_row_offsets[r] > this->m_row_offsets[r + 1])
					SKEPU_ERROR("SparseMatrix: the row offsets must not decrease (row " << r << ")");
				for (SparseIndex k = this->m_row_offsets[r]; k < this->m_row_offsets[r + 1]; ++k)
				{
					if (this->m_col_indices[k] < 0 || (size_t)this->m_col_indices[k] >= this->m_cols)
						SKEPU_ERROR("SparseMatrix: the column " << this->m_col_indices[k] << " of row " << r << " is out of range");
					if (k > this->m_row_offsets[r] && this->m_col_indices[k] <= this->m_col_indices[k - 1])
						SKEPU_ERROR("SparseMatrix: the columns of a row must be sorted and unique (row " << r << ")");
				}
			}
		}

		size_t m_rows, m_cols;
		std::vector<SparseIndex> m_row_offsets, m_col_indices;
		uint64_t m_id;
		mutable std::mutex m_mutex;
		mutable std::shared_ptr<const SparsePattern> m_transposed;
		mutable std::vector<SparseIndex> m_permutation;
	};

	using SparsePatternPtr = std::shared_ptr<const SparsePattern>;


	/*!
	 *  The pattern that combines the patterns of several inputs, and where each entry of it is in each input (-1 if it is not)
	 */
	struct SparseCombination
	{
		SparsePatternPtr pattern;
		std::vector<std::vector<SparseIndex>> positions; // for each input, for each entry of the pattern
	};

	/*!
	 *  The symbolic phase of a combination of sparse inputs: the union or the intersection of the patterns, row by row (a merge of the
	 *  sorted columns). The result is kept for the next time that the same patterns are combined.
	 */
	inline std::shared_ptr<const SparseCombination> sparse_combine(SparsePolicy policy, const std::vector<SparsePatternPtr> &inputs)
	{
		static std::mutex mutex;
		static std::map<std::vector<uint64_t>, std::shared_ptr<const SparseCombination>> cache;

		std::vector<uint64_t> key;
		key.push_back((uint64_t)policy);
		for (const auto &p : inputs)
			key.push_back(p->id());
		{
			std::lock_guard<std::mutex> lock(mutex);
			auto found = cache.find(key);
			if (found != cache.end())
				return found->second;
		}

		for (const auto &p : inputs)
			if (p->rows() != inputs[0]->rows() || p->cols() != inputs[0]->cols())
				SKEPU_ERROR("Sparse inputs must have the same size (" << inputs[0]->rows() << " x " << inputs[0]->cols() << " and "
					<< p->rows() << " x " << p->cols() << ")");

		const size_t rows = inputs[0]->rows(), count = inputs.size();
		std::vector<SparseIndex> offsets(rows + 1, 0), cols;
		auto result = std::make_shared<SparseCombination>();
		result->positions.resize(count);

		for (size_t r = 0; r < rows; ++r)
		{
			std::vector<SparseIndex> head(count), end(count);
			for (size_t j = 0; j < count; ++j)
			{
				head[j] = inputs[j]->row_offsets()[r];
				end[j] = inputs[j]->row_offsets()[r + 1];
			}
			for (;;)
			{
				// the smallest column among the heads (for the intersection, a column that is in all of them)
				bool any = false, all = true;
				SparseIndex smallest = 0, largest = 0;
				for (size_t j = 0; j < count; ++j)
				{
					if (head[j] >= end[j]) { all = false; continue; }
					const SparseIndex c = inputs[j]->col_indices()[head[j]];
					if (!any) { smallest = largest = c; any = true; }
					smallest = std::min(smallest, c);
					largest = std::max(largest, c);
				}
				if (!any)
					break;
				if (policy == SparsePolicy::Intersection)
				{
					if (!all)
						break; // one of the rows has ended: no more columns in common
					if (smallest != largest)
					{
						// advance the ones that are behind
						for (size_t j = 0; j < count; ++j)
							if (inputs[j]->col_indices()[head[j]] == smallest)
								++head[j];
						continue;
					}
				}
				cols.push_back(smallest);
				for (size_t j = 0; j < count; ++j)
				{
					if (head[j] < end[j] && inputs[j]->col_indices()[head[j]] == smallest)
						result->positions[j].push_back(head[j]++);
					else
						result->positions[j].push_back(-1);
				}
			}
			offsets[r + 1] = (SparseIndex)cols.size();
		}
		result->pattern = std::make_shared<const SparsePattern>(rows, inputs[0]->cols(), std::move(offsets), std::move(cols));

		std::lock_guard<std::mutex> lock(mutex);
		if (cache.size() > 256)
			cache.clear();
		cache[key] = result;
		return result;
	}


	/*!
	 *  The proxy of a SparseMatrix in a user function: the rows can be read. The entries of row i are the positions from
	 *  row_offsets[i] to row_offsets[i + 1] - 1, and the entry at position k is in the column col_indices[k] and has the value data[k].
	 *  (The fields, and not member functions, since the precompiler makes a function of a member function that a user function calls.)
	 */
	template<typename T>
	struct SparseMat
	{
		T *data;
		const SparseIndex *row_offsets;
		const SparseIndex *col_indices;
		size_t rows;
		size_t cols;
		size_t count;

		T operator[](size_t index) const
		{
			return this->data[index];
		}
	};


	/*!
	 *  The part of a SparseMatrix that does not depend on the type of the values, to prepare the call of a skeleton.
	 */
	class SparseMatrixBase
	{
	public:
		virtual ~SparseMatrixBase() {}

		const SparsePatternPtr &pattern() const { return this->m_pattern; }
		bool has_pattern() const { return (bool)this->m_pattern; }

		// Gives the matrix a pattern (the values are zero)
		virtual void adopt_pattern(const SparsePatternPtr &pattern) = 0;

		// Makes the matrix have the pattern `target` for a while: the entries of `target` that are at `positions` in the pattern of this matrix
		// get their values (the others get `missing`). The returned function restores the matrix.
		virtual std::function<void()> align_to(const SparsePatternPtr &target, const std::vector<SparseIndex> &positions, double missing) = 0;

	protected:
		SparsePatternPtr m_pattern;
	};

	template<typename T>
	struct is_sparse_matrix: std::false_type {};

	template<typename T>
	class SparseMatrix;

	template<typename T>
	struct is_sparse_matrix<SparseMatrix<T>>: std::true_type {};

	template<typename T>
	class SparseMatrixIterator;


	/*!
	 *  \class SparseMatrix
	 *
	 *  \brief A sparse matrix in compressed sparse row format: a shared immutable pattern and a Vector of the values of the entries.
	 */
	template<typename T>
	class SparseMatrix: public SparseMatrixBase
	{
	public:
		typedef SparseMatrixIterator<T> iterator;
		typedef SparseMat<T> proxy_type;
		typedef T value_type;
		typedef ptrdiff_t difference_type;

		// ----- Constructors -----

		// A matrix with no pattern: a skeleton that has it as its output gives it one
		SparseMatrix() {}

		// A matrix with the given pattern (shared), and zero as the values
		explicit SparseMatrix(const SparsePatternPtr &pattern): m_values(make_values(pattern->nnz()))
		{
			this->m_pattern = pattern;
		}

		// A matrix from CSR arrays (copied)
		SparseMatrix(size_t rows, size_t cols, const std::vector<SparseIndex> &row_offsets, const std::vector<SparseIndex> &col_indices, const std::vector<T> &values)
		: m_values(make_values(values.size()))
		{
			if (values.size() != col_indices.size())
				SKEPU_ERROR("SparseMatrix: " << values.size() << " values for " << col_indices.size() << " column indices");
			this->m_pattern = std::make_shared<const SparsePattern>(rows, cols, row_offsets, col_indices);
			if (!values.empty())
				std::copy(values.begin(), values.end(), this->m_values.getAddress());
		}

		SparseMatrix(size_t rows, size_t cols, size_t nnz, const SparseIndex *row_offsets, const SparseIndex *col_indices, const T *values)
		: SparseMatrix(rows, cols, std::vector<SparseIndex>(row_offsets, row_offsets + rows + 1), std::vector<SparseIndex>(col_indices, col_indices + nnz),
			std::vector<T>(values, values + nnz))
		{}

		// The pattern is shared, the values are copied
		SparseMatrix(const SparseMatrix &other): m_values(other.m_values)
		{
			this->m_pattern = other.m_pattern;
		}

		SparseMatrix(SparseMatrix &&other): m_values(std::move(other.m_values))
		{
			this->m_pattern = std::move(other.m_pattern);
		}

		SparseMatrix &operator=(const SparseMatrix &other)
		{
			if (this != &other)
			{
				this->m_pattern = other.m_pattern;
				this->m_values = other.m_values;
			}
			return *this;
		}

		SparseMatrix &operator=(SparseMatrix &&other)
		{
			this->m_pattern = std::move(other.m_pattern);
			this->m_values = std::move(other.m_values);
			return *this;
		}

		// The nonzero elements of a dense matrix
		static SparseMatrix from_dense(Matrix<T> &dense, T zero = T() SKEPU_CALL_SITE_PARAM)
		{
			dense.updateHost();
			std::vector<SparseIndex> offsets(1, 0), cols;
			std::vector<T> values;
			for (size_t r = 0; r < dense.size_i(); ++r)
			{
				for (size_t c = 0; c < dense.size_j(); ++c)
					if (!(dense(r, c) == zero))
					{
						cols.push_back((SparseIndex)c);
						values.push_back(dense(r, c));
					}
				offsets.push_back((SparseIndex)cols.size());
			}
			SKEPU_TRACE_START_EVENT(trace_handle);
			SparseMatrix result(dense.size_i(), dense.size_j(), offsets, cols, values);
			if (result.nnz() > 0)
				SKEPU_TRACE_EXTERNAL(trace_handle, "SparseMatrix::from_dense", skepu_site_file, skepu_site_line, {result.m_values.getObjectID()}, {dense.getObjectID()});
			return result;
		}

		// From (row, column, value) triplets in any order (the values of a repeated place are added)
		static SparseMatrix from_coo(size_t rows, size_t cols, std::vector<std::tuple<size_t, size_t, T>> entries)
		{
			std::stable_sort(entries.begin(), entries.end(), [](const std::tuple<size_t, size_t, T> &a, const std::tuple<size_t, size_t, T> &b) {
				return std::make_pair(std::get<0>(a), std::get<1>(a)) < std::make_pair(std::get<0>(b), std::get<1>(b));
			});
			std::vector<SparseIndex> offsets(rows + 1, 0), columns;
			std::vector<T> values;
			size_t previous_row = 0;
			for (size_t e = 0; e < entries.size(); ++e)
			{
				const size_t r = std::get<0>(entries[e]), c = std::get<1>(entries[e]);
				if (r >= rows || c >= cols)
					SKEPU_ERROR("SparseMatrix: the entry (" << r << ", " << c << ") is outside the " << rows << " x " << cols << " matrix");
				if (e > 0 && r == previous_row && (size_t)columns.back() == c)
				{
					values.back() = values.back() + std::get<2>(entries[e]);
					continue;
				}
				columns.push_back((SparseIndex)c);
				values.push_back(std::get<2>(entries[e]));
				++offsets[r + 1];
				previous_row = r;
			}
			for (size_t r = 0; r < rows; ++r)
				offsets[r + 1] += offsets[r];
			return SparseMatrix(rows, cols, offsets, columns, values);
		}

		// From a MatrixMarket file (coordinate format, real or integer or pattern values, general or symmetric)
		static SparseMatrix read_matrix_market(const std::string &filename)
		{
			std::ifstream file(filename);
			if (!file)
				SKEPU_ERROR("SparseMatrix: cannot open " << filename);
			std::string line;
			std::getline(file, line);
			if (line.rfind("%%MatrixMarket", 0) != 0 || line.find("coordinate") == std::string::npos)
				SKEPU_ERROR("SparseMatrix: " << filename << " is not a MatrixMarket file in coordinate format");
			const bool symmetric = line.find("symmetric") != std::string::npos;
			const bool pattern_only = line.find("pattern") != std::string::npos;
			while (std::getline(file, line))
				if (!line.empty() && line[0] != '%')
					break;
			size_t rows, cols, count;
			std::istringstream(line) >> rows >> cols >> count;
			std::vector<std::tuple<size_t, size_t, T>> entries;
			for (size_t e = 0; e < count; ++e)
			{
				size_t r, c;
				double v = 1;
				file >> r >> c;
				if (!pattern_only)
					file >> v;
				entries.emplace_back(r - 1, c - 1, (T)v);
				if (symmetric && r != c)
					entries.emplace_back(c - 1, r - 1, (T)v);
			}
			return from_coo(rows, cols, entries);
		}

		// ----- Size and data -----

		size_t rows() const { return this->m_pattern ? this->m_pattern->rows() : 0; }
		size_t cols() const { return this->m_pattern ? this->m_pattern->cols() : 0; }
		size_t nnz() const { return this->m_pattern ? this->m_pattern->nnz() : 0; }
		size_t size() const { return this->nnz(); }

		// (the names of the earlier container)
		size_t total_rows() const { return this->rows(); }
		size_t total_cols() const { return this->cols(); }
		size_t total_nnz() const { return this->nnz(); }

		// The values of the entries, as a Vector: skeletons and the BLAS routines of vectors work on them
		Vector<T> &values() { return this->m_values; }
		const Vector<T> &values() const { return this->m_values; }

		// An element, as a copy (zero for a place that is not an entry)
		T operator()(size_t row, size_t col) const
		{
			if (!this->m_pattern || row >= this->rows() || col >= this->cols())
				SKEPU_ERROR("SparseMatrix: the element (" << row << ", " << col << ") is outside the matrix");
			this->m_values.updateHost();
			const auto &cols_of = this->m_pattern->col_indices();
			const auto begin = cols_of.begin() + this->m_pattern->row_offsets()[row], end = cols_of.begin() + this->m_pattern->row_offsets()[row + 1];
			const auto found = std::lower_bound(begin, end, (SparseIndex)col);
			if (found == end || *found != (SparseIndex)col)
				return T();
			return this->m_values.getAddress()[found - cols_of.begin()];
		}

		// The matrix as a dense Matrix
		Matrix<T> to_dense(SKEPU_CALL_SITE_PARAM_ONLY) const
		{
			this->m_values.updateHost();
			Matrix<T> dense(this->rows(), this->cols(), T());
			SKEPU_TRACE_START_EVENT(trace_handle);
			const T *data = this->m_values.getAddress();
			for (size_t r = 0; r < this->rows(); ++r)
				for (SparseIndex k = this->m_pattern->row_offsets()[r]; k < this->m_pattern->row_offsets()[r + 1]; ++k)
					dense(r, this->m_pattern->col_indices()[k]) = data[k];
			SKEPU_TRACE_EXTERNAL(trace_handle, "SparseMatrix::to_dense", skepu_site_file, skepu_site_line, {dense.getObjectID()}, {this->m_values.getObjectID()});
			return dense;
		}

		// The transpose, as a new matrix (the pattern of the transpose is made once for a pattern)
		SparseMatrix<T> transposed(SKEPU_CALL_SITE_PARAM_ONLY) const
		{
			if (!this->m_pattern)
				SKEPU_ERROR("SparseMatrix: the matrix has no pattern");
			std::vector<SparseIndex> permutation;
			SparsePatternPtr transposed_pattern = this->m_pattern->transposed(permutation);
			SparseMatrix<T> result(transposed_pattern);
			this->m_values.updateHost();
			SKEPU_TRACE_START_EVENT(trace_handle);
			const T *from = this->m_values.getAddress();
			T *to = result.m_values.getAddress();
			for (size_t k = 0; k < permutation.size(); ++k)
				to[k] = from[permutation[k]];
			SKEPU_TRACE_EXTERNAL(trace_handle, "SparseMatrix::transposed", skepu_site_file, skepu_site_line, {result.m_values.getObjectID()}, {this->m_values.getObjectID()});
			return result;
		}

		void setLabel(std::string label) { this->m_values.setLabel(label); }
		std::string getLabel() const { return this->m_values.getLabel(); }
		UniqueIdentifier::ID getObjectID() const { return this->m_values.getObjectID(); }

		// ----- What a skeleton needs of a container -----

		SparseMatrix<T> &getParent() { return *this; }
		const SparseMatrix<T> &getParent() const { return *this; }
		size_t linearIndexOrigin() const { return 0; }

		void updateHost(bool = true) const { this->m_values.updateHost(); }
		void invalidateDeviceData(bool = true) const { this->m_values.invalidateDeviceData(); }
		void updateHostAndInvalidateDevice() { this->m_values.updateHostAndInvalidateDevice(); }

		iterator begin();
		iterator stridedBegin(size_t, int) { return this->begin(); }

		template<typename Ignore>
		SparseMat<T> hostProxy(ProxyTag::Default, Ignore)
		{
			return this->hostProxy();
		}

		SparseMat<T> hostProxy()
		{
			SparseMat<T> proxy;
			proxy.data = this->m_values.getAddress();
			proxy.row_offsets = this->m_pattern ? this->m_pattern->row_offsets().data() : nullptr;
			proxy.col_indices = this->m_pattern ? this->m_pattern->col_indices().data() : nullptr;
			proxy.rows = this->rows();
			proxy.cols = this->cols();
			proxy.count = this->nnz();
			return proxy;
		}

		// ----- Preparing a skeleton call -----

		void adopt_pattern(const SparsePatternPtr &pattern) override
		{
			this->m_pattern = pattern;
			this->m_values = make_values(pattern->nnz());
		}

		std::function<void()> align_to(const SparsePatternPtr &target, const std::vector<SparseIndex> &positions, double missing) override
		{
			this->m_values.updateHost();
			Vector<T> aligned = make_values(target->nnz());
			SKEPU_TRACE_START_EVENT(trace_handle);
			const T *from = this->m_values.getAddress();
			T *to = aligned.getAddress();
			for (size_t k = 0; k < positions.size(); ++k)
				to[k] = (positions[k] >= 0) ? from[positions[k]] : (T)missing;
			SKEPU_TRACE_EXTERNAL(trace_handle, "SparseMatrix: aligned to the pattern of the output", "", 0, {aligned.getObjectID()}, {this->m_values.getObjectID()});
			SparsePatternPtr original_pattern = this->m_pattern;
			Vector<T> original_values;
			original_values.swap(this->m_values);
			this->m_values.swap(aligned);
			this->m_pattern = target;
			return [this, original_pattern, original_values = std::make_shared<Vector<T>>(std::move(original_values))]() mutable
			{
				this->m_pattern = original_pattern;
				this->m_values.swap(*original_values);
			};
		}

	private:
		// (a Vector has at least one element)
		static Vector<T> make_values(size_t count)
		{
			return count > 0 ? Vector<T>(count) : Vector<T>();
		}

		mutable Vector<T> m_values;
	};


	/*!
	 *  \class SparseMatrixIterator
	 *
	 *  \brief Traverses the stored values of a SparseMatrix, in the order of the rows. The index of an element is its (row, column).
	 */
	template<typename T>
	class SparseMatrixIterator
	{
	public:
		typedef SparseMatrixIterator<T> iterator;
		typedef SparseMatrix<T> parent_type;

		SparseMatrixIterator(parent_type *parent, size_t position)
		: m_parent(parent), m_position(position), m_values(parent->values().getAddress()),
		  m_row_offsets(parent->pattern() ? parent->pattern()->row_offsets().data() : nullptr),
		  m_col_indices(parent->pattern() ? parent->pattern()->col_indices().data() : nullptr), m_rows(parent->rows())
		{
			this->m_row = this->find_row(position);
		}

		Index2D getIndex() const
		{
			return Index2D{this->m_row, (size_t)this->m_col_indices[this->m_position]};
		}

		parent_type &getParent() const { return *this->m_parent; }
		iterator &begin() { return *this; }
		size_t size() const { return this->m_parent->nnz() - this->m_position; }
		T *getAddress() const { return this->m_values + this->m_position; }
		T *data() { return this->m_values; }

		T &operator()(const ssize_t index = 0) { return this->m_values[this->m_position + index]; }
		const T &operator()(const ssize_t index = 0) const { return this->m_values[this->m_position + index]; }
		T &operator[](const ssize_t index) { return this->m_values[this->m_position + index]; }
		const T &operator[](const ssize_t index) const { return this->m_values[this->m_position + index]; }

		bool operator==(const iterator &i) const { return this->m_position == i.m_position; }
		bool operator!=(const iterator &i) const { return this->m_position != i.m_position; }
		bool operator<(const iterator &i) const { return this->m_position < i.m_position; }
		bool operator>(const iterator &i) const { return this->m_position > i.m_position; }
		bool operator<=(const iterator &i) const { return this->m_position <= i.m_position; }
		bool operator>=(const iterator &i) const { return this->m_position >= i.m_position; }

		const iterator &operator++()
		{
			++this->m_position;
			while (this->m_row + 1 < this->m_rows && (SparseIndex)this->m_position >= this->m_row_offsets[this->m_row + 1])
				++this->m_row;
			return *this;
		}

		iterator operator++(int)
		{
			iterator previous(*this);
			++(*this);
			return previous;
		}

		const iterator &operator+=(const ssize_t i)
		{
			this->m_position += i;
			this->m_row = this->find_row(this->m_position);
			return *this;
		}

		iterator operator+(const ssize_t i) const { return iterator(this->m_parent, (size_t)((ssize_t)this->m_position + i)); }
		iterator operator-(const ssize_t i) const { return iterator(this->m_parent, (size_t)((ssize_t)this->m_position - i)); }
		ptrdiff_t operator-(const iterator &i) const { return (ptrdiff_t)this->m_position - (ptrdiff_t)i.m_position; }

		T &operator*() { return this->m_values[this->m_position]; }
		const T &operator*() const { return this->m_values[this->m_position]; }

	private:
		// the row of the entry at a position (rows with no entries are skipped)
		size_t find_row(size_t position) const
		{
			if (this->m_rows == 0 || !this->m_row_offsets)
				return 0;
			const SparseIndex *end = this->m_row_offsets + this->m_rows + 1;
			const SparseIndex *above = std::upper_bound(this->m_row_offsets, end, (SparseIndex)position);
			const size_t row = (size_t)(above - this->m_row_offsets) - 1;
			return std::min(row, this->m_rows - 1);
		}

		parent_type *m_parent;
		size_t m_position;
		T *m_values;
		const SparseIndex *m_row_offsets;
		const SparseIndex *m_col_indices;
		size_t m_rows;
		size_t m_row;
	};

	template<typename T>
	typename SparseMatrix<T>::iterator SparseMatrix<T>::begin()
	{
		return iterator(this, 0);
	}


	/*!
	 *  The state of the sparse arguments of a skeleton call: the inputs that have another pattern than the output are aligned to it for
	 *  the call, and are restored when this object goes out of scope.
	 */
	class SparseCallGuard
	{
	public:
		SparseCallGuard() {}
		SparseCallGuard(const SparseCallGuard &) = delete;
		SparseCallGuard(SparseCallGuard &&other): m_restore(std::move(other.m_restore)) { other.m_restore.clear(); }

		~SparseCallGuard()
		{
			for (auto it = this->m_restore.rbegin(); it != this->m_restore.rend(); ++it)
				(*it)();
		}

		std::vector<std::function<void()>> m_restore;
	};

	namespace sparse_detail
	{
		template<size_t OutArity, size_t InArity, typename Arg>
		void collect(size_t k, Arg &&arg, std::vector<SparseMatrixBase *> &inputs, std::vector<SparseMatrixBase *> &outputs)
		{
			if constexpr (is_sparse_matrix<typename std::decay<Arg>::type>::value)
			{
				if (k < OutArity)
					outputs.push_back(&arg);
				else if (k < OutArity + InArity)
					inputs.push_back(&arg);
			}
		}
	}

	/*!
	 *  Prepares the sparse arguments of a call of a skeleton (the first OutArity arguments are the outputs, then InArity element-wise inputs):
	 *  an output that has no pattern gets the pattern of the inputs, which must be the same, or are combined (by the union or the intersection)
	 *  if the policy says so. The returned object restores the arguments after the call.
	 */
	template<size_t OutArity, size_t InArity, typename... CallArgs>
	SparseCallGuard sparse_prepare(SparsePolicy policy, double missing, CallArgs&&... args)
	{
		SparseCallGuard guard;
		if constexpr ((is_sparse_matrix<typename std::decay<CallArgs>::type>::value || ...))
		{
			std::vector<SparseMatrixBase *> inputs, outputs;
			size_t k = 0;
			(sparse_detail::collect<OutArity, InArity>(k++, args, inputs, outputs), ...);
			if (inputs.empty() && outputs.empty())
				return guard; // (the sparse arguments are only read through their proxies)

			for (SparseMatrixBase *in : inputs)
				if (!in->has_pattern())
					SKEPU_ERROR("A sparse input of a skeleton call has no pattern");

			// The pattern of the call
			SparsePatternPtr target;
			std::shared_ptr<const SparseCombination> combination;
			bool differ = false;
			for (SparseMatrixBase *in : inputs)
				if (!in->pattern()->equals(*inputs[0]->pattern()))
					differ = true;

			if (!inputs.empty())
			{
				if (!differ)
				{
					target = inputs[0]->pattern();
				}
				else if (policy == SparsePolicy::SamePattern)
				{
					SKEPU_ERROR("The sparse inputs of the call have different patterns: use setSparsePolicy(SparsePolicy::Union) or Intersection to combine them");
				}
				else
				{
					std::vector<SparsePatternPtr> patterns;
					for (SparseMatrixBase *in : inputs)
						patterns.push_back(in->pattern());
					combination = sparse_combine(policy, patterns);
					target = combination->pattern;
				}
			}
			else if (!outputs.empty() && outputs[0]->has_pattern())
				target = outputs[0]->pattern();

			if (!target)
				SKEPU_ERROR("The sparse output of a skeleton call has no pattern, and there is no sparse input to give it one");

			for (SparseMatrixBase *out : outputs)
			{
				if (!out->has_pattern())
					out->adopt_pattern(target);
				else if (!out->pattern()->equals(*target))
					SKEPU_ERROR("The pattern of a sparse output does not match the pattern of the call");
			}

			if (combination)
			{
				for (size_t j = 0; j < inputs.size(); ++j)
				{
					if (inputs[j]->pattern()->equals(*target))
						continue;
					for (SparseMatrixBase *out : outputs)
						if (out == inputs[j])
							SKEPU_ERROR("A sparse output cannot also be an input of another pattern");
					guard.m_restore.push_back(inputs[j]->align_to(target, combination->positions[j], missing));
				}
			}
		}
		return guard;
	}

} // end namespace skepu

#endif // _SPARSE_MATRIX_H
