// The SparseMatrix container (compressed sparse rows) and its use in skeletons and in the sparse BLAS routines:
// as a random access argument (a row proxy), as the stored values (Map, MapReduce, Reduce) with the index of each entry,
// as the input of row-wise reductions, with inputs of different patterns (union, intersection), and in csrmv and csrmm.
// (CPU and OpenMP backends)

// (an error is an exception, which the tests of the checks catch)
#define SKEPU_ENABLE_EXCEPTIONS
#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <skepu-lib/blas.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace blas = skepu::blas;
using skepu::SparseMatrix;
using skepu::SparsePolicy;
using skepu::Vector;
using skepu::Matrix;


// ======== The user functions ========

// y(i) = the row i of A times x (a row proxy)
float row_times_x(skepu::Index1D row, skepu::SparseMat<float> A, skepu::Vec<float> x)
{
	float sum = 0;
	for (size_t k = A.row_offsets[row.i]; k < A.row_offsets[row.i + 1]; ++k)
		sum += A.data[k] * x(A.col_indices[k]);
	return sum;
}

float scale_value(float a, float s) { return a * s; }

float index_value(skepu::Index2D index, float a) { return a + 100.f * index.row + 1.f * index.col; }

float add_values(float a, float b) { return a + b; }
float multiply_values(float a, float b) { return a * b; }

// the entries of S scaled by the dot product of the row i of U and the row j of V (sampled dense-dense product)
float sampled_product(skepu::Index2D index, float s, skepu::Mat<float> U, skepu::Mat<float> V)
{
	float dot = 0;
	for (size_t l = 0; l < U.cols; ++l)
		dot += U(index.row, l) * V(index.col, l);
	return s * dot;
}

float value_times_x(skepu::Index2D index, float a, skepu::Vec<float> x) { return a * x(index.col); }

float square_value(float a) { return a * a; }
float minimum(float a, float b) { return a < b ? a : b; }


// ======== The helpers ========

// A deterministic sparse matrix, as a dense matrix (with zeros), some rows empty
static std::vector<double> random_dense(size_t rows, size_t cols, unsigned seed, double density)
{
	std::vector<double> dense(rows * cols, 0.0);
	unsigned state = seed * 2654435761u + 12345u;
	auto next = [&]() { state = state * 1664525u + 1013904223u; return (state >> 8) / double(1 << 24); };
	for (size_t r = 0; r < rows; ++r)
	{
		if (r % 5 == 3)
			continue; // an empty row
		for (size_t c = 0; c < cols; ++c)
			if (next() < density)
				dense[r * cols + c] = 1.0 + std::floor(next() * 8) * 0.25 - (next() < 0.3 ? 3.0 : 0.0);
	}
	return dense;
}

static SparseMatrix<float> sparse_of(const std::vector<double> &dense, size_t rows, size_t cols)
{
	std::vector<std::tuple<size_t, size_t, float>> entries;
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
			if (dense[r * cols + c] != 0.0)
				entries.emplace_back(r, c, (float)dense[r * cols + c]);
	return SparseMatrix<float>::from_coo(rows, cols, entries);
}

static std::vector<double> dense_of(SparseMatrix<float> &A)
{
	std::vector<double> dense(A.rows() * A.cols(), 0.0);
	for (size_t r = 0; r < A.rows(); ++r)
		for (size_t c = 0; c < A.cols(); ++c)
			dense[r * A.cols() + c] = A(r, c);
	return dense;
}

static bool close(double got, double want, double tol = 1e-4)
{
	return std::fabs(got - want) <= tol * std::max(1.0, std::fabs(want));
}

static std::vector<float> host(Vector<float> &v)
{
	v.updateHost();
	std::vector<float> h(v.size());
	for (size_t i = 0; i < h.size(); ++i)
		h[i] = v(i);
	return h;
}

static void fill(Vector<float> &v, unsigned salt)
{
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = 1.0f + ((i * 7 + salt * 13) % 11) * 0.25f - ((i + salt) % 3) * 1.5f;
}


// ======== The container ========

TEST_CASE("sparse: construction, conversion and access")
{
	// [[2 0 1], [0 0 0], [4 3 0]]
	std::vector<std::tuple<size_t, size_t, float>> entries = {{2, 1, 3.f}, {0, 2, 1.f}, {2, 0, 4.f}, {0, 0, 2.f}, {0, 2, 0.5f}};
	SparseMatrix<float> A = SparseMatrix<float>::from_coo(3, 3, entries);
	REQUIRE(A.rows() == 3);
	REQUIRE(A.cols() == 3);
	REQUIRE(A.nnz() == 4); // (the two values of the place (0, 2) are added)
	REQUIRE(A(0, 0) == 2.f);
	REQUIRE(A(0, 1) == 0.f);
	REQUIRE(A(0, 2) == 1.5f);
	REQUIRE(A(1, 1) == 0.f);
	REQUIRE(A(2, 0) == 4.f);
	REQUIRE(A(2, 1) == 3.f);

	Matrix<float> dense = A.to_dense();
	REQUIRE(dense(0, 2) == 1.5f);
	REQUIRE(dense(1, 0) == 0.f);
	SparseMatrix<float> B = SparseMatrix<float>::from_dense(dense);
	REQUIRE(B.nnz() == 4);
	REQUIRE(B.pattern()->equals(*A.pattern()));
	REQUIRE(B.pattern() != A.pattern()); // (the same pattern, but not the same object)

	// the pattern is shared, the values are copied
	SparseMatrix<float> C(A.pattern());
	REQUIRE(C.pattern() == A.pattern());
	REQUIRE(C.values().size() == 4);
	SparseMatrix<float> D(A);
	REQUIRE(D.pattern() == A.pattern());
	D.values()[0] = 99.f;
	REQUIRE(A(0, 0) == 2.f);
	REQUIRE(D(0, 0) == 99.f);

	// arrays
	const skepu::SparseIndex offsets[] = {0, 2, 2, 4};
	const skepu::SparseIndex columns[] = {0, 2, 0, 1};
	const float values[] = {2.f, 1.5f, 4.f, 3.f};
	SparseMatrix<float> E(3, 3, 4, offsets, columns, values);
	REQUIRE(E.pattern()->equals(*A.pattern()));
	REQUIRE(E(2, 1) == 3.f);

	// the transpose
	SparseMatrix<float> T = A.transposed();
	REQUIRE(T.rows() == 3);
	for (size_t r = 0; r < 3; ++r)
		for (size_t c = 0; c < 3; ++c)
			REQUIRE(T(r, c) == A(c, r));
	SparseMatrix<float> T2 = A.transposed();
	REQUIRE(T2.pattern() == T.pattern()); // (the pattern of the transpose is made once)

	// a matrix with no pattern
	SparseMatrix<float> empty;
	REQUIRE(!empty.has_pattern());
	REQUIRE(empty.nnz() == 0);
}

TEST_CASE("sparse: the patterns are checked")
{
	// the columns of a row are sorted and unique, and in range; the row offsets are consistent
	REQUIRE_THROWS(SparseMatrix<float>(2, 3, {0, 2, 3}, {1, 0, 2}, {1.f, 2.f, 3.f}));   // not sorted
	REQUIRE_THROWS(SparseMatrix<float>(2, 3, {0, 2, 3}, {1, 1, 2}, {1.f, 2.f, 3.f}));   // repeated
	REQUIRE_THROWS(SparseMatrix<float>(2, 3, {0, 2, 3}, {0, 3, 2}, {1.f, 2.f, 3.f}));   // out of range
	REQUIRE_THROWS(SparseMatrix<float>(2, 3, {0, 3, 2}, {0, 1, 2}, {1.f, 2.f, 3.f}));   // the offsets decrease
	REQUIRE_THROWS(SparseMatrix<float>(2, 3, {0, 1, 2}, {0, 1, 2}, {1.f, 2.f, 3.f}));   // the offsets end at 2
	REQUIRE_NOTHROW(SparseMatrix<float>(3, 3, {0, 0, 0, 0}, {}, {}));                    // no entries
}

TEST_CASE("sparse: a MatrixMarket file")
{
	const char *name = "sparse_test_matrix.mtx";
	{
		std::ofstream file(name);
		file << "%%MatrixMarket matrix coordinate real symmetric\n% a comment\n3 3 4\n1 1 2.0\n2 1 -1.0\n3 2 4.0\n3 3 5.0\n";
	}
	SparseMatrix<float> A = SparseMatrix<float>::read_matrix_market(name);
	std::remove(name);
	REQUIRE(A.rows() == 3);
	REQUIRE(A.nnz() == 6); // (the entries off the diagonal are mirrored)
	REQUIRE(A(0, 0) == 2.f);
	REQUIRE(A(1, 0) == -1.f);
	REQUIRE(A(0, 1) == -1.f);
	REQUIRE(A(1, 2) == 4.f);
	REQUIRE(A(2, 1) == 4.f);
	REQUIRE(A(2, 2) == 5.f);
}


// ======== A random access argument ========

TEST_CASE("sparse: a row proxy in a Map (matrix-vector product)")
{
	for (auto size : std::vector<std::pair<size_t, size_t>>{{1, 1}, {8, 5}, {40, 60}, {100, 100}})
	{
		const size_t rows = size.first, cols = size.second;
		const auto dense = random_dense(rows, cols, (unsigned)(rows + cols), 0.15);
		SparseMatrix<float> A = sparse_of(dense, rows, cols);
		Vector<float> x(cols), y(rows);
		fill(x, 1);
		const auto hx = host(x);

		auto spmv = skepu::Map<0>(row_times_x);
		spmv(y, A, x);
		const auto hy = host(y);
		for (size_t r = 0; r < rows; ++r)
		{
			double want = 0;
			for (size_t c = 0; c < cols; ++c)
				want += dense[r * cols + c] * hx[c];
			INFO("rows " << rows << " cols " << cols << " row " << r);
			REQUIRE(close(hy[r], want));
		}
	}
}


// ======== The stored values, with the index of each ========

TEST_CASE("sparse: Map over the stored values")
{
	const size_t rows = 30, cols = 40;
	const auto dense = random_dense(rows, cols, 3, 0.2);
	SparseMatrix<float> A = sparse_of(dense, rows, cols);

	// an output with no pattern gets the pattern of the input (shared)
	SparseMatrix<float> B;
	skepu::Map<1>(scale_value)(B, A, 2.f);
	REQUIRE(B.pattern() == A.pattern());
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
			REQUIRE(B(r, c) == (float)(2 * dense[r * cols + c]));

	// in place
	skepu::Map<1>(scale_value)(B, B, 0.5f);
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
			REQUIRE(B(r, c) == (float)dense[r * cols + c]);

	// the index of an entry is its row and column
	SparseMatrix<float> I;
	skepu::Map<1>(index_value)(I, A);
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
			if (dense[r * cols + c] != 0.0)
				REQUIRE(I(r, c) == (float)(dense[r * cols + c] + 100 * r + c));
			else
				REQUIRE(I(r, c) == 0.f);

	// two inputs of the same pattern
	SparseMatrix<float> S;
	skepu::Map<2>(add_values)(S, A, B);
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
			REQUIRE(S(r, c) == (float)(2 * dense[r * cols + c]));

	// the values can be used as a Vector, by the skeletons and the BLAS routines of vectors
	blas::scal(S.nnz(), 0.5f, S.values(), 1);
	REQUIRE(S(2, 2) == A(2, 2));
	REQUIRE(close(blas::nrm2(A.nnz(), A.values(), 1) * blas::nrm2(A.nnz(), A.values(), 1), [&]() { double s = 0; for (double v : dense) s += v * v; return s; }(), 1e-3));
}

TEST_CASE("sparse: Map with a sparse output of a given pattern (sampled dense-dense product)")
{
	const size_t rows = 25, cols = 30, k = 7;
	const auto dense = random_dense(rows, cols, 9, 0.1);
	SparseMatrix<float> S = sparse_of(dense, rows, cols);
	Matrix<float> U(rows, k), V(cols, k);
	for (size_t i = 0; i < U.size(); ++i) U[i] = 0.5f + (i % 5) * 0.25f;
	for (size_t i = 0; i < V.size(); ++i) V[i] = 1.0f - (i % 7) * 0.125f;
	U.updateHost(); V.updateHost();

	SparseMatrix<float> R(S.pattern());
	skepu::Map<1>(sampled_product)(R, S, U, V);
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
		{
			double dot = 0;
			for (size_t l = 0; l < k; ++l)
				dot += U(r, l) * V(c, l);
			REQUIRE(close(R(r, c), dense[r * cols + c] * dot));
		}
}

TEST_CASE("sparse: the patterns of the inputs and the output must match")
{
	const auto d1 = random_dense(10, 10, 1, 0.3), d2 = random_dense(10, 10, 2, 0.3);
	SparseMatrix<float> A = sparse_of(d1, 10, 10), B = sparse_of(d2, 10, 10), C;
	REQUIRE(!A.pattern()->equals(*B.pattern()));
	auto add = skepu::Map<2>(add_values);
	auto scale = skepu::Map<1>(scale_value);
	REQUIRE_THROWS(add(C, A, B));      // different patterns, and the policy is SamePattern
	SparseMatrix<float> D(B.pattern());
	REQUIRE_THROWS(scale(D, A, 1.f));  // the output has another pattern than the input
	SparseMatrix<float> E;
	SparseMatrix<float> F;
	REQUIRE_THROWS(scale(E, F, 1.f));  // the input has no pattern
}


// ======== Reductions ========

TEST_CASE("sparse: Reduce and MapReduce of the stored values")
{
	const size_t rows = 50, cols = 20;
	const auto dense = random_dense(rows, cols, 5, 0.25);
	SparseMatrix<float> A = sparse_of(dense, rows, cols);

	double sum = 0, sum_squares = 0, smallest = 1e30;
	for (double v : dense)
		if (v != 0.0)
		{
			sum += v;
			sum_squares += v * v;
			smallest = std::min(smallest, v);
		}

	auto add = skepu::Reduce(add_values);
	REQUIRE(close(add(A), sum, 1e-3));

	auto min = skepu::Reduce(minimum);
	min.setStartValue(1e30f);
	REQUIRE(close(min(A), smallest));

	auto sum_of_squares = skepu::MapReduce<1>(square_value, add_values);
	REQUIRE(close(sum_of_squares(A), sum_squares, 1e-3));
}

TEST_CASE("sparse: Reduce by rows")
{
	const size_t rows = 50, cols = 20;
	const auto dense = random_dense(rows, cols, 6, 0.25);
	SparseMatrix<float> A = sparse_of(dense, rows, cols);
	Vector<float> sums(rows), mins(rows), wrong(rows + 1);

	auto add = skepu::Reduce(add_values);
	add.setReduceMode(skepu::ReduceMode::RowWise);
	add(sums, A);
	auto min = skepu::Reduce(minimum);
	min.setStartValue(1e30f);
	min.setReduceMode(skepu::ReduceMode::RowWise);
	min(mins, A);

	const auto hs = host(sums), hm = host(mins);
	for (size_t r = 0; r < rows; ++r)
	{
		double sum = 0, smallest = 1e30;
		for (size_t c = 0; c < cols; ++c)
			if (dense[r * cols + c] != 0.0)
			{
				sum += dense[r * cols + c];
				smallest = std::min(smallest, dense[r * cols + c]);
			}
		REQUIRE(close(hs[r], sum));
		REQUIRE(close(hm[r], smallest)); // (an empty row gives the start value)
	}
	REQUIRE_THROWS(add(wrong, A));
}

TEST_CASE("sparse: MapReduce by rows (matrix-vector product)")
{
	for (auto size : std::vector<std::pair<size_t, size_t>>{{1, 1}, {8, 5}, {40, 60}, {200, 150}})
	{
		const size_t rows = size.first, cols = size.second;
		const auto dense = random_dense(rows, cols, (unsigned)(rows * 3 + cols), 0.1);
		SparseMatrix<float> A = sparse_of(dense, rows, cols);
		Vector<float> x(cols), y(rows);
		fill(x, 2);
		const auto hx = host(x);

		auto spmv = skepu::MapReduce<1>(value_times_x, add_values);
		spmv.setReduceMode(skepu::ReduceMode::RowWise);
		spmv(y, A, x);
		const auto hy = host(y);
		for (size_t r = 0; r < rows; ++r)
		{
			double want = 0;
			for (size_t c = 0; c < cols; ++c)
				want += dense[r * cols + c] * hx[c];
			INFO("rows " << rows << " cols " << cols << " row " << r);
			REQUIRE(close(hy[r], want));
		}
	}
}


// ======== Inputs of different patterns ========

TEST_CASE("sparse: Map of the union and the intersection of the patterns")
{
	const size_t rows = 40, cols = 35;
	const auto d1 = random_dense(rows, cols, 11, 0.2), d2 = random_dense(rows, cols, 12, 0.2);
	SparseMatrix<float> A = sparse_of(d1, rows, cols), B = sparse_of(d2, rows, cols);
	REQUIRE(!A.pattern()->equals(*B.pattern()));
	const auto pattern_a = A.pattern(), pattern_b = B.pattern();
	const auto values_a = host(A.values()), values_b = host(B.values());

	// the union: sparse addition
	auto sum = skepu::Map<2>(add_values);
	sum.setSparsePolicy(SparsePolicy::Union);
	SparseMatrix<float> S;
	sum(S, A, B);
	size_t union_count = 0, intersection_count = 0;
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
		{
			const bool a = d1[r * cols + c] != 0.0, b = d2[r * cols + c] != 0.0;
			union_count += (a || b);
			intersection_count += (a && b);
			REQUIRE(close(S(r, c), d1[r * cols + c] + d2[r * cols + c]));
		}
	REQUIRE(S.nnz() == union_count);

	// the inputs are as they were
	REQUIRE(A.pattern() == pattern_a);
	REQUIRE(B.pattern() == pattern_b);
	REQUIRE(host(A.values()) == values_a);
	REQUIRE(host(B.values()) == values_b);

	// the same patterns again: the result has the pattern that was made before
	SparseMatrix<float> S2;
	sum(S2, A, B);
	REQUIRE(S2.pattern() == S.pattern());

	// the intersection: the product of the elements
	auto product = skepu::Map<2>(multiply_values);
	product.setSparsePolicy(SparsePolicy::Intersection);
	SparseMatrix<float> P;
	product(P, A, B);
	REQUIRE(P.nnz() == intersection_count);
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
			REQUIRE(close(P(r, c), d1[r * cols + c] * d2[r * cols + c]));

	// the value of a missing entry
	auto times = skepu::Map<2>(multiply_values);
	times.setSparsePolicy(SparsePolicy::Union).setImplicitZero(2.0);
	SparseMatrix<float> Q;
	times(Q, A, B);
	for (size_t r = 0; r < rows; ++r)
		for (size_t c = 0; c < cols; ++c)
		{
			const bool a = d1[r * cols + c] != 0.0, b = d2[r * cols + c] != 0.0;
			const double want = (a && b) ? d1[r * cols + c] * d2[r * cols + c] : (a ? d1[r * cols + c] * 2.0 : (b ? 2.0 * d2[r * cols + c] : 0.0));
			REQUIRE(close(Q(r, c), want));
		}

	// an output that is an input of another pattern would be overwritten
	REQUIRE_THROWS(sum(A, A, B));

	// different sizes
	SparseMatrix<float> small = sparse_of(random_dense(rows - 1, cols, 3, 0.2), rows - 1, cols), R;
	REQUIRE_THROWS(sum(R, A, small));
}


// ======== The sparse BLAS routines ========

TEST_CASE("sparse: csrmv")
{
	for (auto size : std::vector<std::pair<size_t, size_t>>{{1, 1}, {9, 5}, {40, 60}, {120, 100}})
		for (blas::Op op : {blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans})
			for (auto ab : std::vector<std::pair<float, float>>{{1.f, 0.f}, {-1.5f, 0.5f}, {0.f, 2.f}})
				for (int incx : {1, 2})
				{
					const size_t rows = size.first, cols = size.second;
					const bool t = op != blas::Op::NoTrans;
					const size_t lenx = t ? rows : cols, leny = t ? cols : rows;
					INFO("rows " << rows << " cols " << cols << " op " << (int)op << " alpha " << ab.first << " beta " << ab.second << " incx " << incx);
					const auto dense = random_dense(rows, cols, (unsigned)(rows + 7 * cols), 0.12);
					SparseMatrix<float> A = sparse_of(dense, rows, cols);
					Vector<float> x(1 + (lenx - 1) * incx), y(leny);
					fill(x, 3); fill(y, 4);
					const auto hx = host(x), hy = host(y);
					if (ab.second == 0.f)
						for (size_t i = 0; i < leny; ++i)
							y[i] = NAN; // (not read)

					blas::sparse::csrmv(op, ab.first, A, x, incx, ab.second, y, 1);
					const auto r = host(y);
					for (size_t i = 0; i < leny; ++i)
					{
						double sum = 0;
						for (size_t j = 0; j < lenx; ++j)
							sum += (t ? dense[j * cols + i] : dense[i * cols + j]) * hx[j * incx];
						REQUIRE(close(r[i], ab.first * sum + ab.second * hy[i]));
					}
				}
}

TEST_CASE("sparse: csrmm")
{
	const size_t rows = 30, cols = 20, n = 6;
	for (blas::Op op : {blas::Op::NoTrans, blas::Op::Trans})
		for (auto ab : std::vector<std::pair<float, float>>{{1.f, 0.f}, {2.f, -0.5f}})
		{
			const bool t = op != blas::Op::NoTrans;
			const size_t m = t ? cols : rows, k = t ? rows : cols;
			INFO("op " << (int)op << " alpha " << ab.first << " beta " << ab.second);
			const auto dense = random_dense(rows, cols, 17, 0.15);
			SparseMatrix<float> A = sparse_of(dense, rows, cols);
			Matrix<float> B(k, n), C(m, n);
			for (size_t i = 0; i < B.size(); ++i) B[i] = 1.0f + (i % 6) * 0.25f;
			for (size_t i = 0; i < C.size(); ++i) C[i] = (i % 4) * 0.5f;
			B.updateHost(); C.updateHost();
			std::vector<double> hc(C.size());
			for (size_t i = 0; i < C.size(); ++i) hc[i] = C[i];

			blas::sparse::csrmm(op, ab.first, A, B, ab.second, C);
			C.updateHost();
			for (size_t i = 0; i < m; ++i)
				for (size_t j = 0; j < n; ++j)
				{
					double sum = 0;
					for (size_t l = 0; l < k; ++l)
						sum += (t ? dense[l * cols + i] : dense[i * cols + l]) * B(l, j);
					REQUIRE(close(C(i, j), ab.first * sum + ab.second * hc[i * n + j]));
				}
		}

	Matrix<float> wrong_b(5, n), c(rows, n);
	SparseMatrix<float> A = sparse_of(random_dense(rows, cols, 1, 0.1), rows, cols);
	REQUIRE_THROWS(blas::sparse::csrmm(blas::Op::NoTrans, 1.f, A, wrong_b, 0.f, c));
}

TEST_CASE("sparse: a conjugate gradient solver with a sparse matrix")
{
	// the matrix of the 2D Poisson problem on a g x g grid (symmetric positive definite)
	const size_t g = 12, n = g * g;
	std::vector<std::tuple<size_t, size_t, float>> entries;
	for (size_t i = 0; i < g; ++i)
		for (size_t j = 0; j < g; ++j)
		{
			const size_t row = i * g + j;
			entries.emplace_back(row, row, 4.f);
			if (i > 0) entries.emplace_back(row, row - g, -1.f);
			if (i + 1 < g) entries.emplace_back(row, row + g, -1.f);
			if (j > 0) entries.emplace_back(row, row - 1, -1.f);
			if (j + 1 < g) entries.emplace_back(row, row + 1, -1.f);
		}
	SparseMatrix<float> A = SparseMatrix<float>::from_coo(n, n, entries);
	REQUIRE(A.nnz() == 5 * n - 4 * g);

	Vector<float> b(n), x(n, 0.f), r(n), p(n), Ap(n);
	for (size_t i = 0; i < n; ++i) b[i] = 1.f;
	blas::copy(n, b, 1, r, 1);
	blas::copy(n, r, 1, p, 1);
	float rr = blas::dot(n, r, 1, r, 1);
	size_t iterations = 0;
	for (; iterations < n && std::sqrt(rr) > 1e-4f; ++iterations)
	{
		blas::sparse::csrmv(blas::Op::NoTrans, 1.f, A, p, 1, 0.f, Ap, 1);
		const float alpha = rr / blas::dot(n, p, 1, Ap, 1);
		blas::axpy(n, alpha, p, 1, x, 1);
		blas::axpy(n, -alpha, Ap, 1, r, 1);
		const float rr_new = blas::dot(n, r, 1, r, 1);
		blas::scal(n, rr_new / rr, p, 1);
		blas::axpy(n, 1.f, r, 1, p, 1);
		rr = rr_new;
	}
	REQUIRE(iterations < n);

	// A x = b
	Vector<float> check(n);
	blas::sparse::csrmv(blas::Op::NoTrans, 1.f, A, x, 1, 0.f, check, 1);
	for (size_t i = 0; i < n; ++i)
		REQUIRE(std::fabs(check(i) - 1.f) < 1e-3f);
}
