// A tour of SkePU-BLAS: a call of each routine of skepu::blas, and the important variations of it (transposes, strides, a part of a
// vector or a matrix, the triangle of a matrix, real and complex elements, sparse matrices). The data is synthetic and the program has
// no purpose other than to show the calls. Each use is in a labelled trace scope (SKEPU_TRACE_SCOPE), so that a trace of the program,
// as SkeVU shows it, groups what a use does: the setup of its data, and the skeletons of a routine that has several stages (the
// substitutions of trsv and trsm are a skeleton call for each step, a sparse matrix-vector product is a reduction by rows).
//
// Usage: blas_tour size backend
// (Not for the OpenCL, CUDA and MPI backends: the sparse matrices are not supported by them yet.)
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <tuple>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/randomize.hpp>
#include <skepu-lib/blas.hpp>

namespace blas = skepu::blas;
using blas::Op;
using blas::Uplo;
using blas::Diag;
using blas::Side;

using T = float;
using Complex = skepu::complex::complex<float>;


// ---------------------------------------------------------------- user functions for the setup of the data

// A := (R + R^T) / 2 + n * I: symmetric and strongly diagonally dominant (so positive definite)
T symmetrize(skepu::Index2D index, const skepu::Mat<T> R, T n)
{
	return (R(index.row, index.col) + R(index.col, index.row)) / 2 + (index.row == index.col ? n : 0);
}

// A matrix with a large diagonal and small other elements: a triangle of it is a well conditioned triangular matrix
T well_conditioned(skepu::Index2D index, T a, T n)
{
	return (index.row == index.col) ? n : 0.1f * a;
}

// Complex numbers from the index
Complex complex_element(skepu::Index2D index)
{
	return Complex{1.f + 0.25f * (float)((index.row * 3 + index.col * 5) % 7), 0.5f - 0.125f * (float)((index.row + 2 * index.col) % 5)};
}

Complex complex_vector_element(skepu::Index1D index)
{
	return Complex{1.f + 0.25f * (float)((index.i * 3) % 7), 0.5f - 0.125f * (float)(index.i % 5)};
}

// R + R^T over the union of the patterns, made symmetric positive definite
T sparse_symmetrize(skepu::Index2D index, T r, T r_transposed, T n)
{
	return (r + r_transposed) / 2 + (index.row == index.col ? n : 0);
}


// The entries of a random sparse matrix: the diagonal and some other places in each row
static skepu::SparseMatrix<T> random_sparse(size_t n, size_t per_row)
{
	std::vector<std::tuple<size_t, size_t, T>> places;
	std::mt19937 generator(7);
	for (size_t row = 0; row < n; ++row)
	{
		std::set<size_t> columns{row};
		for (size_t e = 0; e < per_row && columns.size() < n; ++e)
			columns.insert(generator() % n);
		for (size_t column : columns)
			places.emplace_back(row, column, T(0));
	}
	return skepu::SparseMatrix<T>::from_coo(n, n, places);
}


int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::external([&]{ std::cout << "Usage: " << argv[0] << " size backend\n"; });
		exit(1);
	}

	const size_t n = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	const size_t half = n / 2;

	// ------------------------------------------------------------ the data
	skepu::Vector<T> x(n, "x"), y(n, "y"), z(n, "z");
	skepu::Matrix<T> A(n, n, "A"), B(n, n, "B"), C(n, n, "C"), R(n, n, "R"), S(n, n, "S"), L(n, n, "L");
	{
		SKEPU_TRACE_SCOPE("Setup: dense vectors and matrices");
		skepu::randomize::uniform(x, -1.f, 1.f);
		skepu::randomize::uniform(y, -1.f, 1.f);
		skepu::randomize::uniform(z, -1.f, 1.f);
		skepu::randomize::uniform(A, -1.f, 1.f);
		skepu::randomize::uniform(B, -1.f, 1.f);
		skepu::randomize::uniform(C, -1.f, 1.f);
		skepu::randomize::uniform(R);

		// S: a symmetric matrix, L: a triangular matrix that can be solved with
		skepu::Map<0>(symmetrize)(S, R, static_cast<T>(n));
		skepu::Map<1>(well_conditioned)(L, A, static_cast<T>(n));
	}

	// ------------------------------------------------------------ level 1
	{
		SKEPU_TRACE_SCOPE("Level 1: copy, swap, scal, axpy, axpby");
		skepu::Vector<T> w(n, "w");
		blas::copy(n, x, 1, w, 1);          // w := x
		blas::swap(n, w, 1, z, 1);          // w and z change places
		blas::scal(n, 2.f, w, 1);           // w := 2 w
		blas::axpy(n, 0.5f, x, 1, w, 1);    // w := 0.5 x + w
		blas::axpby(n, 1.f, y, 1, 0.25f, w, 1); // w := y + 0.25 w
	}

	{
		SKEPU_TRACE_SCOPE("Level 1: strides, a part of a vector, and a negative increment");
		skepu::Vector<T> w(n, "w", 0.f);
		blas::copy(half, x, 2, w, 1);       // the first half of w := every second element of x
		blas::axpy(half, 1.f, y, -2, w, 1); // w += every second element of y, backwards from the last one
		blas::copy(n, x, -1, z, 1);         // z := x reversed
		blas::scal(half, 3.f, w, 1);        // the first half of w only
	}

	T dot_xy = 0, norm_x = 0;
	{
		SKEPU_TRACE_SCOPE("Level 1: dot, dotu, nrm2, asum (reductions to a scalar)");
		skepu::Scalar<T> d = blas::dot(n, x, 1, y, 1);   // (conjugated: the same as dotu for real vectors)
		skepu::Scalar<T> du = blas::dotu(n, x, 1, y, 1);
		skepu::Scalar<T> nx = blas::nrm2(n, x, 1);
		skepu::Scalar<T> a = blas::asum(n, y, 1);
		dot_xy = T(d) + T(du) + T(a);
		norm_x = T(nx);
	}

	size_t largest = 0, smallest = 0;
	{
		SKEPU_TRACE_SCOPE("Level 1: iamax, iamin");
		largest = blas::iamax(n, x, 1);
		smallest = blas::iamin(n, x, 1);
	}

	double extended = 0;
	{
		SKEPU_TRACE_SCOPE("Level 1: dsdot, sdsdot (the sum in double precision)");
		extended = double(blas::dsdot(n, x, 1, y, 1));
		extended += double(blas::sdsdot(n, 1.f, x, 1, y, 1));
	}

	{
		SKEPU_TRACE_SCOPE("Level 1: Givens rotations: rotg, rot, rotmg, rotm");
		T a = 3, b = 4, c = 0, s = 0;
		blas::rotg(&a, &b, &c, &s);                   // (on the host)
		skepu::Vector<T> u(n, "u"), v(n, "v");
		blas::copy(n, x, 1, u, 1);
		blas::copy(n, y, 1, v, 1);
		blas::rot(n, u, 1, v, 1, c, s);               // (u, v) := rotation of (u, v)

		T d1 = 1.5f, d2 = 2.f, x1 = 1.f, y1 = 0.5f;
		T param[5] = {0, 0, 0, 0, 0};
		blas::rotmg(&d1, &d2, &x1, y1, param);        // (on the host)
		blas::rotm(n, u, 1, v, 1, param);             // the modified rotation
	}

	{
		SKEPU_TRACE_SCOPE("Level 1: complex vectors (dot is conjugated, nrm2 and asum are real)");
		skepu::Vector<Complex> cx(n, "cx"), cy(n, "cy");
		skepu::Map<0>(complex_vector_element)(cx);
		blas::copy(n, cx, 1, cy, 1);
		blas::scal(n, Complex{0.f, 1.f}, cy, 1);       // cy := i cx
		skepu::Scalar<Complex> conj_dot = blas::dot(n, cx, 1, cy, 1);
		skepu::Scalar<Complex> plain_dot = blas::dotu(n, cx, 1, cy, 1);
		skepu::Scalar<T> cn = blas::nrm2(n, cx, 1);
		skepu::Scalar<T> ca = blas::asum(n, cy, 1);
		size_t ci = blas::iamax(n, cx, 1);
		(void)conj_dot; (void)plain_dot; (void)cn; (void)ca; (void)ci;
	}

	// ------------------------------------------------------------ level 2
	{
		SKEPU_TRACE_SCOPE("Level 2: gemv (no transpose, transpose, and a part of the matrix)");
		skepu::Vector<T> w(n, "w", 0.f);
		blas::gemv(Op::NoTrans, n, n, 1.f, A, n, x, 1, 0.f, w, 1);        // w := A x
		blas::gemv(Op::Trans, n, n, 1.f, A, n, y, 1, 1.f, w, 1);          // w := A^T y + w
		blas::gemv(Op::NoTrans, half, half, 2.f, A, n, x, 1, 0.5f, w, 1); // the first half of w := 2 (the upper left quarter of A) x + 0.5 w
	}

	{
		SKEPU_TRACE_SCOPE("Level 2: ger, geru (rank-1 updates)");
		blas::ger(n, n, 0.1f, x, 1, y, 1, C, n);    // C := C + 0.1 x y^T
		blas::geru(half, n, 0.1f, x, 1, y, 1, C, n); // the first half of the rows only
	}

	{
		SKEPU_TRACE_SCOPE("Level 2: symmetric and triangular matrices: symv, hemv, trmv");
		skepu::Vector<T> w(n, "w");
		blas::symv(Uplo::Upper, n, 1.f, S, n, x, 1, 0.f, w, 1);          // w := S x (only the upper triangle of S is read)
		blas::hemv(Uplo::Lower, n, 1.f, S, n, y, 1, 1.f, w, 1);          // (the same, for a Hermitian matrix)
		blas::trmv(Uplo::Lower, Op::NoTrans, Diag::NonUnit, n, L, n, w, 1); // w := (the lower triangle of L) w
	}

	{
		SKEPU_TRACE_SCOPE("Level 2: trsv (a triangular solve: a skeleton call for each element of the solution)");
		skepu::Vector<T> w(n, "w");
		blas::copy(n, x, 1, w, 1);
		blas::trsv(Uplo::Lower, Op::NoTrans, Diag::NonUnit, n, L, n, w, 1); // solves (the lower triangle of L) w = x
	}

	{
		SKEPU_TRACE_SCOPE("Level 2: symmetric rank-1 and rank-2 updates: syr, her, syr2, her2");
		skepu::Matrix<T> W1(n, n, "W1", 0.f), W2(n, n, "W2", 0.f);
		blas::syr(Uplo::Upper, n, 0.5f, x, 1, W1, n);        // the upper triangle of W1 += 0.5 x x^T
		blas::her(Uplo::Lower, n, 0.5f, y, 1, W1, n);        // the lower triangle of W1 += 0.5 y y^H
		blas::syr2(Uplo::Upper, n, 0.5f, x, 1, y, 1, W2, n); // the upper triangle of W2 += 0.5 (x y^T + y x^T)
		blas::her2(Uplo::Lower, n, 0.5f, x, 1, y, 1, W2, n);
	}

	// ------------------------------------------------------------ level 3
	{
		SKEPU_TRACE_SCOPE("Level 3: gemm (transposes, and a part of the matrices)");
		skepu::Matrix<T> W(n, n, "W", 0.f);
		blas::gemm(Op::NoTrans, Op::NoTrans, n, n, n, 1.f, A, n, B, n, 0.f, W, n);   // W := A B
		blas::gemm(Op::Trans, Op::NoTrans, n, n, n, 1.f, A, n, B, n, 1.f, W, n);     // W := A^T B + W
		blas::gemm(Op::NoTrans, Op::Trans, half, half, n, 0.5f, A, n, B, n, 0.f, C, n); // the upper left quarter of C := 0.5 A B^T
	}

	{
		SKEPU_TRACE_SCOPE("Level 3: symm, hemm (a symmetric matrix on the left or on the right)");
		skepu::Matrix<T> W(n, n, "W", 0.f);
		blas::symm(Side::Left, Uplo::Upper, n, n, 1.f, S, n, B, n, 0.f, W, n);   // W := S B
		blas::hemm(Side::Right, Uplo::Lower, n, n, 1.f, S, n, A, n, 1.f, W, n);  // W := A S + W
	}

	{
		SKEPU_TRACE_SCOPE("Level 3: symmetric rank-k and rank-2k updates: syrk, herk, syr2k, her2k");
		skepu::Matrix<T> W(n, n, "W", 0.f);
		blas::syrk(Uplo::Upper, Op::NoTrans, n, n, 1.f, A, n, 0.f, W, n);          // the upper triangle of W := A A^T
		blas::herk(Uplo::Lower, Op::ConjTrans, n, n, 1.f, B, n, 1.f, W, n);       // the lower triangle of W += B^H B
		blas::syr2k(Uplo::Upper, Op::NoTrans, n, n, 0.5f, A, n, B, n, 1.f, W, n);  // the upper triangle of W += 0.5 (A B^T + B A^T)
		blas::her2k(Uplo::Lower, Op::NoTrans, n, n, 0.5f, A, n, B, n, 1.f, W, n);
	}

	{
		SKEPU_TRACE_SCOPE("Level 3: trmm");
		skepu::Matrix<T> W(B); // (a copy of B)
		W.setLabel("W");
		blas::trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, n, n, 1.f, L, n, W, n);  // W := (the lower triangle of L) W
		blas::trmm(Side::Right, Uplo::Upper, Op::Trans, Diag::Unit, n, half, 0.5f, L, n, W, n);   // the first half of the columns of W := 0.5 W (the upper triangle of L)^T
	}

	{
		SKEPU_TRACE_SCOPE("Level 3: trsm (a skeleton call for each row of the solution)");
		skepu::Matrix<T> W(B);
		W.setLabel("W");
		blas::trsm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, n, n, 1.f, L, n, W, n); // solves (the lower triangle of L) X = B
	}

	// ------------------------------------------------------------ complex matrices
	{
		SKEPU_TRACE_SCOPE("Complex matrices: gemm (conjugate transpose), hemv, her, her2, hemm, herk, her2k");
		skepu::Matrix<Complex> CA(n, n, "CA"), CB(n, n, "CB"), CW(n, n, "CW", Complex{0.f, 0.f});
		skepu::Vector<Complex> cx(n, "cx"), cy(n, "cy", Complex{0.f, 0.f});
		skepu::Map<0>(complex_element)(CA);
		skepu::Map<0>(complex_element)(CB);
		skepu::Map<0>(complex_vector_element)(cx);

		blas::gemm(Op::ConjTrans, Op::NoTrans, n, n, n, Complex{1.f, 0.f}, CA, n, CB, n, Complex{0.f, 0.f}, CW, n); // CW := CA^H CB
		blas::hemv(Uplo::Upper, n, Complex{1.f, 0.f}, CA, n, cx, 1, Complex{0.f, 0.f}, cy, 1);   // cy := (the Hermitian matrix of the upper triangle of CA) cx
		blas::her(Uplo::Lower, n, 0.5f, cx, 1, CW, n);
		blas::her2(Uplo::Upper, n, Complex{0.5f, 0.25f}, cx, 1, cy, 1, CW, n);
		blas::hemm(Side::Left, Uplo::Upper, n, n, Complex{1.f, 0.f}, CA, n, CB, n, Complex{1.f, 0.f}, CW, n);
		blas::herk(Uplo::Lower, Op::NoTrans, n, n, 1.f, CA, n, 1.f, CW, n);
		blas::her2k(Uplo::Upper, Op::NoTrans, n, n, Complex{0.5f, 0.25f}, CA, n, CB, n, 1.f, CW, n);
	}

	// ------------------------------------------------------------ sparse matrices
	skepu::SparseMatrix<T> P = random_sparse(n, 4);
	{
		SKEPU_TRACE_SCOPE("Setup: a sparse matrix (compressed sparse rows)");
		P.setLabel("P");
		skepu::randomize::uniform(P.values(), -1.f, 1.f); // the values of the entries are a Vector
	}

	skepu::SparseMatrix<T> Q;
	{
		SKEPU_TRACE_SCOPE("Setup: a symmetric positive definite sparse matrix (a Map over the union of the patterns of P and its transpose)");
		Q.setLabel("Q");
		skepu::SparseMatrix<T> Pt = P.transposed();
		auto make_spd = skepu::Map<2>(sparse_symmetrize);
		make_spd.setSparsePolicy(skepu::SparsePolicy::Union);
		make_spd(Q, P, Pt, static_cast<T>(n));
	}

	{
		SKEPU_TRACE_SCOPE("Sparse: csrmv (no transpose and transpose)");
		skepu::Vector<T> w(n, "w", 0.f);
		blas::sparse::csrmv(Op::NoTrans, 1.f, P, x, 1, 0.f, w, 1);   // w := P x
		blas::sparse::csrmv(Op::Trans, 1.f, P, y, 1, 1.f, w, 1);     // w := P^T y + w
	}

	{
		SKEPU_TRACE_SCOPE("Sparse: csrmm (a sparse matrix times a dense matrix)");
		skepu::Matrix<T> W(n, n, "W", 0.f);
		blas::sparse::csrmm(Op::NoTrans, 1.f, P, B, 0.f, W);         // W := P B
	}

	// ------------------------------------------------------------ several routines together
	{
		SKEPU_TRACE_SCOPE("Conjugate gradient with a sparse matrix (3 iterations)");
		skepu::Vector<T> sx(n, "sx", 0.f), r(n, "r"), p(n, "p"), Ap(n, "Ap");
		blas::copy(n, y, 1, r, 1);
		blas::sparse::csrmv(Op::NoTrans, -1.f, Q, sx, 1, 1.f, r, 1);  // r := y - Q sx
		blas::copy(n, r, 1, p, 1);
		skepu::Scalar<T> rTr = blas::dot(n, r, 1, r, 1);
		for (int k = 0; k < 3; ++k)
		{
			SKEPU_TRACE_SCOPE("Iteration");
			blas::sparse::csrmv(Op::NoTrans, 1.f, Q, p, 1, 0.f, Ap, 1);
			skepu::Scalar<T> alpha = rTr / blas::dot(n, p, 1, Ap, 1);
			blas::axpy(n, alpha, p, 1, sx, 1);
			blas::axpy(n, -alpha, Ap, 1, r, 1);
			skepu::Scalar<T> rTr_new = blas::dot(n, r, 1, r, 1);
			skepu::Scalar<T> beta = rTr_new / rTr;
			blas::scal(n, beta, p, 1);
			blas::axpy(n, 1.f, r, 1, p, 1);
			rTr = rTr_new;
		}
	}

	skepu::io::cout << "BLAS tour, size " << n << "\n";
	skepu::io::cout << "dot + dotu + asum: " << dot_xy << ", ||x||: " << norm_x << "\n";
	skepu::io::cout << "iamax: " << largest << ", iamin: " << smallest << ", dsdot + sdsdot: " << extended << "\n";
	return 0;
}
