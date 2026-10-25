// The conjugate gradient solver of conjugate_gradient.cpp for a sparse matrix: the same method, with the matrix in compressed sparse row format
// (skepu::SparseMatrix) and the matrix-vector product of skepu::blas::sparse. (Not for the OpenCL, CUDA and MPI backends.)
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <tuple>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/randomize.hpp>
#include <skepu-lib/blas.hpp>


// A := (R + R^T) / 2 + n * I: symmetric and strongly diagonally dominant, which makes it
// positive definite (conjugate gradient needs that). R^T has other entries than R: the sum is over the union of the two.
template<typename T>
T symmetrize(skepu::Index2D index, T r, T r_transposed, T n)
{
	return (r + r_transposed) / 2 + (index.row == index.col ? n : 0);
}

/*
 * Solves A x = b for a symmetric positive definite matrix A with the conjugate gradient
 * method, using skepu::blas. x holds the initial guess and, afterwards, the solution.
 * Returns the number of iterations: in exact arithmetic N are enough.
 */
template<typename T>
size_t conjugate_gradient(skepu::SparseMatrix<T> &A, skepu::Vector<T> BLAS_CONST& b, skepu::Vector<T> &x, T tolerance)
{
	size_t N = b.size();
	assert(A.rows() == N && A.cols() == N && x.size() == N);

	skepu::Vector<T> p(N, "p"), r(N, "r"), Ap(N, "Ap");

	// Set up initial r and p
	skepu::blas::copy(N, b, 1, r, 1);
	skepu::blas::sparse::csrmv(skepu::blas::Op::NoTrans, -1.f, A, x, 1, 1.f, r, 1); // r := b - A * x
	skepu::blas::copy(N, r, 1, p, 1); // p := r
	// (the dot products are skepu::Scalar: the trace follows them through the arithmetic to the calls that use them)
	skepu::Scalar<T> rTr = skepu::blas::dot(N, r, 1, r, 1); // rTr = r * r

	for (size_t k = 0; k < N; ++k)
	{
		if (std::sqrt(T(rTr)) < tolerance)
			return k;

		SKEPU_TRACE_SCOPE("Iteration");

		// Compute alpha
		skepu::blas::sparse::csrmv(skepu::blas::Op::NoTrans, 1.f, A, p, 1, 0.f, Ap, 1); // Ap := A * p
		skepu::Scalar<T> tmp = skepu::blas::dot(N, p, 1, Ap, 1); // tmp := p * Ap = p * A * p
		skepu::Scalar<T> alpha = rTr / tmp;

		// Update x
		skepu::blas::axpy(N,  alpha,  p, 1, x, 1); // x := x + alpha * p

		// Update r
		skepu::blas::axpy(N, -alpha, Ap, 1, r, 1); // r := r - alpha * Ap

		// Compute beta
		skepu::Scalar<T> rTr_new = skepu::blas::dot(N, r, 1, r, 1); // rTr_new := r * r
		skepu::Scalar<T> beta = rTr_new / rTr;

		// Update p
		skepu::blas::scal(N, beta, p, 1); // p := beta * p
		skepu::blas::axpy(N, 1.f, r, 1, p, 1); // p := r + p

		rTr = rTr_new;
	}
	return N;
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

	using T = float;
	const size_t N = n;
	const size_t per_row = 8; // the entries of a row of R, besides the diagonal

	// The pattern of R: the diagonal and some random places in each row; the values of R are random
	std::vector<std::tuple<size_t, size_t, T>> places;
	std::mt19937 generator(1);
	for (size_t row = 0; row < N; ++row)
	{
		std::set<size_t> columns{row};
		for (size_t e = 0; e < per_row && columns.size() < N; ++e)
			columns.insert(generator() % N);
		for (size_t column : columns)
			places.emplace_back(row, column, T(0));
	}

	skepu::Vector<T> b(N, "b");
	skepu::Vector<T> x(N, "x", 0);
	auto R = skepu::SparseMatrix<T>::from_coo(N, N, places);
	R.setLabel("R");

	skepu::randomize::uniform(b, 0, 10); // PRNG
	skepu::randomize::uniform(R.values()); // PRNG (the values of the entries are a Vector)

	skepu::SparseMatrix<T> A;
	A.setLabel("A");
	skepu::SparseMatrix<T> Rt = R.transposed();
	auto make_spd = skepu::Map<2>(symmetrize<T>);
	make_spd.setSparsePolicy(skepu::SparsePolicy::Union); // R and R^T have different patterns
	make_spd(A, R, Rt, static_cast<T>(N));

	size_t iterations = conjugate_gradient(A, b, x, 1e-4f);

	// Check the solution: the residual b - A x
	skepu::Vector<T> r(N, "r");
	skepu::blas::copy(N, b, 1, r, 1);
	skepu::blas::sparse::csrmv(skepu::blas::Op::NoTrans, -1.f, A, x, 1, 1.f, r, 1);
	T residual = skepu::blas::nrm2(N, r, 1);

	skepu::io::cout << "Conjugate gradient, size " << N << ", " << A.nnz() << " entries\n";
	skepu::io::cout << "x: " << x << "\n";
	skepu::io::cout << "Converged: " << (residual < 1e-3f * (1 + skepu::blas::nrm2(N, b, 1)) && iterations <= N ? "yes" : "no") << "\n";

	return 0;
}
