#include <cassert>
#include <cmath>
#include <iostream>

#include <skepu>
#include <skepu-lib/io.hpp>
#include <skepu-lib/blas.hpp>


/*
 * Solves A x = b for a symmetric positive definite matrix A with the conjugate gradient
 * method, using skepu::blas. x holds the initial guess and, afterwards, the solution.
 * Returns the number of iterations: in exact arithmetic N are enough.
 */
template<typename T>
size_t conjugate_gradient(skepu::Matrix<T> BLAS_CONST& A, skepu::Vector<T> BLAS_CONST& b, skepu::Vector<T> &x, T tolerance)
{
	size_t N = b.size();
	assert(A.size_i() == N && A.size_j() == N && x.size() == N);

	skepu::Vector<T> p(N, "p"), r(N, "r"), Ap(N, "Ap");

	// Set up initial r and p
	skepu::blas::copy(N, b, 1, r, 1);
	skepu::blas::gemv(skepu::blas::Op::NoTrans, N, N, -1.f, A, N, x, 1, 1.f, r, 1); // r := b - A * x
	skepu::blas::copy(N, r, 1, p, 1); // p := r
	T rTr = skepu::blas::dot(N, r, 1, r, 1); // rTr = r * r

	for (size_t k = 0; k < N; ++k)
	{
		if (std::sqrt(rTr) < tolerance)
			return k;

		SKEPU_TRACE_SCOPE("Iteration");

		// Compute alpha
		skepu::blas::gemv(skepu::blas::Op::NoTrans, N, N, 1.f, A, N, p, 1, 0.f, Ap, 1); // Ap := A * p
		T tmp = skepu::blas::dot(N, p, 1, Ap, 1); // tmp := p * Ap = p * A * p
		T alpha = rTr / tmp;

		// Update x
		skepu::blas::axpy(N,  alpha,  p, 1, x, 1); // x := x + alpha * p

		// Update r
		skepu::blas::axpy(N, -alpha, Ap, 1, r, 1); // r := r - alpha * Ap

		// Compute beta
		T rTr_new = skepu::blas::dot(N, r, 1, r, 1); // rTr_new := r * r
		T beta = rTr_new / rTr;

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

	skepu::Vector<T> b(N, "b");
	skepu::Matrix<T> A(N, N, "A");
	skepu::Vector<T> x(N, "x", 0);

	b.randomize(0, 10); // PRNG
	A.randomizeReal(); // PRNG

	// Make A symmetric and strongly diagonally dominant, which makes it positive definite
	// (conjugate gradient needs that): A := (R + R^T) / 2 + N * I
	skepu::external(skepu::read(A), [&]
	{
		for (size_t i = 0; i < N; ++i)
		{
			for (size_t j = i + 1; j < N; ++j)
				A(i, j) = A(j, i) = (A(i, j) + A(j, i)) / 2;
			A(i, i) += N;
		}
	}, skepu::write(A));

	size_t iterations = conjugate_gradient(A, b, x, 1e-4f);

	// Check the solution: the residual b - A x
	skepu::Vector<T> r(N, "r");
	skepu::blas::copy(N, b, 1, r, 1);
	skepu::blas::gemv(skepu::blas::Op::NoTrans, N, N, -1.f, A, N, x, 1, 1.f, r, 1);
	T residual = skepu::blas::nrm2(N, r, 1);

	skepu::io::cout << "Conjugate gradient, size " << N << "\n";
	skepu::io::cout << "x: " << x << "\n";
	skepu::io::cout << "Converged: " << (residual < 1e-3f * (1 + skepu::blas::nrm2(N, b, 1)) && iterations <= N ? "yes" : "no") << "\n";

	return 0;
}
