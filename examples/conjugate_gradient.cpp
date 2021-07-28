#include <iostream>

#include <skepu>
#include <skepu-lib/blas.hpp>


template<typename T>
void conjugate_gradient(skepu::Matrix<T> BLAS_CONST& A, skepu::Vector<T> BLAS_CONST& b, skepu::Vector<T> &x)
{
	size_t N = b.size();
	assert(A.size_i() == N && A.size_j() == N && x.size() == N);
	skepu::Vector<T> p(N), r(N), Ap(N);
	
	// Set up initial r and p
	skepu::blas::copy(N, b, 1, r, 1);
	skepu::blas::gemv(skepu::blas::Op::NoTrans, N, N, -1.f, A, N, x, 1, 1.f, r, 1);
	skepu::blas::copy(N, r, 1, p, 1); // p := r
	
	float rTr = skepu::blas::dot(N, r, 1, r, 1); // rTr = r * r
	
	for (size_t k = 0; k < N; ++k)
	{
		// Compute alpha
		skepu::blas::gemv(skepu::blas::Op::NoTrans, N, N, 1.f, A, N, p, 1, 0.f, Ap, 1); // Ap := A * p
		float tmp = skepu::blas::dot(N, p, 1, Ap, 1); // tmp := p * Ap = p * A * p
		float alpha = rTr / tmp;
		
		// Update x
		skepu::blas::axpy(N,  alpha,  p, 1, x, 1); // x := x + alpha * p
		
		// Update r
		skepu::blas::axpy(N, -alpha, Ap, 1, r, 1); // r := r - alpha * Ap
		
		// Compute beta
		float rTr_new = skepu::blas::dot(N, r, 1, r, 1); // rTr_new := r * r
		float beta = rTr_new / rTr;
		
		// Early exit condition
		if (sqrt(rTr_new) < 1e-10f)
      return;
		
		// Update p
		skepu::blas::scal(N, beta, p, 1); // p := beta * p
		skepu::blas::axpy(N, 1.f, r, 1, p, 1); // p := r + p
		
		rTr = rTr_new;
		
	//	skepu::cout << "\n--- Iteration " << (k+1) << "---\n";
	//	skepu::cout << "x: " << x << "\nr: " << r << "\np: " << p << "\n";
	}
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
	
	skepu::Vector<T> b(N);
	skepu::Matrix<T> A(N, N);
	skepu::Vector<T> x(N, 0);
//	b = {1, 2};
//	A = {4, 1, 1, 3};
	b.randomize(0, 10);
	A.randomizeReal();
	
//	skepu::cout << "~~~ Conjugate gradient ~~~\n";
//	skepu::cout << "A: " << A << "\n";
//	skepu::cout << "b: " << b << "\n";
	
	conjugate_gradient(A, b, x);
	
//	skepu::cout << "x: " << x << "\n";
	
	return 0;
}
