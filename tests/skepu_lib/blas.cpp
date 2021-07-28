#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>

#include <skepu-lib/blas.hpp>
#include <skepu-lib/io.hpp>


TEST_CASE("BLAS level 1")
{
	const size_t n{1000000};
	
	skepu::Vector<skepu::complex::complex<float>> v1(n), v2(n), r(n);
	
	v1.randomize(-10, 10);
	
	
	// LEVEL 1
	
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::blas::scal(n, 5.f, v1, 1);
	skepu::io::cout << "scal(n, 5.f, v1, 1);" << "\n";
	skepu::io::cout << "v1: " << v1 << "\n\n";
	
	
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::blas::copy(n, v1, 1, r, 1);
	skepu::io::cout << "copy(n, v1, 1, r, 1)" << "\n";
	skepu::io::cout << "r: " << v1 << "\n\n";
	
	
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::io::cout << "v2: " << v2 << "\n";
	skepu::blas::axpy(n, 5.f, v1, 1, v2, 1);
	skepu::io::cout << "axpy(n, 5.f, v1, 1, v2, 1);" << "\n";
	skepu::io::cout << "v2: " << v2 << "\n\n";
	
	
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::io::cout << "v2: " << v2 << "\n";
	auto res = skepu::blas::dot(n, v1, 1, v2, 1);
	skepu::io::cout << "dot(n, v1, 1, v2, 1);" << "\n";
	skepu::io::cout << "res: " << res << "\n\n";
	
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::io::cout << "v2: " << v2 << "\n";
	res = skepu::blas::dotu(n, v1, 1, v2, 1);
	skepu::io::cout << "dotu(n, v1, 1, v2, 1);" << "\n";
	skepu::io::cout << "res: " << res << "\n\n";
	
	
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::io::cout << "v2: " << v2 << "\n";
	res = skepu::blas::nrm2(n, v1, 1);
	skepu::io::cout << "nrm2(n, v1, 1);" << "\n";
	skepu::io::cout << "res: " << res << "\n\n";
	
	
	skepu::io::cout << "v1: " << v1 << "\n";
	res = skepu::blas::asum(n, v1, 1);
	skepu::io::cout << "asum(n, v1, 1);" << "\n";
	skepu::io::cout << "res: " << res << "\n\n";
	
	
	skepu::io::cout << "v1: " << v1 << "\n";
	res = skepu::blas::iamax(n, v1, 1);
	skepu::io::cout << "iamax(n, v1, 1);" << "\n";
	skepu::io::cout << "res: " << res << "\n\n";
	
	
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::io::cout << "v2: " << v2 << "\n";
	skepu::blas::swap(n, v1, 1, v2, 1);
	skepu::io::cout << "iamax(n, v1, 1, v2, 1);" << "\n";
	skepu::io::cout << "v1: " << v1 << "\n";
	skepu::io::cout << "v2: " << v2 << "\n\n";
	
	
	{
		using TestType = skepu::complex::complex<float>;
		double a = .34;
		double b = .23;
		double c = .9;
		double s = .5;
		
		skepu::blas::rotg(&a, &b, &c, &s);
		skepu::io::cout << "a: " << a << "\n"; // now r
		skepu::io::cout << "b: " << b << "\n"; // now z
		skepu::io::cout << "c: " << c << "\n";
		skepu::io::cout << "s: " << s << "\n";
		
		skepu::Vector<TestType> x(n, .34), y(n, .23);
		
		skepu::io::cout << "x: " << x << "\n";
		skepu::io::cout << "y: " << y << "\n\n";
		skepu::blas::rot(n, x, 1, y, 1, c, s);
		skepu::io::cout << "x: " << x << "\n";
		skepu::io::cout << "y: " << y << "\n\n";
		
	}
	
}

TEST_CASE("BLAS level 2")
{
	const size_t n{1000};
	
	{
		using TestType = skepu::complex::complex<float>;
		size_t m = n + 2;
		skepu::Matrix<TestType> A(m, n);
		skepu::Vector<TestType> x(n), y(m);
		
		A.randomize(0, 10);
		x.randomize(0, 10);
		y.randomize(0, 10);
		
		skepu::io::cout << "y: " << y << "\n";
		skepu::io::cout << "A: " << A << "\n";
		skepu::io::cout << "x: " << x << "\n";
		
		float alpha = 0.5;
		float beta = 0.5;
		skepu::blas::gemv(skepu::blas::Op::NoTrans,
			m, n, alpha, A, m, x, 1, beta, y, 1
		);
		skepu::io::cout << "y: " << y << "\n\n";
	}
	
	
	{
		using TestType = skepu::complex::complex<float>;
		size_t my_m = 3;
		size_t my_n = 3;
		skepu::Matrix<TestType> A(my_m, my_n);
		skepu::Vector<TestType> x(my_n), y(my_m);
		
		A = {
			1,	3,	4,
			4,	5,	6,
			6,	7,	8
		};
		y = {1,1,3};
		x = {9,6,7};
		
		skepu::io::cout << "y: " << y << "\n";
		skepu::io::cout << "A: " << A << "\n";
		skepu::io::cout << "x: " << x << "\n";
		
		float alpha = .1;
		float beta = 0.5;
		skepu::blas::gemv(skepu::blas::Op::NoTrans,
			my_m, my_n, alpha, A, my_m, x, 1, beta, y, 1
		);
		skepu::io::cout << "y: " << y << "\n\n";
	}
	
	{
		skepu::io::cout << "###########\n### GER\n###########\n";
		
		using TestType = skepu::complex::complex<float>;
		size_t my_m = 3;
		size_t my_n = 3;
		skepu::Matrix<TestType> A(my_m, my_n);
		skepu::Vector<TestType> x(my_n), y(my_m);
		
		A = {
			1,	3,	4,
			4,	5,	6,
			6,	7,	8
		};
		y = {1,1,3};
		x = {9,6,7};
		
		skepu::io::cout << "y: " << y << "\n";
		skepu::io::cout << "A: " << A << "\n";
		skepu::io::cout << "x: " << x << "\n";
		
		float alpha = 1;
		skepu::blas::ger(
			my_m, my_n, alpha, x, 1, y, 1, A, my_m
		);
		skepu::io::cout << "A: " << A << "\n\n";
		skepu::blas::geru(
			my_m, my_n, alpha, x, 1, y, 1, A, my_m
		);
		skepu::io::cout << "###########\n### ---\n###########\n\n";
	}
	
}

TEST_CASE("BLAS level 3")
{
	const size_t n{100};

	using TestType = skepu::complex::complex<float>;
	size_t m = n + 2;
	size_t k = n + 1;
	skepu::Matrix<TestType> C(m, n), A(m, k), B(k, n);
	
	A.randomize(0, 10);
	B.randomize(0, 10);
	C.randomize(0, 10);
	
	skepu::io::cout << "C: " << C << "\n";
	skepu::io::cout << "A: " << A << "\n";
	skepu::io::cout << "B: " << B << "\n";
	
	float alpha = 0.5;
	float beta = 0.5;
	skepu::blas::gemm(skepu::blas::Op::NoTrans, skepu::blas::Op::NoTrans,
		m, n, k, alpha, A, 1, B, 1, beta, C, 1
	);
	skepu::io::cout << "C: " << C << "\n\n";
	
	
}

