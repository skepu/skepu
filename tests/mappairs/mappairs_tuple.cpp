#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>

skepu::multiple<int, float> uf(int a, int b)
{
	return skepu::ret(a * b, (float)a / b);
}

auto pairs = skepu::MapPairs(uf);

TEST_CASE("Test 1")
{
	const size_t Vsize{100};
	const size_t Hsize{100};

	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);

	for (int i = 0; i < Vsize; ++i) v1(i) = i+1;
	for (int i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;

	skepu::external(
		skepu::read(v1, h1),
		[&]{
			std::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";
		});

	skepu::Matrix<int> resA(Vsize, Hsize);
	skepu::Matrix<float> resB(Vsize, Hsize);

	pairs(resA, resB, v1, h1);

	skepu::external(
		skepu::read(resA, resB),
		[&]{
			std::cout << "\ntest 1 resA: " << resA;
			std::cout << "\ntest 1 resB: " << resB;
		});
}



skepu::multiple<int, int, float>
uf2(skepu::Index2D i, int ve1, int ve2, int ve3, int he1, int he2, skepu::Vec<int> test, int u1, int u2)
{
//	std:: cout << "(" << i.row << ", " << i.col << ")\n";
	return skepu::ret(i.row, i.col, ve1 + ve2 + ve3 + he1 + he2 + test.data[0] + u1 + u2);
}

auto pairs2 = skepu::MapPairs<3, 2>(uf2);

TEST_CASE("Test 2")
{
	const size_t Vsize{100};
	const size_t Hsize{100};

	skepu::Vector<int> v1(Vsize), v2(Vsize), v3(Vsize);
	skepu::Vector<int> h1(Hsize), h2(Hsize);
	skepu::Vector<int> testarg(10);
	skepu::Matrix<int> resA(Vsize, Hsize);
	skepu::Matrix<int> resB(Vsize, Hsize);
	skepu::Matrix<float> resC(Vsize, Hsize);

	pairs2(resA, resB, resC, v1, v2, v3, h1, h2, testarg, 10, 2);

	skepu::external(
		skepu::read(resA, resB, resC),
		[&]{
			std::cout << "\ntest 2 resA: " << resA << "\n";
			std::cout << "\ntest 3 resB: " << resB << "\n";
			std::cout << "\ntest 3 resC: " << resC << "\n";
		});
}



skepu::multiple<int> uf3(skepu::Index2D i, int u)
{
	return skepu::ret(i.row + i.col + u);
}

auto pairs3 = skepu::MapPairs<0, 0>(uf3);

TEST_CASE("Test 3")
{
	const size_t Vsize{100};
	const size_t Hsize{100};
	
	skepu::Matrix<int> res(Vsize, Hsize);

	pairs3(res, 2);

	std::cout << "\ntest 3 res: " << res << "\n";
}


