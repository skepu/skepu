#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>
#include <skepu-lib/io.hpp>


int uf1(int a, int b)
{
	return a * b;
}

auto pairs = skepu::MapPairs(uf1);

TEST_CASE("MapPairs fundamentals 1")
{
	const size_t Vsize{100};
	const size_t Hsize{100};

	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);

	skepu::external([&]()
	{
		for(int i = 0; i < Vsize; ++i) v1(i) = i+1;
		for(int i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;

		std::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";
	},
	skepu::write(v1, h1));

	skepu::Matrix<int> res(Vsize, Hsize);

	pairs(res, v1, h1);

	skepu::io::cout << "\ntest 1 res: " << res;
}



int uf2(skepu::Index2D i, int ve1, int ve2, int ve3, int he1, int he2, skepu::Vec<int> test, int u1, int u2)
{
//	std:: cout << "(" << i.row << ", " << i.col << ")\n";
	return i.row + i.col + u1; // + ve1 + ve2 + ve3 + he1 + he2 + test.data[0] + u1 + u2;
}

auto pairs2 = skepu::MapPairs<3, 2>(uf2);

TEST_CASE("MapPairs fundamentals 2")
{
	const size_t Vsize{100};
	const size_t Hsize{100};

	skepu::Vector<int> v1(Vsize), v2(Vsize), v3(Vsize);
	skepu::Vector<int> h1(Hsize), h2(Hsize);
	skepu::Vector<int> testarg(10);
	skepu::Matrix<int> res(Vsize, Hsize);

	pairs2(res, v1, v2, v3, h1, h2, testarg, 10, 2);

	skepu::io::cout << "\ntest 2 res: " << res << "\n";
}



int uf3(skepu::Index2D i, int u)
{
	return i.row + i.col + u;
}

auto pairs3 = skepu::MapPairs<0, 0>(uf3);

TEST_CASE("MapPairs fundamentals 3")
{
	const size_t Vsize{100};
	const size_t Hsize{100};

	skepu::Matrix<int> res(Vsize, Hsize);

	pairs3(res, 2);

	skepu::io::cout << "\ntest 3 res: " << res << "\n";
}
