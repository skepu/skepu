#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>


int uf5(skepu::Index2D testidx, int a, int b)
{
	return a * b;
}

int sumd(int lhs, int rhs)
{
	return lhs + rhs;
}

int uf4(skepu::Index2D i, skepu::Vec<int> test, int u)
{
	return i.row + i.col + u;
}

auto pairs = skepu::MapPairsReduce(uf5, sumd);
auto pairs4 = skepu::MapPairsReduce<0, 0>(uf4, sumd);

TEST_CASE("MapPairsReduce fundamentals")
{
	const size_t Vsize{100};
	const size_t Hsize{100};
	
	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);
	skepu::Vector<int> resV(Vsize), resH(Hsize);

	for (int i = 0; i < Vsize; ++i) v1(i) = i+1;
	for (int i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;

	skepu::io::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";

	pairs.setReduceMode(skepu::ReduceMode::RowWise);
	pairs(resV, v1, h1);
	skepu::io::cout << "\nRow-wise resV: " << resV << "\n";

	pairs.setReduceMode(skepu::ReduceMode::ColWise);
	pairs(resH, v1, h1);
	skepu::io::cout << "\nCo-wise resH: " << resH << "\n";
	
	// Test implicit dimensions
	{
		pairs4.setDefaultSize(Vsize, Hsize);
		pairs4.setReduceMode(skepu::ReduceMode::ColWise);
		pairs4(resH, v1, 0);
		skepu::io::cout << "\nresH: " << resH << "\n";

		pairs4.setReduceMode(skepu::ReduceMode::RowWise);
		pairs4(resV, v1, 0);
		skepu::io::cout << "\nresV: " << resV << "\n";
	}
}
