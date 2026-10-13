// Merged from: mappairsreduce.cpp, mpr_multi_return.cpp
// (one program instead of one for each: the build of a test program has a cost of its own; each part is in a namespace, and its test cases have its name as a prefix)

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <iostream>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>


// ======== mappairsreduce ========

namespace mappairsreduce_part
{
int uf1(skepu::Index2D testidx, int a, int b)
{
	return a * b;   
}

int sumd(int lhs, int rhs)
{
	return lhs + rhs;
}

int uf2(skepu::Index2D i, skepu::Vec<int> test, int u)
{
	return i.row + i.col + u;
}

auto pairs = skepu::MapPairsReduce(uf1, sumd);
auto pairs4 = skepu::MapPairsReduce<0, 0>(uf2, sumd);

TEST_CASE("mappairsreduce: MapPairsReduce fundamentals")
{
	const size_t Vsize{76};
	const size_t Hsize{65};
	
	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);
	skepu::Vector<int> resV(Vsize), resH(Hsize);

	skepu::external([&]
	{
		for (size_t i = 0; i < Vsize; ++i) v1(i) = i+1;
		for (size_t i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;
	}, skepu::write(v1, h1));

//	skepu::io::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";

	pairs.setReduceMode(skepu::ReduceMode::RowWise);
	pairs(resV, v1, h1);
	skepu::external(skepu::read(resV), [&]
	{
		for (size_t i = 0; i < Vsize; ++i)
		{
			int res = 0;
			for (size_t j = 0; j < Hsize; ++j)
				res += (i+1) * (j+1) * 10;
			CHECK(resV(i) == res);
		}
	});
//	skepu::io::cout << "\nRow-wise resV: " << resV << "\n";

	pairs.setReduceMode(skepu::ReduceMode::ColWise);
	pairs(resH, v1, h1);
	skepu::external(skepu::read(resH), [&]
	{
		for (size_t j = 0; j < Hsize; ++j)
		{
			INFO("pairs, j=" + std::to_string(j));
			int res = 0;
			for (size_t i = 0; i < Vsize; ++i)
				res += (i+1) * (j+1) * 10;
			CHECK(resH(j) == res);
		}
	});
//	skepu::io::cout << "\nCol-wise resH: " << resH << "\n";
	
	// Test implicit dimensions
	{
		pairs4.setDefaultSize(Vsize, Hsize);

		pairs4.setReduceMode(skepu::ReduceMode::RowWise);
		pairs4(resV, v1, 0);
		skepu::external(skepu::read(resV), [&]
		{
			for (size_t i = 0; i < Vsize; ++i)
			{
				int res = 0;
				for (size_t j = 0; j < Hsize; ++j)
					res += i + j;
				CHECK(resV(i) == res);
			}
		});
	//	skepu::io::cout << "\nRow-wise resV: " << resV << "\n";
		
		pairs4.setReduceMode(skepu::ReduceMode::ColWise);
		pairs4(resH, v1, 0);
		skepu::external(skepu::read(resH), [&]
		{
			for (size_t j = 0; j < Hsize; ++j)
			{
				INFO("pairs4, j=" + std::to_string(j));
				int res = 0;
				for (size_t i = 0; i < Vsize; ++i)
					res += i + j;
				CHECK(resH(j) == res);
			}
		});
	//	skepu::io::cout << "\nCol-wise resH: " << resH << "\n";
	}
}
}


// ======== mpr_multi_return ========

namespace mpr_multi_return_part
{
skepu::multiple<int, int> uf_multi(int a, int b)
{
	return skepu::ret(a * b, a + b);
}

skepu::multiple<int, int> uf4_multi(skepu::Index2D i, skepu::Vec<int> test, int u)
{
	return skepu::ret(i.row + i.col + u, test(0));
}

auto pairs = skepu::MapPairsReduce(uf_multi, skepu::util::add<int>);
auto pairs4 = skepu::MapPairsReduce<0, 0>(uf4_multi, skepu::util::add<int>);

TEST_CASE("mpr_multi_return: MapPairsReduce with variadic return")
{
	const size_t Vsize{100};
	const size_t Hsize{100};

	skepu::Vector<int> v1(Vsize, 3), h1(Hsize, 7);
	skepu::Vector<int> resV1(Vsize), resV2(Vsize), resH1(Hsize), resH2(Hsize);

	for (int i = 0; i < Vsize; ++i) v1(i) = i+1;
	for (int i = 0; i < Hsize; ++i) h1(i) = (i+1)*10;

	// Column-wise: one result for each element of h1, from all the elements of v1
	pairs.setReduceMode(skepu::ReduceMode::ColWise);
	pairs(resH1, resH2, v1, h1);

	skepu::external(
		skepu::read(resH1, resH2),
		[&]{
			for (size_t j = 0; j < Hsize; ++j)
			{
				int sumProduct{0}, sumSum{0};
				for (size_t i = 0; i < Vsize; ++i)
				{
					sumProduct += v1(i) * h1(j);
					sumSum += v1(i) + h1(j);
				}
				CHECK(resH1(j) == sumProduct);
				CHECK(resH2(j) == sumSum);
			}
		});

	// Row-wise: one result for each element of v1, from all the elements of h1
	pairs.setReduceMode(skepu::ReduceMode::RowWise);
	pairs(resV1, resV2, v1, h1);

	skepu::external(
		skepu::read(resV1, resV2),
		[&]{
			for (size_t i = 0; i < Vsize; ++i)
			{
				int sumProduct{0}, sumSum{0};
				for (size_t j = 0; j < Hsize; ++j)
				{
					sumProduct += v1(i) * h1(j);
					sumSum += v1(i) + h1(j);
				}
				CHECK(resV1(i) == sumProduct);
				CHECK(resV2(i) == sumSum);
			}
		});

	// Test implicit dimensions: the index of the pair and a random access argument
	{
		pairs4.setDefaultSize(Vsize, Hsize);
		pairs4.setReduceMode(skepu::ReduceMode::ColWise);
		pairs4(resH1, resH2, v1, 0);

		skepu::external(
			skepu::read(resH1, resH2),
			[&]{
				for (size_t j = 0; j < Hsize; ++j)
				{
					int sumIndex{0};
					for (size_t i = 0; i < Vsize; ++i)
						sumIndex += i + j + 0;
					CHECK(resH1(j) == sumIndex);
					CHECK(resH2(j) == (int)Vsize * v1(0));
				}
			});

		pairs4.setReduceMode(skepu::ReduceMode::RowWise);
		pairs4(resV1, resV2, v1, 0);

		skepu::external(
			skepu::read(resV1, resV2),
			[&]{
				for (size_t i = 0; i < Vsize; ++i)
				{
					int sumIndex{0};
					for (size_t j = 0; j < Hsize; ++j)
						sumIndex += i + j + 0;
					CHECK(resV1(i) == sumIndex);
					CHECK(resV2(i) == (int)Hsize * v1(0));
				}
			});
	}
}
}
