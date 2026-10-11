#include "../../external/catch2/catch.hpp"

#include <iostream>
#include <skepu>
#include <skepu-lib/util.hpp>
#include <test_helper_automatic_setup.hpp>

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

TEST_CASE("MapPairsReduce with variadic return")
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
