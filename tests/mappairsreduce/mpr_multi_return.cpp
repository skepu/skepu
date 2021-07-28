#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>
#include <skepu-lib/util.hpp>

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

	skepu::external(
		skepu::read(v1, h1),
		[&]{
			std::cout << "\nv1: " << v1 << "\nh1: " << h1 << "\n\n";
		});

	pairs.setReduceMode(skepu::ReduceMode::ColWise);
	pairs(resH1, resH2, v1, h1);

	skepu::external(
		skepu::read(resH1, resH2),
		[&]{
			std::cout << "\nresH1: " << resH1 << "\n";
			std::cout << "\nresH2: " << resH2 << "\n";
		});

	pairs.setReduceMode(skepu::ReduceMode::RowWise);
	pairs(resV1, resV2, v1, h1);

	skepu::external(
		skepu::read(resV1, resV2),
		[&]{
			std::cout << "\nresV1: " << resV1 << "\n";
			std::cout << "\nresV2: " << resV2 << "\n";
		});

	// Test implicit dimensions
	{

		pairs4.setDefaultSize(Vsize, Hsize);
		pairs4.setReduceMode(skepu::ReduceMode::ColWise);
		pairs4(resH1, resH2, v1, 0);

		skepu::external(
			skepu::read(resH1, resH2),
			[&]{
				std::cout << "\nresH1: " << resH1 << "\n";
				std::cout << "\nresH2: " << resH2 << "\n";
			});

		pairs4.setReduceMode(skepu::ReduceMode::RowWise);
		pairs4(resV1, resV2, v1, 0);
		skepu::external(
			skepu::read(resV1, resV2),
			[&]{
				std::cout << "\nresV1: " << resV1 << "\n";
				std::cout << "\nresV2: " << resV2 << "\n";
			});
	}
}