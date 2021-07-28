#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

auto sum = skepu::Reduce(skepu::util::add<float>);
auto max_sum = skepu::Reduce(skepu::util::add<float>, skepu::util::max<float>);

TEST_CASE("Reduce fundamentals")
{
  const size_t size{100};
	
	skepu::Matrix<float> m(size / 2, size / 2);
	skepu::Vector<float> v(size), rv(size / 2);
	m.randomize(0, 10);
	v.randomize(0, 10);

	// With containers
	float r = sum(v);
	skepu::io::cout << "Sum of v: " << r << "\n";

	r = sum(m);
	skepu::io::cout << "Sum of m: " << r << "\n";

	sum.setReduceMode(skepu::ReduceMode::RowWise);
	sum(rv, m);
	skepu::io::cout << "1D Row-wise reducetion of m: " << rv << "\n";

	sum.setReduceMode(skepu::ReduceMode::ColWise);
	sum(rv,m);
	skepu::io::cout << "1D Column-wise reducetion of m: " << rv << "\n";

	// 2D reduce
	max_sum.setReduceMode(skepu::ReduceMode::RowWise);
	r = max_sum(m);
	skepu::io::cout << "Reduce 2D max row-sum: r = " << r << "\n";

	max_sum.setReduceMode(skepu::ReduceMode::ColWise);
	r = max_sum(m);
	skepu::io::cout << "Reduce 2D max col-sum: r = " << r << "\n";

	r = max_sum(v);
	skepu::io::cout << "Sum of v (Reduce2D): " << r << "\n";
}

