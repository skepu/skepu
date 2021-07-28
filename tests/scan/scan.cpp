#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/io.hpp>

float plus_f(float a, float b)
{
	return a + b;
}

auto prefix_sum = skepu::Scan(plus_f);

TEST_CASE("Scan fundamentals")
{
	const size_t size{100};

	skepu::Vector<float> v(size), r(size);
	v.randomize(0, 10);

	skepu::io::cout << "v: " << v << "\n";


	prefix_sum.setStartValue(10);
	prefix_sum.setScanMode(skepu::ScanMode::Inclusive);

	// With containers
	prefix_sum(r, v);
	skepu::io::cout << "Scan inclusive: r = " << r << "\n";

	// With iterators
	prefix_sum(r.begin(), r.end(), v.begin());
	skepu::io::cout << "Scan inclusive: r = " << r << "\n";

	// With mixed iterators
	prefix_sum(r.begin(), r.end(), v);
	skepu::io::cout << "Scan inclusive: r = " << r << "\n";


	prefix_sum.setScanMode(skepu::ScanMode::Exclusive);

	// With containers
	prefix_sum(r, v);
	skepu::io::cout << "Scan exclusive: r = " << r << "\n";

	// With iterators
	prefix_sum(r.begin(), r.end(), v.begin());
	skepu::io::cout << "Scan exclusive: r = " << r << "\n";

	// With mixed iterators
	prefix_sum(r, v.begin());
	skepu::io::cout << "Scan exclusive: r = " << r << "\n";
}

