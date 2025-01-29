#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>


skepu::multiple<int, float> uf_mr_multi(int a, float b)
{
	return skepu::ret(a + b, a * b);
}

skepu::multiple<int, int> uf_mr_multi_zero(skepu::Index1D index)
{
	return skepu::ret(index.i, 2*index.i);
}

int sum(int a, int b)
{
	return a + b;
}

auto skel = skepu::MapReduce(uf_mr_multi, sum);
auto skel_zero = skepu::MapReduce(uf_mr_multi_zero, sum);

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::Vector<int> v1(size);
	skepu::Vector<float> v2(size);

	skepu::external(
	[&]{
		for (size_t i = 0; i < size; ++i)
		{
			v1(i) = i;
			v2(i) = 5;
		}},
	skepu::write(v1,v2));

	int sum1;
	float sum2;
	std::tie(sum1, sum2) = skel(v1, v2);
	//	auto [sum1, sum2] = skel(v1, v2);
	skepu::external(
	[&]{
		std::cout << "sum1: " << sum1 << ", sum2: " << sum2 << std::endl;
	});
	
	skel_zero.setDefaultSize(size);
	std::tie(sum1, sum2) = skel_zero();
	//	auto [sum1, sum2] = skel(v1, v2);
	skepu::external(
	[&]{
		std::cout << "sum1: " << sum1 << ", sum2: " << sum2 << std::endl;
	});
	
	return 0;
}
