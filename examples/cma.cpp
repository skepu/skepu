#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

template<typename T, typename U>
U avg(skepu::Index1D index, T sum)
{
	return (U)sum / (index.i + 1);
}


auto prefix_sum = skepu::Scan(skepu::util::add<int>);
auto average = skepu::Map(avg<int, float>);

void cma(skepu::Vector<int> &in, skepu::Vector<float> &out)
{
	prefix_sum(in, in);
	average(out, in);
}

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
	
	skepu::Vector<int> in(size);
	skepu::Vector<float> out(size);
	in.randomize(0, 10);
	
	if (size <= 50)
		skepu::io::cout << "Elements: " << in << "\n";
	
	cma(in, out);
	
	if (size <= 50)
		skepu::io::cout << "Cumulative moving average: " << out << "\n";
	else
		skepu::io::cout << "Average: " << out[size-1] << "\n";
	
	return 0;
}
