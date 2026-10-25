#include <skepu>
#include <skepu-lib/randomize.hpp>
#include <skepu-lib/io.hpp>

template<typename T>
T square_f(T a)
{
	return a * a;
}


int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		skepu::io::cout << "Usage: " << argv[0] << " size\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	
	auto square = skepu::Map(square_f<float>);
	
	skepu::Vector<float> in(size), out(size);
	skepu::randomize::uniform(in, 0, 9);
	
	square(out, in);
	
	skepu::io::cout << in << "\n";
	skepu::io::cout << out << "\n";
}
