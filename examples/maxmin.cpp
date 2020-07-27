#include <iostream>
#include <skepu>

struct MaxMin
{
	float max;
	float min;
};

MaxMin preprocess(float val)
{
	MaxMin res;
	res.max = val;
	res.min = val;
	return res;
}

MaxMin max_min_f(MaxMin a, MaxMin b)
{
	MaxMin res;
	res.max = (a.max > b.max) ? a.max : b.max;
	res.min = (a.min < b.min) ? a.min : b.min;
	return res; 
}

void find_max_min(skepu::Vector<float> floats)
{
	auto maxmin = skepu::MapReduce(preprocess, max_min_f);
	maxmin.setStartValue({-INFINITY, INFINITY});
	MaxMin result = maxmin(floats);
	
	std::cout << "Max: " << result.max << "\n";
	std::cout << "Min: " << result.min << "\n";
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		if(!skepu::cluster::mpi_rank())
			std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{skepu::Backend::typeFromString(argv[2])};
	skepu::setGlobalBackendSpec(spec);
	
	
	skepu::Vector<float> floats(size);
	floats.randomize(0, 10000);
	std::cout << "Input: " << floats << "\n";
	
	find_max_min(floats);
	
	return 0;
}

