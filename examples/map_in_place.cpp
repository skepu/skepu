#include <iostream>
#include <skepu>


int update_map(int value, int increment)
{
	return value + increment;
}

float update_mapoverlap(skepu::Region1D<float> r)
{
	return (r(-1) + r(1)) / 2;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		if(!skepu::cluster::mpi_rank())
			std::cout << "Usage: " << argv[0] << " start end samples backend\n";
		exit(1);
	}
	
	const size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	// Map in place
	// Valid as long as the result container is not also passed as a random-access input
	{
		skepu::Vector<int> values(size);
		values.randomize(0, 9);
		std::cout << values << "\n";
		
		auto instance = skepu::Map<1>(update_map);
		
		instance(values, values, 1);
		std::cout << values << "\n";
	}
	
	
	// MapOverlap in place
	// Valid only for Red-Black update pattern with edge mode other than Cyclic
	// and as long as the result container is not also passed as a random-access input
	{
		skepu::Vector<float> domain(size, 0);
		domain.randomize(0, 2);
		std::cout << domain << "\n";
		
		auto update = skepu::MapOverlap(update_mapoverlap);
		update.setOverlap(1);
		update.setEdgeMode(skepu::Edge::Duplicate);
		update.setUpdateMode(skepu::UpdateMode::RedBlack);
		
		update(domain, domain);
		std::cout << domain << "\n";
	}
	
	return 0;
}

