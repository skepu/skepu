#include <skepu>
#include <skepu-lib/io.hpp>

#define ENABLE_1D_EXAMPLE 1
#define ENABLE_2D_EXAMPLE 1
#define ENABLE_DEBUG 1

char automaton1D(skepu::Region1D<char> r, skepu::Mat<char> updateRules)
{
	float newval = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		if (i != 0)
			newval += r(i) ? 1 : 0;
	return updateRules(r(0), newval) ? 1 : 0;
}

char automaton2D(skepu::Region2D<char> r, skepu::Mat<char> updateRules)
{
	float newval = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			if (!(i == 0 && j == 0))
				newval += r(i, j) ? 1 : 0;
	return updateRules(r(0, 0), newval) ? 1 : 0;
}


int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		skepu::io::cout << "Usage: " << argv[0] << " dim size iterations backend\n";
		exit(1);
	}
	
	const float dim = atof(argv[1]);
	const float size = atof(argv[2]);
	const float iters = atof(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	

#if ENABLE_1D_EXAMPLE
	if (dim == 1)
	{
		auto update = skepu::MapOverlap(automaton1D);
		update.setOverlap(1);
		update.setEdgeMode(skepu::Edge::Pad);
		update.setPad(0);
		skepu::Vector<char> domainA(size, 0), domainB(size);
		skepu::Matrix<char> updateRules(2, 2);
		updateRules(0, 1) = true;
		
		domainA.randomize(0, 2);
		
		for (size_t i = 0; i < iters; i += 2)
		{
			update(domainB, domainA, updateRules);
			update(domainA, domainB, updateRules);
			
#if ENABLE_DEBUG
			skepu::io::cout << domainB << "\n";
			skepu::io::cout << domainA << "\n";
#endif
		}
		
		skepu::io::cout << domainA << "\n";
		exit(0);
	}
#endif
	
#if ENABLE_2D_EXAMPLE
	if (dim == 2)
	{
		auto update = skepu::MapOverlap(automaton2D);
		update.setOverlap(1, 1);
		update.setEdgeMode(skepu::Edge::Pad);
		update.setPad(0);
		skepu::Matrix<char> domainA(size, size, 0), domainB(size, size);
		skepu::Matrix<char> updateRules(2, 8);
		
		// Each empty cell with three neighbors becomes populated.
		updateRules(0, 3) = true;
		
		// Each populated cell with two or three neighbors survives.
		updateRules(1, 2) = true;
		updateRules(1, 3) = true;
		
		domainA.randomize(0, 2);
		
		for (size_t i = 0; i < iters; ++i)
		{
			update(domainB, domainA, updateRules);
			update(domainA, domainB, updateRules);
			
#if ENABLE_DEBUG
			skepu::io::cout << domainB << "\n";
			skepu::io::cout << domainA << "\n";
#endif
		}
		
		skepu::io::cout << domainA << "\n";
		exit(0);
	}
#endif
	
	return 0;
}

