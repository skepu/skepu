#include <skepu>
#include <skepu-lib/io.hpp>

#define ENABLE_1D_EXAMPLE 1
#define ENABLE_2D_EXAMPLE 1
//#define ENABLE_DEBUG 1

int init_random_f(skepu::Random<1> &random, float aliveRatio)
{
    float r = random.getNormalized();
    return r < aliveRatio ? 1 : 0;
}

int automaton1D(skepu::Region1D<int> r, skepu::Mat<int> updateRules)
{
	float newval = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		if (i != 0)
			newval += r(i) ? 1 : 0;
	return updateRules(r(0), newval) ? 1 : 0;
}

int automaton2D(skepu::Region2D<int> r, skepu::Mat<int> updateRules)
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
		skepu::io::cout << "Usage: " << argv[0] << " dim size iterations init-alive-ratio backend\n";
		exit(1);
	}
	
	const float dim = atof(argv[1]);
	const float size = atof(argv[2]);
	const float iters = atof(argv[3]);
	const float aliveRatio = atof(argv[4]);
	auto spec = skepu::BackendSpec{argv[5]};
	skepu::setGlobalBackendSpec(spec);

	auto init_random = skepu::Map<0>(init_random_f);

#if ENABLE_1D_EXAMPLE
	if (dim == 1)
	{
		auto update = skepu::MapOverlap(automaton1D);
		update.setOverlap(1);
		update.setEdgeMode(skepu::Edge::Pad);
		update.setPad(0);
		skepu::Vector<int> domainA(size, "A", 0), domainB(size, "B");
		skepu::Matrix<int> updateRules(2, 2, "Rules");
		updateRules(0, 1) = true;
		SKEPU_TRACE_VIRTUAL_DATASET("Domain", domainA, domainB);
		
		init_random(domainA, aliveRatio);
		SKEPU_TRACE_SNAPSHOT(domainA);
		
		for (size_t i = 0; i < iters; i += 2)
		{
		    SKEPU_TRACE_SCOPE("Double-iteration");
			update(domainB, domainA, updateRules);
			update(domainA, domainB, updateRules);
			SKEPU_TRACE_SNAPSHOT(domainB);
			SKEPU_TRACE_SNAPSHOT(domainA);
		}
		
		skepu::io::cout << domainA << "\n";
		return 0;
	}
#endif
	
#if ENABLE_2D_EXAMPLE
	if (dim == 2)
	{
		auto update = skepu::MapOverlap(automaton2D);
		update.setOverlap(1, 1);
		update.setEdgeMode(skepu::Edge::Pad);
		update.setPad(0);
		skepu::Matrix<int> domainA(size, size, "A", 0), domainB(size, size, "B");
		skepu::Matrix<int> updateRules(2, 8, "Rules");
		SKEPU_TRACE_VIRTUAL_DATASET("Domain", domainA, domainB);
		
		// Each empty cell with three neighbors becomes populated.
		updateRules(0, 3) = true;
		
		// Each populated cell with two or three neighbors survives.
		updateRules(1, 2) = true;
		updateRules(1, 3) = true;
		
		init_random(domainA, aliveRatio);
		SKEPU_TRACE_SNAPSHOT(domainA);
		
		for (size_t i = 0; i < iters; ++i)
		{
			SKEPU_TRACE_SCOPE("Double-iteration");
			update(domainB, domainA, updateRules);
			update(domainA, domainB, updateRules);
			SKEPU_TRACE_SNAPSHOT(domainB);
			SKEPU_TRACE_SNAPSHOT(domainA);
		}
		
		skepu::io::cout << domainA << "\n";
		return 0;
	}
#endif
	
}
