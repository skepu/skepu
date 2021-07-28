#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

int monte_carlo_sample(skepu::Random<2> &random)
{
  float x = random.getNormalized();
  float y = random.getNormalized();
  
  // check if (x,y) is inside region
  return ((x*x + y*y) < 1) ? 1 : 0;
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		skepu::io::cout << "Usage: " << argv[0] << " start end samples backend\n";
		exit(1);
	}
	
	const size_t samples = atol(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	skepu::io::cout << "Samples: " << samples << "\n";

	auto montecarlo = skepu::MapReduce<0>(monte_carlo_sample, skepu::util::add<int>);
  
  skepu::PRNG prng;
  montecarlo.setPRNG(prng);
	montecarlo.setDefaultSize(samples);

	double pi = (double)montecarlo() / samples * 4;	
	
  skepu::io::cout << pi << "\n";
	
	return 0;
}

