#include <iostream>
#include <skepu>


int indirect(skepu::Random<> &&prng)
{
  return prng.get() % 100;
}


int mapfunc(skepu::Random<2> &prng, int el)
{
  int randval = (prng.get(), prng.get()) % 100;
//  return indirect(prng);
  return randval;
}

int mapoverlapfunc(skepu::Random<2> &prng, skepu::Region4D<int> r)
{
  int randval = (prng.get(), prng.get()) % 100;
  return randval;
}


int main(int argc, char *argv[])
{
  if (argc < 2)
	{
		if(!skepu::cluster::mpi_rank())
			std::cout << "Usage: " << argv[0] << " size backend\n";
		exit(1);
	}
  
  size_t size = atoi(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
//  spec.setCPUThreads(3);
	skepu::setGlobalBackendSpec(spec);
  
  // Map
  {
    skepu::Vector<int> in(size, 0), out(size);
    
    auto mapper = skepu::Map(mapfunc);
    
    skepu::PRNG prng;
    mapper.setPRNG(prng);
    
    mapper(out, in);
    std::cout << "Result: " << out << "\n";
  
    // Iterative Map
    const size_t iterations = 10;    
    for (size_t i = 0; i < iterations; ++i)
    {
      mapper(in, in);
      std::cout << "Result: " << in << "\n";
    }
    
    
  }
  
  
  /*
  // MapOverlap 4D
  {
    size = 4;
    skepu::Tensor4<int> in(size, size, size, size, 1), out(size, size, size, size);
  
    auto mapoverlapper = skepu::MapOverlap(mapoverlapfunc);
    mapoverlapper.setEdgeMode(skepu::Edge::Cyclic);
    
    skepu::PRNG prng;
    mapoverlapper.setPRNG(prng);
    
    mapoverlapper(out, in);
    std::cout << "Result: " << out << "\n";
    
    mapoverlapper(out, in);
    std::cout << "Result: " << out << "\n";
  }*/
  
  
  return 0;
}