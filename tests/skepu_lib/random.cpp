#include <catch2/catch.hpp>

#include <iostream>
#include <skepu>

#include <skepu-lib/io.hpp>



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

int mapfunc_idx(skepu::Index1D index, skepu::Random<2> &prng, int el)
{
  int randval = (prng.get(), prng.get()) % 100;
//  return indirect(prng);
  return randval;
}

int mapfunc_0(skepu::Random<2> &prng)
{
  int randval = (prng.get(), prng.get()) % 100;
//  return indirect(prng);
  return randval;
}

int mapoverlapfunc_1d(skepu::Random<2> &prng, skepu::Region1D<int> r)
{
  int randval = (prng.get(), prng.get()) % 100;
  return randval;
}

int mapoverlapfunc_2d(skepu::Random<2> &prng, skepu::Region2D<int> r)
{
  int randval = (prng.get(), prng.get()) % 100;
  return randval;
}

int mapoverlapfunc_3d(skepu::Random<2> &prng, skepu::Region3D<int> r)
{
  int randval = (prng.get(), prng.get()) % 100;
  return randval;
}

int mapoverlapfunc_4d(skepu::Random<2> &prng, skepu::Region4D<int> r)
{
  int randval = (prng.get(), prng.get()) % 100;
  return randval;
}


int mappairsfunc(skepu::Random<2> &prng, int v, int h)
{
  int randval = (prng.get(), prng.get()) % 100;
//  return indirect(prng);
  return randval;
}

int redfunc(int a, int b) { return a + b; }


auto mapper1 = skepu::Map(mapfunc);
auto mapper2 = skepu::Map(mapfunc_idx);

auto mapper3 = skepu::Map([](skepu::Random<2> &prng, int el)
{
  int randval = (prng.get(), prng.get()) % 100;
  return randval;
});

auto mapper4 = skepu::Map([](skepu::Index1D index, skepu::Random<2> &prng, int el)
{
  int randval = (prng.get(), prng.get()) % 100;
  return randval;
});

auto mapper5 = skepu::MapReduce(mapfunc, redfunc);
auto mapper6 = skepu::MapReduce(mapfunc_0, redfunc);
auto mapper7 = skepu::MapPairs(mappairsfunc);
auto mapoverlapper1d = skepu::MapOverlap(mapoverlapfunc_1d);
auto mapoverlapper2d = skepu::MapOverlap(mapoverlapfunc_2d);

TEST_CASE("PRNG API")
{
  size_t size{20};
  
  // Map
  {
    std::cout << "\n~~~ MAP ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng;
    mapper1.setPRNG(prng);
    
    mapper1(out, in);
    skepu::io::cout << "Result: " << out << "\n";
  
    // Iterative Map
    const size_t iterations = 10;    
    for (size_t i = 0; i < iterations; ++i)
    {
      mapper1(in, in);
      skepu::io::cout << "Result: " << in << "\n";
    }
  }
  
  // Map, indexed
  {
    skepu::io::cout << "\n~~~ MAP, INDEXED ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng;
    mapper2.setPRNG(prng);
    
    mapper2(out, in);
    skepu::io::cout << "Result: " << out << "\n";
  }
  
  // Map, lambda
  {
    std::cout << "\n~~~ MAP, LAMBDA ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng;
    mapper3.setPRNG(prng);
    
    mapper3(out, in);
    skepu::io::cout << "Result: " << out << "\n";
  }
  
  // Map, indexed, lambda
  {
    skepu::io::cout << "\n~~~ MAP, LAMBDA INDEXED ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng;
    mapper4.setPRNG(prng);
    
    mapper4(out, in);
    std::cout << "Result: " << out << "\n";
  }
  
  // MapReduce
  {
    skepu::io::cout << "\n~~~ MAPREDUCE ~~~\n";
    skepu::Vector<int> in(size, 0);
    
    skepu::PRNG prng;
    mapper5.setPRNG(prng);
    
    auto out = mapper5(in);
    std::cout << "Result: " << out << "\n";
  }
  
  // MapReduce<0>
  {
    skepu::io::cout << "\n~~~ MAPREDUCE<0> ~~~\n";
    mapper6.setDefaultSize(size);
    
    skepu::PRNG prng;
    mapper6.setPRNG(prng);
    
    auto out = mapper6();
    skepu::io::cout << "Result: " << out << "\n";
  }
  
  // MapPairs
  {
    skepu::io::cout << "\n~~~ MAP, INDEXED ~~~\n";
    skepu::Vector<int> inV(size, 0), inH(size, 0);
    skepu::Matrix<int> out(size, size);
    
    skepu::PRNG prng;
    mapper7.setPRNG(prng);
    
    mapper7(out, inV, inH);
    skepu::io::cout << "Result: " << out << "\n";
  }
  /*
  // MapPairsReduce
  {
    std::cout << "\n~~~ MAP, INDEXED ~~~\n";
    skepu::Vector<int> inV(size, 0), inH(size, 0), out(size);
    
    auto mapper = skepu::MapPairsReduce(mappairsfunc, redfunc);
    
    skepu::PRNG prng;
    mapper.setPRNG(prng);
    
    mapper(out, inV, inH);
    std::cout << "Result: " << out << "\n";
  }
  */
  // MapOverlap 1D
  {
    std::cout << "\n~~~ MAPOVERLAP 1D ~~~\n";
    skepu::Vector<int> in(size, 1), out(size);
    
    mapoverlapper1d.setEdgeMode(skepu::Edge::Cyclic);
    
    skepu::PRNG prng;
    mapoverlapper1d.setPRNG(prng);
    
    mapoverlapper1d(out, in);
    std::cout << "Result A: " << out << "\n";
    
    mapoverlapper1d(out, in);
    std::cout << "Result B: " << out << "\n";
    
    
    skepu::Matrix<int> in_m(size, size, 1), out_m(size, size);
    
    mapoverlapper1d(out_m, in_m);
    std::cout << "Result C: " << out_m << "\n";
    
    mapoverlapper1d(out_m, in_m);
    std::cout << "Result D: " << out_m << "\n";
    
  }
  
  // MapOverlap 2D
  {
    std::cout << "\n~~~ MAPOVERLAP 2D ~~~\n";
    size = 8;
    skepu::Matrix<int> in(size, size, 1), out(size, size);
    
    mapoverlapper2d.setEdgeMode(skepu::Edge::Cyclic);
    
    skepu::PRNG prng;
    mapoverlapper2d.setPRNG(prng);
    
    mapoverlapper2d(out, in);
    std::cout << "Result A: " << out << "\n";
    
    mapoverlapper2d(out, in);
    std::cout << "Result B: " << out << "\n";
  }
  
  
  /*
  // MapOverlap 3D
  {
    std::cout << "\n~~~ MAPOVERLAP 3D ~~~\n";
    size = 4;
    skepu::Tensor3<int> in(size, size, size, 1), out(size, size, size);
    
    auto mapoverlapper = skepu::MapOverlap(mapoverlapfunc_3d);
    mapoverlapper.setEdgeMode(skepu::Edge::Cyclic);
    
    skepu::PRNG prng;
    mapoverlapper.setPRNG(prng);
    
    mapoverlapper(out, in);
    std::cout << "Result: " << out << "\n";
    
    mapoverlapper(out, in);
    std::cout << "Result: " << out << "\n";
  }
  
  // MapOverlap 4D
  {
    std::cout << "\n~~~ MAPOVERLAP 4D ~~~\n";
    size = 4;
    skepu::Tensor4<int> in(size, size, size, size, 1), out(size, size, size, size);
    
    auto mapoverlapper = skepu::MapOverlap(mapoverlapfunc_4d);
    mapoverlapper.setEdgeMode(skepu::Edge::Cyclic);
    
    skepu::PRNG prng;
    mapoverlapper.setPRNG(prng);
    
    mapoverlapper(out, in);
    std::cout << "Result: " << out << "\n";
    
    mapoverlapper(out, in);
    std::cout << "Result: " << out << "\n";
  }*/
  
}