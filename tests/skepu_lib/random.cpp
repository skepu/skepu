#include "../../external/catch2/catch.hpp"
//#define SKEPU_DEBUG_PRNG
#define SKEPU_DEBUG 3
#include <iostream>
#include <skepu>

#include <skepu-lib/io.hpp>


// TODO: Indirect user functions
// TODO: Higher random counts


int mapfunc(skepu::Random<1> &prng, int el)
{
  int randval = prng.get() % 100;
  return randval;
}

int mapfunc_idx(skepu::Index1D index, skepu::Random<1> &prng, int el)
{
  int randval = prng.get() % 100;
  return randval;
}

int mapfunc_0(skepu::Random<1> &prng)
{
  int randval = prng.get() % 100;
//  return indirect(prng);
  return randval;
}

int mapoverlapfunc_1d(skepu::Random<1> &prng, skepu::Region1D<int> r)
{
  int randval = prng.get() % 100;
  return randval;
}

int mapoverlapfunc_2d(skepu::Random<1> &prng, skepu::Region2D<int> r)
{
  int randval = prng.get() % 100;
  return randval;
}

int mapoverlapfunc_3d(skepu::Random<1> &prng, skepu::Region3D<int> r)
{
  int randval = prng.get() % 100;
  return randval;
}

int mapoverlapfunc_4d(skepu::Random<1> &prng, skepu::Region4D<int> r)
{
  int randval = prng.get() % 100;
  return randval;
}


int mappairsfunc(skepu::Random<1> &prng, int v, int h)
{
  int randval = prng.get() % 100;
//  return indirect(prng);
  return randval;
}

int redfunc(int a, int b) { return a + b; }


auto mapper1 = skepu::Map(mapfunc);
auto mapper2 = skepu::Map(mapfunc_idx);

auto mapper3 = skepu::Map([](skepu::Random<1> &prng, int el)
{
  int randval = prng.get() % 100;
  return randval;
});

auto mapper4 = skepu::Map([](skepu::Index1D index, skepu::Random<1> &prng, int el)
{
  int randval = prng.get() % 100;
  return randval;
});

auto mapper5 = skepu::MapReduce(mapfunc, redfunc);
auto mapper6 = skepu::MapReduce(mapfunc_0, redfunc);
auto mapper7 = skepu::MapPairs(mappairsfunc);
auto mapper8 = skepu::MapPairsReduce(mappairsfunc, redfunc);
auto mapoverlapper1d = skepu::MapOverlap(mapoverlapfunc_1d);
auto mapoverlapper2d = skepu::MapOverlap(mapoverlapfunc_2d);
auto mapoverlapper3d = skepu::MapOverlap(mapoverlapfunc_3d);
//auto mapoverlapper4d = skepu::MapOverlap(mapoverlapfunc_4d);

TEST_CASE("PRNG API: Map")
{
  size_t size{10};
  
  // Map
  {
    skepu::io::cout << "\n~~~ MAP ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng, ref_prng;
    mapper1.setPRNG(prng);
    
    mapper1(out, in);
    skepu::external(skepu::read(out), [&] { for (size_t i = 0; i < size; ++i) CHECK(out(i) == ref_prng.get() % 100); });
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << out << "\n";
  
    // Extract some values outside skeleton
    prng.get(); prng.get(); prng.get();
    ref_prng.get(); ref_prng.get(); ref_prng.get();
    CHECK(prng.get() == ref_prng.get());
  
    // Iterative Map
    const size_t iterations = 10;    
    for (size_t i = 0; i < iterations; ++i)
    {
      mapper1(in, in);
      skepu::external(skepu::read(in), [&] { for (size_t i = 0; i < size; ++i) CHECK(in(i) == ref_prng.get() % 100); });
      CHECK(prng.get() == ref_prng.get()); // next value should also be correct
    //  skepu::io::cout << "Result: " << in << "\n";
    }
  }
  
  // Map, indexed
  {
    skepu::io::cout << "\n~~~ MAP, INDEXED ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng, ref_prng;
    mapper2.setPRNG(prng);
    
    mapper2(out, in);
    skepu::external(skepu::read(out), [&] { for (size_t i = 0; i < size; ++i) CHECK(out(i) == ref_prng.get() % 100); });
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << out << "\n";
  }
  
  // Map, lambda
  {
    skepu::io::cout << "\n~~~ MAP, LAMBDA ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng, ref_prng;
    mapper3.setPRNG(prng);
    
    mapper3(out, in);
    skepu::external(skepu::read(out), [&] { for (size_t i = 0; i < size; ++i) CHECK(out(i) == ref_prng.get() % 100); });
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << out << "\n";
  }
  
  // Map, indexed, lambda
  {
    skepu::io::cout << "\n~~~ MAP, LAMBDA INDEXED ~~~\n";
    skepu::Vector<int> in(size, 0), out(size);
    
    skepu::PRNG prng, ref_prng;
    mapper4.setPRNG(prng);
    
    mapper4(out, in);
    skepu::external(skepu::read(out), [&] { for (size_t i = 0; i < size; ++i) CHECK(out(i) == ref_prng.get() % 100); });
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << out << "\n";
  }
}



TEST_CASE("PRNG API: MapReduce")
{
  size_t size{20};
  // MapReduce
  {
    skepu::io::cout << "\n~~~ MAPREDUCE ~~~\n";
    skepu::Vector<int> in(size, 0);
    
    skepu::PRNG prng, ref_prng;
    mapper5.setPRNG(prng);
    
    auto out = mapper5(in);
    int result = 0;
    for (size_t i = 0; i < size; ++i)
      result += ref_prng.get() % 100;
    CHECK(static_cast<bool>(out == result));
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << out << "\n";
  }
  
  // MapReduce<0>
  {
    skepu::io::cout << "\n~~~ MAPREDUCE<0> ~~~\n";
    mapper6.setDefaultSize(size);
    
    skepu::PRNG prng, ref_prng;
    mapper6.setPRNG(prng);
    
    auto out = mapper6();
    int result = 0;
    for (size_t i = 0; i < size; ++i)
      result += ref_prng.get() % 100;
    CHECK(static_cast<bool>(out == result));
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << out << "\n";
  }
}



TEST_CASE("PRNG API: MapPairs + MapPairsReduce")
{
  size_t sizeV{76}, sizeH{65};
  // MapPairs
  {
    skepu::io::cout << "\n~~~ MAPPAIRS ~~~\n";
    skepu::Vector<int> inV(sizeV, 0), inH(sizeH, 0);
    skepu::Matrix<int> out(sizeV, sizeH);
    
    skepu::PRNG prng, ref_prng;
    mapper7.setPRNG(prng);
    
    mapper7(out, inV, inH);
    skepu::external(skepu::read(out), [&]
    {
      for (size_t i = 0; i < sizeV; ++i)
        for (size_t j = 0; j < sizeH; ++j)
          CHECK(out(i, j) == ref_prng.get() % 100);
    });
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << out << "\n";
  }
  
  // MapPairsReduce
  {
    skepu::io::cout << "\n~~~ MAPPAIRSREDUCE ~~~\n";
    skepu::Vector<int> inV(sizeV, 0), inH(sizeH, 0), outV(sizeV), outH(sizeH);
    
    skepu::PRNG prng, ref_prng;
    mapper8.setPRNG(prng);
    
    mapper8.setReduceMode(skepu::ReduceMode::RowWise);
    mapper8(outV, inV, inH);
    skepu::external(skepu::read(outV), [&]
    {
      for (size_t i = 0; i < sizeV; ++i)
      {
        int result = 0;
        for (size_t j = 0; j < sizeH; ++j)
          result += ref_prng.get() % 100;
        CHECK(outV(i) == result);
      }
    });
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << outV << "\n";
    
    mapper8.setReduceMode(skepu::ReduceMode::ColWise);
    mapper8(outH, inV, inH);
    skepu::external(skepu::read(outH), [&] 
    {
      for (size_t j = 0; j < sizeH; ++j)
      {
        int result = 0;
        for (size_t i = 0; i < sizeV; ++i)
          result += ref_prng.get() % 100;
        CHECK(outH(j) == result);
      }
    });
    CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //  skepu::io::cout << "Result: " << outH << "\n";
  }
  
}

/*
TEST_CASE("PRNG API: MapOverlap 1D EdgeMode == None")
{
  size_t size_i{10}, size_j{9};
  size_t overlap{3};
  
  skepu::Vector<int> in(size_i, 1), out(size_i, 0);
  
  mapoverlapper1d.setOverlap(overlap);
  mapoverlapper1d.setEdgeMode(skepu::Edge::None);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper1d.setPRNG(prng);
  
  mapoverlapper1d(out, in);
  skepu::external(skepu::read(out), [&] { for (size_t i = overlap; i < size_i - overlap; ++i) CHECK(out(i) == ref_prng.get() % 100); });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  skepu::io::cout << "Result: " << out << "\n";
  
  
  skepu::Matrix<int> in_m(size_i, size_j, 1), out_mR(size_i, size_j, 0), out_mC(size_i, size_j, 0);
  
  mapoverlapper1d.setOverlapMode(skepu::Overlap::RowWise);
  mapoverlapper1d(out_mR, in_m);
  skepu::external(skepu::read(out_mR), [&] 
  {
    for (size_t i = 0; i < size_i; ++i)
      for (size_t j = overlap; j < size_j - overlap; ++j)
        CHECK(out_mR(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  skepu::io::cout << "Result: " << out_mR << "\n";
  
  
  mapoverlapper1d.setOverlapMode(skepu::Overlap::ColWise);
  mapoverlapper1d(out_mC, in_m);
  skepu::external(skepu::read(out_mC), [&] 
  {
    for (size_t j = 0; j < size_j; ++j)
      for (size_t i = overlap; i < size_i - overlap; ++i)
        CHECK(out_mC(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  skepu::io::cout << "Result: " << out_mC << "\n";
  
}


TEST_CASE("PRNG API: MapOverlap 1D EdgeMode != None")
{
  size_t size_i{101}, size_j{65};
  
  skepu::Vector<int> in(size_i, 1), out(size_i);
  
  mapoverlapper1d.setEdgeMode(skepu::Edge::Cyclic);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper1d.setPRNG(prng);
  
  mapoverlapper1d(out, in);
  skepu::external(skepu::read(out), [&] { for (size_t i = 0; i < size_i; ++i) CHECK(out(i) == ref_prng.get() % 100); });
//  skepu::io::cout << "Result: " << out << "\n";
  
  
  skepu::Matrix<int> in_m(size_i, size_j, 1), out_m(size_i, size_j);
  
  mapoverlapper1d.setOverlapMode(skepu::Overlap::RowWise);
  mapoverlapper1d(out_m, in_m);
  skepu::external(skepu::read(out_m), [&] 
  {
    for (size_t i = 0; i < size_i; ++i)
      for (size_t j = 0; j < size_j; ++j)
        CHECK(out_m(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
//  skepu::io::cout << "Result: " << out_m << "\n";
  
  
  mapoverlapper1d.setOverlapMode(skepu::Overlap::ColWise);
  mapoverlapper1d(out_m, in_m);
  skepu::external(skepu::read(out_m), [&] 
  {
    for (size_t j = 0; j < size_j; ++j)
      for (size_t i = 0; i < size_i; ++i)
        CHECK(out_m(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
//  skepu::io::cout << "Result: " << out_m << "\n";
  
}
*/  

TEST_CASE("PRNG API: MapOverlap 2D EdgeMode == None")
{
  size_t size_i{58}, size_j{27};
  size_t overlap_i{3}, overlap_j{3};
  
  skepu::Matrix<int> in(size_i, size_j, 1), out(size_i, size_j, 0);
  
  mapoverlapper2d.setOverlap(overlap_i, overlap_j);
  mapoverlapper2d.setEdgeMode(skepu::Edge::None);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper2d.setPRNG(prng);
  
  mapoverlapper2d(out, in);
  skepu::external(skepu::read(out), [&] 
  {
    for (size_t i = overlap_i; i < size_i - overlap_i; ++i)
      for (size_t j = overlap_j; j < size_j - overlap_j; ++j)
        CHECK(out(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //skepu::io::cout << "Result: " << out << "\n";
}


TEST_CASE("PRNG API: MapOverlap 2D EdgeMode != None")
{
  size_t size_i{58}, size_j{27};
  skepu::Matrix<int> in(size_i, size_j, 1), out(size_i, size_j);
  
  mapoverlapper2d.setEdgeMode(skepu::Edge::Cyclic);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper2d.setPRNG(prng);
  
  mapoverlapper2d(out, in);
  skepu::external(skepu::read(out), [&] 
  {
    for (size_t i = 0; i < size_i; ++i)
      for (size_t j = 0; j < size_j; ++j)
        CHECK(out(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //skepu::io::cout << "Result: " << out << "\n";
}


TEST_CASE("PRNG API: MapOverlap 3D EdgeMode == None")
{
  size_t size_i{38}, size_j{27}, size_k{15};
  size_t overlap_i{3}, overlap_j{3}, overlap_k{3};
  
  skepu::Tensor3<int> in(size_i, size_j, size_k, "", 1), out(size_i, size_j, size_k);
  
  mapoverlapper3d.setOverlap(overlap_i, overlap_j, overlap_k);
  mapoverlapper3d.setEdgeMode(skepu::Edge::None);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper3d.setPRNG(prng);
  
  mapoverlapper3d(out, in);
  skepu::external(skepu::read(out), [&] 
  {
    for (size_t i = overlap_i; i < size_i - overlap_i; ++i)
      for (size_t j = overlap_j; j < size_j - overlap_j; ++j)
        for (size_t k = overlap_k; k < size_k - overlap_k; ++k)
          CHECK(out(i, j, k) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
//  skepu::io::cout << "Result: " << out << "\n";
}


TEST_CASE("PRNG API: MapOverlap 3D EdgeMode != None")
{
  size_t size_i{38}, size_j{27}, size_k{15};
  skepu::Tensor3<int> in(size_i, size_j, size_k, "", 1), out(size_i, size_j, size_k);
  
  mapoverlapper3d.setEdgeMode(skepu::Edge::Cyclic);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper3d.setPRNG(prng);
  
  mapoverlapper3d(out, in);
  skepu::external(skepu::read(out), [&] 
  {
    for (size_t i = 0; i < size_i; ++i)
      for (size_t j = 0; j < size_j; ++j)
        for (size_t k = 0; k < size_k; ++k)
          CHECK(out(i, j, k) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
//  skepu::io::cout << "Result: " << out << "\n";
}


/*
// MapOverlap 4D
{
  skepu::io::cout << "\n~~~ MAPOVERLAP 4D ~~~\n";
  size_t size_i{38}, size_j{27}, size_k{17}, size_l{9};
  skepu::Tensor4<int> in(size_i, size_j, size_k, size_l, 1), out(size_i, size_j, size_k, size_l);
  
  mapoverlapper4d.setEdgeMode(skepu::Edge::Cyclic);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper4d.setPRNG(prng);
  
  mapoverlapper4d(out, in);
  skepu::external(skepu::read(out), [&] 
  {
    for (size_t i = 0; i < size_i; ++i)
      for (size_t j = 0; j < size_j; ++j)
        for (size_t k = 0; k < size_k; ++k)
          for (size_t l = 0; l < size_l; ++l)
            CHECK(out(i, j, k, l) == ref_prng.get() % 100);
  });
//  skepu::io::cout << "Result: " << out << "\n";
}*/
