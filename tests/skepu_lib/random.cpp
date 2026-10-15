// Each part is in its own namespace; its test cases are prefixed with its name.

#include "../../external/catch2/catch.hpp"
#include <skepu>
#include <iostream>
#include <skepu-lib/io.hpp>
#include <test_helper_automatic_setup.hpp>
#include <skepu-lib/complex.hpp>
#include <skepu-lib/util.hpp>
#include <cmath>
#include <skepu-lib/randomize.hpp>


// ======== random ========

namespace random_part
{
//#define SKEPU_DEBUG_PRNG


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

TEST_CASE("random: PRNG API: Map")
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



TEST_CASE("random: PRNG API: MapReduce")
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



TEST_CASE("random: PRNG API: MapPairs + MapPairsReduce")
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

// The edge mode None makes an output that is 2 * overlap smaller than the input
TEST_CASE("random: PRNG API: MapOverlap 1D EdgeMode == None")
{
  size_t size_i{10}, size_j{9};
  size_t overlap{3};
  
  skepu::Vector<int> in(size_i + 2 * overlap, 1), out(size_i, 0);
  
  mapoverlapper1d.setOverlap(overlap);
  mapoverlapper1d.setEdgeMode(skepu::Edge::None);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper1d.setPRNG(prng);
  
  mapoverlapper1d(out, in);
  skepu::external(skepu::read(out), [&] { for (size_t i = 0; i < size_i; ++i) CHECK(out(i) == ref_prng.get() % 100); });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  
  
  skepu::Matrix<int> in_mR(size_i, size_j + 2 * overlap, 1), in_mC(size_i + 2 * overlap, size_j, 1), out_mR(size_i, size_j, 0), out_mC(size_i, size_j, 0);
  
  mapoverlapper1d.setOverlapMode(skepu::Overlap::RowWise);
  mapoverlapper1d(out_mR, in_mR);
  skepu::external(skepu::read(out_mR), [&] 
  {
    for (size_t i = 0; i < size_i; ++i)
      for (size_t j = 0; j < size_j; ++j)
        CHECK(out_mR(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  
  
  mapoverlapper1d.setOverlapMode(skepu::Overlap::ColWise);
  mapoverlapper1d(out_mC, in_mC);
  skepu::external(skepu::read(out_mC), [&] 
  {
    for (size_t j = 0; j < size_j; ++j)
      for (size_t i = 0; i < size_i; ++i)
        CHECK(out_mC(i, j) == ref_prng.get() % 100);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
}


TEST_CASE("random: PRNG API: MapOverlap 1D EdgeMode != None")
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

// Enough elements for the backends that share the work between devices to use more than one (the numbers of the elements must not depend on that)
TEST_CASE("random: PRNG API: MapOverlap 1D with a large vector")
{
  size_t size{200000};
  size_t overlap{2};
  
  skepu::Vector<int> in(size, 1), out(size, 0), out_none(size - 2 * overlap, 0);
  
  mapoverlapper1d.setOverlap(overlap);
  mapoverlapper1d.setOverlapMode(skepu::Overlap::RowWise);
  
  skepu::PRNG prng, ref_prng;
  mapoverlapper1d.setPRNG(prng);
  
  mapoverlapper1d.setEdgeMode(skepu::Edge::Cyclic);
  mapoverlapper1d(out, in);
  skepu::external(skepu::read(out), [&] 
  {
    size_t wrong{0};
    for (size_t i = 0; i < size; ++i)
      if (out(i) != ref_prng.get() % 100)
        ++wrong;
    CHECK(wrong == 0);
  });
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  
  mapoverlapper1d.setEdgeMode(skepu::Edge::None);
  mapoverlapper1d(out_none, in);
  skepu::external(skepu::read(out_none), [&] 
  {
    size_t wrong{0};
    for (size_t i = 0; i < size - 2 * overlap; ++i)
      if (out_none(i) != ref_prng.get() % 100)
        ++wrong;
    CHECK(wrong == 0);
  });
  CHECK(prng.get() == ref_prng.get());
}


TEST_CASE("random: PRNG API: MapOverlap 2D EdgeMode == None")
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
  // (the inner elements get the numbers in order; the stream is advanced by the number of elements of the output)
  ref_prng.forward(size_i * size_j - (size_i - 2 * overlap_i) * (size_j - 2 * overlap_j));
  CHECK(prng.get() == ref_prng.get()); // next value should also be correct
  //skepu::io::cout << "Result: " << out << "\n";
}


TEST_CASE("random: PRNG API: MapOverlap 2D EdgeMode != None")
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


TEST_CASE("random: PRNG API: MapOverlap 3D EdgeMode == None")
{
  size_t size_i{38}, size_j{27}, size_k{15};
  size_t overlap_i{3}, overlap_j{3}, overlap_k{3};
  
  // (the edge mode None makes an output that is 2 * overlap smaller than the input)
  skepu::Tensor3<int> in(size_i + 2 * overlap_i, size_j + 2 * overlap_j, size_k + 2 * overlap_k, "", 1), out(size_i, size_j, size_k);
  
  mapoverlapper3d.setOverlap(overlap_i, overlap_j, overlap_k);
  mapoverlapper3d.setEdgeMode(skepu::Edge::None);
  
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


TEST_CASE("random: PRNG API: MapOverlap 3D EdgeMode != None")
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

TEST_CASE("random: PRNG constructors: a name and a seed")
{
  // (a name is the label that a trace shows; the stream has the seed that PRNG() has when none is given)
  skepu::PRNG plain, named("named"), seeded(7), both(7, "both"), named_first("first", 7);
  for (int i = 0; i < 5; ++i)
  {
    auto a = plain.get(), b = named.get(), c = seeded.get(), d = both.get(), e = named_first.get();
    CHECK(a == b);
    CHECK(c == d);
    CHECK(c == e);
    CHECK(a != c);
  }
}
}


// ======== using_namespace ========

namespace using_namespace_part
{
float test_uf(float a, float b){ return a + b;}


auto test1 = skepu::Map(test_uf);
auto test2 = skepu::Map([](float a, float b){ return a + b;});
auto test3 = skepu::Map<1>(test_uf);
auto test4 = skepu::Map<1>([](float a, float b){ return a + b;});


TEST_CASE("using_namespace: Standard library namespaces")
{
	const size_t size{1000};
	
	skepu::Vector<float> a(size, size), b(size, size), c(size, size);
	
	for (size_t i = 0; i < size; ++i)
	{
		a(i) = i;
		b(i) = 1;
	}
	
	// c is checked after each call: a(i) is i and b(i) is 1
	auto check = [&](auto expected)
	{
		skepu::external(skepu::read(c), [&]
		{
			for (size_t i = 0; i < size; ++i)
				CHECK(c(i) == Approx(expected(i)));
		});
	};
	
	test1(c, a, b);
	check([](size_t i){ return i + 1.f; });
	
	test2(c, a, b);
	check([](size_t i){ return i + 1.f; });
	
	test3(c, a, 100);
	check([](size_t i){ return i + 100.f; });
	
	test4(c, a, 100);
	check([](size_t i){ return i + 100.f; });
	
	// vector sum
	skepu::Map([](float lhs, float rhs){ return lhs + rhs;})(c, a, b);
	check([](size_t i){ return i + 1.f; });
	
	// vector scale (library function)
	skepu::Map<1>(skepu::util::mul<float>)(c, a, 2);
	check([](size_t i){ return i * 2.f; });
	
	using namespace skepu;
	
	// sum (library function + using namespace)
	float sum = Reduce(util::add<float>)(a);
	CHECK(sum == Approx(size * (size - 1) / 2.f));
}
}


// ======== randomize ========

namespace randomize_part
{
// The fills of skepu::randomize against what a single stream gives, element by element, on every backend: the values
// do not depend on how the backend splits the work.

namespace
{
	// 62 random bits from two draws of a stream, as the fills make them
	unsigned long long bits62(skepu::PRNG &stream)
	{
		unsigned long long high = stream.get();
		unsigned long long low = stream.get();
		return (high << 31) | low;
	}

	void require_same_state(skepu::PRNG &used, skepu::PRNG &reference)
	{
		// the next number of the two streams is the same: the fill advanced the stream as much as the reference did
		REQUIRE(used.get() == reference.get());
	}
}

TEST_CASE("randomize: randomize::uniform, integers, the values of the stream, in all shapes")
{
	const size_t n = 1000;
	skepu::PRNG stream(7), reference(7);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<int> v(n);
	skepu::randomize::uniform(v, -50, 49);
	v.updateHost();
	for (size_t i = 0; i < n; ++i)
		REQUIRE(v(i) == -50 + (int)(bits62(reference) % 100));
	require_same_state(stream, reference); // (one number on in both)

	skepu::Matrix<long> m(20, 30);
	skepu::randomize::uniform(m, 99);
	m.updateHost();
	for (size_t i = 0; i < 20; ++i)
		for (size_t j = 0; j < 30; ++j)
			REQUIRE(m(i, j) == (long)(bits62(reference) % 100));

	skepu::Tensor3<unsigned> t3(4, 5, 6);
	skepu::randomize::uniform(t3, 9u);
	t3.updateHost();
	for (size_t i = 0; i < 4 * 5 * 6; ++i)
		REQUIRE(t3.getAddress()[i] == (unsigned)(bits62(reference) % 10));

	skepu::Tensor4<short> t4(3, 4, 2, 5);
	skepu::randomize::uniform(t4, (short)3, (short)7);
	t4.updateHost();
	for (size_t i = 0; i < 3 * 4 * 2 * 5; ++i)
		REQUIRE(t4.getAddress()[i] == (short)(3 + bits62(reference) % 5));
}

TEST_CASE("randomize: randomize::uniform, floating point, the values of the stream, and the defaults")
{
	const size_t n = 1000;
	skepu::PRNG stream(11), reference(11);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<float> v(n);
	skepu::randomize::uniform(v);
	v.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		const float want = (float)(bits62(reference) >> 38) * (1.0f / 16777216.0f);
		REQUIRE(v(i) == want);
		REQUIRE(v(i) >= 0.f);
		REQUIRE(v(i) < 1.f);
	}

	skepu::Matrix<float> m(30, 20);
	skepu::randomize::uniform(m, -2.f, 6.f);
	m.updateHost();
	for (size_t i = 0; i < 600; ++i)
	{
		const float want = -2.f + (float)(bits62(reference) >> 38) * (1.0f / 16777216.0f) * 8.f;
		REQUIRE(m.getAddress()[i] == want);
	}
}

TEST_CASE("randomize: randomize, the stream: the program's, a set one, a scope, and the order of the calls")
{
	skepu::PRNG a(3), b(3), other(99);
	skepu::Vector<int> first(100), second(100), both(200);

	{
		skepu::randomize::Scope scope(a);
		skepu::randomize::uniform(first, 1000);
		skepu::randomize::uniform(second, 1000); // the stream goes on
	}
	skepu::randomize::set_prng(b);
	skepu::randomize::uniform(both, 1000);
	skepu::randomize::reset_prng();

	first.updateHost(); second.updateHost(); both.updateHost();
	for (size_t i = 0; i < 100; ++i)
	{
		REQUIRE(both(i) == first(i));
		REQUIRE(both(100 + i) == second(i));
	}

	// without a stream set, the one of the program is used (and advanced)
	skepu::PRNG &global = skepu::getGlobalPRNG();
	skepu::PRNG expected(SKEPU_RAND_DEFAULT_SEED);
	REQUIRE(&skepu::randomize::prng() == &global);
	skepu::Vector<int> c(10);
	skepu::randomize::uniform(c, 10);
	c.updateHost();
	for (size_t i = 0; i < 10; ++i)
		REQUIRE(c(i) == (int)(bits62(expected) % 11));

	// a scope restores what was before it
	skepu::randomize::set_prng(a);
	{
		skepu::randomize::Scope scope(other);
		REQUIRE(&skepu::randomize::prng() == &other);
	}
	REQUIRE(&skepu::randomize::prng() == &a);
	skepu::randomize::reset_prng();
}

TEST_CASE("randomize: randomize::normal, the moments")
{
	const size_t n = 200000;
	skepu::PRNG stream(13);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<float> v(n);
	skepu::randomize::normal(v, 3.f, 2.f);
	v.updateHost();
	double sum = 0, sum2 = 0;
	for (size_t i = 0; i < n; ++i) { sum += v(i); sum2 += (double)v(i) * v(i); }
	const double mean = sum / n, variance = sum2 / n - mean * mean;
	REQUIRE(std::fabs(mean - 3.0) < 0.03);
	REQUIRE(std::fabs(std::sqrt(variance) - 2.0) < 0.03);

	// the standard normal is the default, and the shape does not matter
	skepu::Matrix<float> m(400, 500);
	skepu::randomize::normal(m);
	m.updateHost();
	sum = sum2 = 0;
	for (size_t i = 0; i < n; ++i) { sum += m.getAddress()[i]; sum2 += (double)m.getAddress()[i] * m.getAddress()[i]; }
	REQUIRE(std::fabs(sum / n) < 0.02);
	REQUIRE(std::fabs(std::sqrt(sum2 / n) - 1.0) < 0.02);
}

TEST_CASE("randomize: randomize::bernoulli and randomize::poisson, the moments")
{
	const size_t n = 100000;
	skepu::PRNG stream(17);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<int> b(n);
	skepu::randomize::bernoulli(b, 0.3);
	b.updateHost();
	size_t ones = 0;
	for (size_t i = 0; i < n; ++i) { REQUIRE((b(i) == 0 || b(i) == 1)); ones += b(i); }
	REQUIRE(std::fabs((double)ones / n - 0.3) < 0.01);

	skepu::randomize::bernoulli(b, 0.0);
	b.updateHost();
	for (size_t i = 0; i < n; ++i) REQUIRE(b(i) == 0);
	skepu::randomize::bernoulli(b, 1.0);
	b.updateHost();
	for (size_t i = 0; i < n; ++i) REQUIRE(b(i) == 1);

	skepu::Vector<int> p(n);
	skepu::randomize::poisson(p, 4.0);
	p.updateHost();
	double sum = 0, sum2 = 0;
	for (size_t i = 0; i < n; ++i) { REQUIRE(p(i) >= 0); sum += p(i); sum2 += (double)p(i) * p(i); }
	const double mean = sum / n;
	REQUIRE(std::fabs(mean - 4.0) < 0.05);
	REQUIRE(std::fabs((sum2 / n - mean * mean) - 4.0) < 0.1);
}

// Double precision: on an OpenCL device that does not have it, the skeletons with doubles run on another backend

TEST_CASE("randomize: randomize::uniform, double precision")
{
	const size_t n = 500;
	skepu::PRNG stream(5), reference(5);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<double> v(n);
	skepu::randomize::uniform(v, 10.0, 20.0);
	v.updateHost();
	for (size_t i = 0; i < n; ++i)
	{
		const double want = 10.0 + (double)(bits62(reference) >> 9) * std::ldexp(1.0, -53) * 10.0;
		REQUIRE(v(i) == want);
	}
}


TEST_CASE("randomize: randomize::normal, double precision")
{
	const size_t n = 100000;
	skepu::PRNG stream(21);
	skepu::randomize::Scope scope(stream);

	skepu::Vector<double> v(n);
	skepu::randomize::normal(v, -1.0, 0.5);
	v.updateHost();
	double sum = 0, sum2 = 0;
	for (size_t i = 0; i < n; ++i) { sum += v(i); sum2 += v(i) * v(i); }
	const double mean = sum / n;
	REQUIRE(std::fabs(mean + 1.0) < 0.01);
	REQUIRE(std::fabs(std::sqrt(sum2 / n - mean * mean) - 0.5) < 0.01);
}
}
