#include <catch2/catch.hpp>

#include <skepu>


auto square = skepu::Map<2>([](float a, float b) { return a * b; });

auto dotprod = skepu::MapReduce<2>(
  [](float a, float b) { return a * b; },
  [](float a, float b) { return a + b; }
);

auto sum =  skepu::Reduce([](float a, float b) { return a + b; });

auto sum2d =  skepu::Reduce([](float a, float b) { return a + b; }, [](float a, float b) { return a + b; });

auto cum_sum = skepu::Scan([](float a, float b) { return a + b; });

auto conv = skepu::MapOverlap([](skepu::Region1D<float> a)
{
  float sum = 0;
  for (int i = -a.oi; i <= a.oi; i++)
    sum += a(i);
  return sum;
});

auto conv2d = skepu::MapOverlap([](skepu::Region2D<float> m, const skepu::Mat<float> filter)
{
  float res = 0;
  for (int y = -m.oi; y <= m.oi; ++y)
    for (int x = -m.oj; x <= m.oj; ++x)
      res += m(y, x) * filter.data[(y+m.oi)*m.oj + (x+m.oj)];
  return res;
});

auto caller = skepu::Call([](skepu::Vec<float> r, float a, float b) { r[0] = a * b; });


TEST_CASE("Test skeleton instances using lambda expressions")
{
  size_t constexpr size{100};
	
	skepu::Vector<float> v1(size, 3), v2(size, 7), r(size);
	skepu::Matrix<float> m1(size, size, 3);
	
	square(r, v1, v2);
  r.flush();
  CHECK(r(0) == 21);
	
	square(r.begin(), r.end(), v1.begin(), v2.begin());
  r.flush();
  CHECK(r(0) == 21);
	
	square(r.begin(), r.end(), v1, v2);
  r.flush();
  CHECK(r(0) == 21);
	
	square(r, v1.begin(), v2);
  r.flush();
  CHECK(r(0) == 21);
	
  float res1 = dotprod(v1, v2);
  float res2 = sum(v1);
  float res3 = sum2d(m1);
	
  CHECK(res1 == 2100);
	
  CHECK(res2 == 300);
	
  CHECK(res3 == 30000);
	
	cum_sum(r, v1);
  r.flush();
  CHECK(r(0) == 3);
	
	conv.setOverlap(2);
	conv.setEdgeMode(skepu::Edge::Pad);
	conv.setPad(0);
	conv(r, v1);
  r.flush();
  CHECK(r(0) == 9);
	
	int o = 1;
	skepu::Matrix<float> filter(2*o+1, 2*o+1, 1), m(size, size, 5), rm(size, size);
	
	conv2d.setOverlap(o);
	conv2d.setEdgeMode(skepu::Edge::None);
	conv2d(rm, m, filter);
  rm.flush();
  CHECK(rm(o, o) == 45);
}

