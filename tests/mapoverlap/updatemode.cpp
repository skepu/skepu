#include <catch2/catch.hpp>

#include <skepu>


int over_2d(skepu::Region2D<int> r, const skepu::Mat<float> stencil)
{
	return r(1,0);
	float res = 0;
	for (int i = -r.oi; i <= r.oi; ++i)
		for (int j = -r.oj; j <= r.oj; ++j)
			res += r(i, j) * stencil(i + r.oi, j + r.oj);
	return res;
}

auto conv2 = skepu::MapOverlap(over_2d);


TEST_CASE("MapOverlap update mode")
{
	const size_t size{20};
  
  skepu::Matrix<int> m(size, size);
  skepu::Matrix<float> filter(2*1+1, 2*1+1, 1);
  skepu::external(
    skepu::read(m),
    [&]
    {
      int i = 0;
      for(size_t y = 0; y < size; ++y)
        for(size_t x = 0; x < size; ++x)
          m(y, x) = i++;
      std::cout << "m: " << m <<"\n";
    },
    skepu::write(m)
  );
  
  // Red-black update mode
  {
    conv2.setUpdateMode(skepu::UpdateMode::RedBlack);
    
    conv2.setEdgeMode(skepu::Edge::None);
    conv2(m, m, filter);
    std::cout << "Matrix 2D None:    rm = " << m << "\n";
  }
}