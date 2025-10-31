#include "../../catch2/catch.hpp"

#include <skepu>
#include <skepu-lib/io.hpp>


int updatemode_1d_uf(skepu::Region1D<int> r)
{
	return 1;
}

int updatemode_2d_uf(skepu::Region2D<int> r)
{
	return 1;
}


auto updatemode_1d = skepu::MapOverlap(updatemode_1d_uf);
auto updatemode_2d = skepu::MapOverlap(updatemode_2d_uf);


TEST_CASE("MapOverlap update mode 1D")
{
	const size_t size{17};
  
  skepu::Vector<int> v(size), resR(size, 0), resB(size, 0), resRB(size, 0);
  
	updatemode_1d.setEdgeMode(skepu::Edge::Pad);
  
  updatemode_1d.setUpdateMode(skepu::UpdateMode::Red);
  updatemode_1d(resR, v);
	
  skepu::external(skepu::read(resR), [&]
	{
    for (size_t i = 0; i < size; ++i)
      CHECK(resR(i) == ((i % 2 == 0) ? 0 : 1));
  });
	skepu::io::cout << resR << "\n";
	
	updatemode_1d.setUpdateMode(skepu::UpdateMode::Black);
  updatemode_1d(resB, v);
	
  skepu::external(skepu::read(resB), [&]
	{
    for (size_t i = 0; i < size; ++i)
      CHECK(resB(i) == ((i % 2 == 0) ? 1 : 0));
  });
	skepu::io::cout << resB << "\n";
	
	
  updatemode_1d.setUpdateMode(skepu::UpdateMode::RedBlack);
  updatemode_1d(resRB, v);
	
  skepu::external(skepu::read(resRB), [&]
	{
    for (size_t i = 0; i < size; ++i)
      CHECK(resRB(i) == 1);
  });
	skepu::io::cout << resRB << "\n";
	
}


TEST_CASE("MapOverlap update mode 2D")
{
	const size_t size_i{20}, size_j{31};
  
  skepu::Matrix<int> m(size_i, size_j), resR(size_i, size_j, 0), resB(size_i, size_j, 0), resRB(size_i, size_j, 0);
  
	updatemode_2d.setEdgeMode(skepu::Edge::Pad);
  
  updatemode_2d.setUpdateMode(skepu::UpdateMode::Red);
  updatemode_2d(resR, m);
	
  skepu::external(skepu::read(resR), [&]
	{
    for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
      	CHECK(resR(i, j) == (((i + j) % 2 == 0) ? 0 : 1));
  });
	skepu::io::cout << resR << "\n";
	
	updatemode_2d.setUpdateMode(skepu::UpdateMode::Black);
  updatemode_2d(resB, m);
	
  skepu::external(skepu::read(resB), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
      	CHECK(resB(i, j) == (((i + j) % 2 == 0) ? 1 : 0));
  });
	skepu::io::cout << resB << "\n";
	
	
  updatemode_2d.setUpdateMode(skepu::UpdateMode::RedBlack);
  updatemode_2d(resRB, m);
	
  skepu::external(skepu::read(resRB), [&]
	{
		for (size_t i = 0; i < size_i; ++i)
			for (size_t j = 0; j < size_j; ++j)
      	CHECK(resRB(i, j) == 1);
  });
	skepu::io::cout << resRB << "\n";
	
}