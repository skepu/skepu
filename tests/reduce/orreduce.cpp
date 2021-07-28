#include <catch2/catch.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <skepu>


bool or_reduce (bool a, bool b)
{
	return (a || b);
}

auto reduce = skepu::Reduce(or_reduce);

TEST_CASE("Reduce with 'or' user function")
{
  const size_t size{100};

  skepu::Vector<bool> bool_cond(size, false);
  bool_cond[0] = true;
  reduce.setStartValue(false);
  auto res = reduce(bool_cond);
}