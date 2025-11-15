#include "../../catch2/catch.hpp"
#define SKEPU_DEBUG 3
#include <skepu>
#include <skepu-lib/util.hpp>
#include <skepu-lib/io.hpp>

void check_values(skepu::Vector<int> v, int expected)
{
    for (size_t i = 0; i < v.size(); ++i)
        CHECK(v(i) == expected);
}

int uf_int(int a)
{
    return a;
}

int uf_add(int a)
{
    return a + 1;
}

auto copy = skepu::Map(uf_int);

auto addOne = skepu::Map(uf_add);
	
TEST_CASE("skepu::external")
{
	constexpr size_t size{10};
	
	skepu::Vector<int> v(size, 1), r(size, 0);

    // write
    skepu::external([&]{
        for (size_t i = 0; i < v.size(); ++i)
            v(i) = 2;
    }, skepu::write(v));

    check_values(v, 2);
    copy(r, v);

    // read
    skepu::external(skepu::read(r), [&]{
        check_values(r, 2);
    });

    addOne(v, v);

    // read & write
    skepu::external(skepu::read(v), [&]{
        check_values(v, 3);
        for (size_t i = 0; i < r.size(); ++i)
            r(i) = v(i);
    }, skepu::write(r));

    check_values(r, 3);
}

