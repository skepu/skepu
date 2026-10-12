#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

// Skeleton instances that are local variables and are passed by reference to function templates (a deduced type or a template
// parameter), which call them. The precompiler must replace the type of the instance, and the calls inside the templates must
// work with every backend.

int twice_plus(int a, int b) { return 2 * a + b; }
int plus(int a, int b) { return a + b; }
int pick_largest(int a, int b) { return a > b ? a : b; }
int square(int a) { return a * a; }

skepu::multiple<int, int> neighbours(skepu::Region1D<int> r)
{
	return skepu::ret(r(-1) + r(1), r(-1) - r(1));
}

template<typename Skeleton, typename... Args>
void call_with(Skeleton &skeleton, Args&&... args)
{
	skeleton(std::forward<Args>(args)...);
}

template<typename Skeleton, typename Container>
int reduce_with(Skeleton &skeleton, Container &c)
{
	return (int)skeleton(c);
}

template<typename Skeleton>
struct Wrapper
{
	explicit Wrapper(Skeleton &s): skeleton(s) {}
	template<typename... Args>
	void run(Args&&... args) { skeleton(std::forward<Args>(args)...); }
	Skeleton &skeleton;
};


TEST_CASE("Skeleton instances passed by reference to function templates")
{
	const size_t size{50};
	skepu::Vector<int> a(size), b(size), r(size, 0), r2(size, 0);
	for (size_t i = 0; i < size; ++i)
	{
		a(i) = (int)i;
		b(i) = 3;
	}

	auto map = skepu::Map<2>(twice_plus);
	call_with(map, r, a, b);
	r.flush();
	for (size_t i = 0; i < size; ++i)
		CHECK(r(i) == 2 * (int)i + 3);

	auto sum = skepu::Reduce(plus);
	CHECK(reduce_with(sum, a) == (int)(size * (size - 1) / 2));

	auto mapreduce = skepu::MapReduce<1>(square, pick_largest);
	CHECK(reduce_with(mapreduce, a) == (int)((size - 1) * (size - 1)));

	// Multiple return values, and the update mode, through a wrapper object
	auto overlap = skepu::MapOverlap(neighbours);
	overlap.setOverlap(1);
	overlap.setEdgeMode(skepu::Edge::None);
	Wrapper<decltype(overlap)> wrapper(overlap);
	wrapper.run(r, r2, a);
	r.flush();
	r2.flush();
	for (size_t i = 1; i + 1 < size; ++i)
	{
		CHECK(r(i) == (int)(2 * i));
		CHECK(r2(i) == -2);
	}
}


TEST_CASE("Skeleton instances passed by reference: reductions of matrices and tensors")
{
	skepu::Matrix<int> m(8, 9, 2);
	skepu::Tensor3<int> t(4, 5, 6, "", 3);

	auto sum = skepu::Reduce(plus);
	auto largest = skepu::Reduce(pick_largest);

	CHECK(reduce_with(sum, m) == 8 * 9 * 2);
	CHECK(reduce_with(largest, m) == 2);
	CHECK(reduce_with(sum, t) == 4 * 5 * 6 * 3);
	CHECK(reduce_with(largest, t) == 3);
}
