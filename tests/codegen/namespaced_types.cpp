#include "../../external/catch2/catch.hpp"

#include <skepu>
#include <test_helper_automatic_setup.hpp>

// User types in namespaces, as parameters and results of user functions and as elements of
// containers. In OpenCL C the qualified C++ name (geometry::Point) is not an identifier: the generated
// kernel code must refer to the types by names that are defined there, whichever way the user
// function spells the type (qualified, unqualified in its own namespace, or through a typedef).

namespace geometry
{
	struct Point
	{
		float x, y;
	};

	namespace detail
	{
		struct Tagged
		{
			int id;
			float weight;
		};
	}

	using Real = float;

	// unqualified type names, as the standard library of SkePU (skepu::filter) spells them
	Point add_points(Point a, Point b)
	{
		Point r;
		r.x = a.x + b.x;
		r.y = a.y + b.y;
		return r;
	}

	Point scale_point(Point p, Point factor)
	{
		Point r;
		r.x = p.x * factor.x;
		r.y = p.y * factor.y;
		return r;
	}

	float squared_norm(Point p)
	{
		return p.x * p.x + p.y * p.y;
	}

	float sum_floats(float a, float b)
	{
		return a + b;
	}

	detail::Tagged tag_point(Point p, int id)
	{
		detail::Tagged t;
		t.id = id;
		t.weight = p.x + p.y;
		return t;
	}

	skepu::multiple<Point, detail::Tagged> split_tagged(detail::Tagged t)
	{
		Point p;
		p.x = t.weight;
		p.y = -t.weight;
		return skepu::ret(p, t);
	}

	Point stencil_sum(skepu::Region1D<Point> r)
	{
		Point s;
		s.x = r(-1).x + r(0).x + r(1).x;
		s.y = r(-1).y + r(0).y + r(1).y;
		return s;
	}

	float stencil_real(skepu::Region1D<Real> r)
	{
		return r(-1) + r(0) + r(1);
	}

	template<typename T>
	T mirrored(T p)
	{
		T r;
		r.x = -p.x;
		r.y = -p.y;
		return r;
	}

	float lookup_x(skepu::Index1D i, skepu::Vec<Point> points)
	{
		return points[points.size - 1 - i.i].x;
	}
}

// qualified type names
geometry::Point qualified_difference(geometry::Point a, geometry::Point b)
{
	geometry::Point r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	return r;
}

auto add_map = skepu::Map<2>(geometry::add_points);
auto scale_map = skepu::Map<1>(geometry::scale_point);
auto difference_map = skepu::Map<2>(qualified_difference);
auto tag_map = skepu::Map<1>(geometry::tag_point);
auto split_map = skepu::Map<1>(geometry::split_tagged);
auto norm_sum = skepu::MapReduce<1>(geometry::squared_norm, geometry::sum_floats);
auto point_sum = skepu::Reduce(geometry::add_points);
auto stencil = skepu::MapOverlap(geometry::stencil_sum);
auto stencil_r = skepu::MapOverlap(geometry::stencil_real);
auto lookup = skepu::Map<0>(geometry::lookup_x);
auto mirror = skepu::Map<1>(geometry::mirrored<geometry::Point>);

static skepu::Vector<geometry::Point> points(size_t size)
{
	skepu::Vector<geometry::Point> v(size);
	for (size_t i = 0; i < size; ++i)
	{
		v(i).x = float(i);
		v(i).y = 2.f * float(i);
	}
	return v;
}

TEST_CASE("Map with a namespaced type as element type and uniform")
{
	const size_t size{16};
	auto a = points(size), b = points(size);
	skepu::Vector<geometry::Point> sum(size), diff(size), scaled(size);

	add_map(sum, a, b);
	difference_map(diff, sum, a);

	geometry::Point factor;
	factor.x = 3.f;
	factor.y = 0.5f;
	scale_map(scaled, a, factor);

	sum.flush(); diff.flush(); scaled.flush();
	for (size_t i = 0; i < size; ++i)
	{
		CHECK(sum(i).x == 2.f * i);
		CHECK(sum(i).y == 4.f * i);
		CHECK(diff(i).x == float(i));
		CHECK(diff(i).y == 2.f * i);
		CHECK(scaled(i).x == 3.f * i);
		CHECK(scaled(i).y == 1.f * i);
	}
}

TEST_CASE("Map from and to namespaced types in different namespaces")
{
	const size_t size{8};
	auto a = points(size);
	skepu::Vector<geometry::detail::Tagged> tagged(size);

	tag_map(tagged, a, 7);

	tagged.flush();
	for (size_t i = 0; i < size; ++i)
	{
		CHECK(tagged(i).id == 7);
		CHECK(tagged(i).weight == 3.f * i);
	}
}

TEST_CASE("Map with several namespaced result types")
{
	const size_t size{8};
	auto a = points(size);
	skepu::Vector<geometry::detail::Tagged> tagged(size), tagged_back(size);
	skepu::Vector<geometry::Point> split(size);

	tag_map(tagged, a, 2);
	split_map(split, tagged_back, tagged);

	split.flush(); tagged_back.flush();
	for (size_t i = 0; i < size; ++i)
	{
		CHECK(split(i).x == 3.f * i);
		CHECK(split(i).y == -3.f * i);
		CHECK(tagged_back(i).id == 2);
		CHECK(tagged_back(i).weight == 3.f * i);
	}
}

// (Scan of a struct type does not work in OpenCL, with or without namespaces: its kernel has a literal 0 in
// the place of the identity)
TEST_CASE("MapReduce and Reduce of a namespaced type")
{
	const size_t size{10};
	auto a = points(size);

	float expected_norm = 0.f;
	for (size_t i = 0; i < size; ++i)
		expected_norm += float(i) * float(i) * 5.f;
	const float norm = norm_sum(a);
	CHECK(norm == expected_norm);

	geometry::Point total = point_sum(a);
	CHECK(total.x == 45.f);
	CHECK(total.y == 90.f);
}

TEST_CASE("MapOverlap with regions of a namespaced type and of a typedef")
{
	const size_t size{12};
	auto a = points(size);
	skepu::Vector<geometry::Point> out(size);
	skepu::Vector<float> reals(size, 1.f), real_out(size);

	stencil.setOverlap(1);
	stencil.setEdgeMode(skepu::Edge::Pad);
	geometry::Point zero;
	zero.x = 0.f;
	zero.y = 0.f;
	stencil.setPad(zero);
	stencil(out, a);

	stencil_r.setOverlap(1);
	stencil_r.setEdgeMode(skepu::Edge::Pad);
	stencil_r.setPad(0.f);
	stencil_r(real_out, reals);

	out.flush(); real_out.flush();
	for (size_t i = 0; i < size; ++i)
	{
		const bool edge = (i == 0 || i == size - 1);
		const float expected_x = (i == 0) ? 1.f : (i == size - 1) ? 2.f * i - 1.f : 3.f * i;
		CHECK(out(i).x == expected_x);
		CHECK(real_out(i) == (edge ? 2.f : 3.f));
	}
}

TEST_CASE("Random access to a container of a namespaced type")
{
	const size_t size{9};
	auto a = points(size);
	skepu::Vector<float> out(size);

	lookup(out, a);

	out.flush();
	for (size_t i = 0; i < size; ++i)
		CHECK(out(i) == float(size - 1 - i));
}

TEST_CASE("Map with a user function template instantiated with a namespaced type")
{
	const size_t size{8};
	auto a = points(size);
	skepu::Vector<geometry::Point> out(size);

	mirror(out, a);

	out.flush();
	for (size_t i = 0; i < size; ++i)
	{
		CHECK(out(i).x == -float(i));
		CHECK(out(i).y == -2.f * i);
	}
}
