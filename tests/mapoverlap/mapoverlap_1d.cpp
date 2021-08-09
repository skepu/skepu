#include <catch2/catch.hpp>

#include <skepu>
#include <skepu-lib/io.hpp>

float rotate_f(skepu::Region1D<int> r, int dir)
{
	return r(dir);
}


auto rotate = skepu::MapOverlap(rotate_f);

void helper_vector_rotate(skepu::Vector<int> &v, int dir)
{
	const size_t size{v.size()};
	const int defaultval = 9268;
	const	int pad = 4561;
	skepu::Vector<float> rv_a(size, defaultval), rv_b(size), rv_c(size), rv_d(size);
	
	const size_t overlap = std::abs(dir);
	rotate.setOverlap(overlap);
	
	rotate.setEdgeMode(skepu::Edge::None);
	rotate(rv_a, v, dir);
	skepu::io::cout << "none:    rv_a = " << rv_a << "\n";
	skepu::external(skepu::read(rv_a), [&]
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (i < overlap || i > size-overlap-1);//  CHECK(rv_a(i) == defaultval); // TODO: Maybe assert this?
			else                                    CHECK(rv_a(i) == (i+1 + dir));
		}
	});
	
	
	rotate.setEdgeMode(skepu::Edge::Cyclic);
	rotate(rv_b, v, dir);
	skepu::io::cout << "cyclic:    rv_b = " << rv_b << "\n";
	skepu::external(skepu::read(rv_b), [&]
	{
		for (size_t i = 0; i < size; ++i)
		{
			CHECK(rv_b(i) == (i + dir + size) % size + 1);
		}
	});

	rotate.setEdgeMode(skepu::Edge::Duplicate);
	rotate(rv_c, v, dir);
	skepu::io::cout << "duplicate:    rv_c = " << rv_c << "\n";
	skepu::external(skepu::read(rv_c), [&]
	{
		for (size_t i = 0; i < size; ++i)
		{
			if      (dir < 0 && i < overlap)        CHECK(rv_c(i) == 1);
			else if (dir > 0 && i > size-overlap-1) CHECK(rv_c(i) == (size));
			else                                    CHECK(rv_c(i) == (i+1 + dir));
		}
	});

	rotate.setEdgeMode(skepu::Edge::Pad);
	rotate.setPad(pad);
	rotate(rv_d, v, dir);
	skepu::io::cout << "pad:    rv_d = " << rv_d << "\n";
	skepu::external(skepu::read(rv_d), [&]
	{
		for (size_t i = 0; i < size; ++i)
		{
			if      (dir < 0 && i < overlap)        CHECK(rv_d(i) == pad);
			else if (dir > 0 && i > size-overlap-1) CHECK(rv_d(i) == pad);
			else                                    CHECK(rv_d(i) == (i+1 + dir));
		}
	});
}


TEST_CASE("MapOverlap 1D vector fundamentals")
{
	const size_t size{20};

	skepu::Vector<int> v(size);
	skepu::external([&]
	{
		for (size_t i = 0; i < size; ++i)
			v(i) = i+1;
		std::cout << "v: " << v <<"\n";
	}, skepu::write(v));
	
	
//	SECTION( "Rotate:  0" ) { helper_vector_rotate(v,  0); } // Invalid for OpenCL backend
	SECTION( "Rotate: -1" ) { helper_vector_rotate(v, -1); }
	SECTION( "Rotate: +1" ) { helper_vector_rotate(v, +1); }
	SECTION( "Rotate: -2" ) { helper_vector_rotate(v, -2); }
	SECTION( "Rotate: +2" ) { helper_vector_rotate(v, +2); }
}



void helper_matrix_rotate(skepu::Matrix<int> &m, int dir)
{
	const size_t rows{m.size_i()};
	const size_t cols{m.size_j()};
	const int defaultval = 9268;
	const	int pad = 4561;
	
	skepu::Matrix<float>
		rm_a(m.size_i(), m.size_j(), defaultval),
		rm_b(m.size_i(), m.size_j(), defaultval),
		rm_c(m.size_i(), m.size_j(), defaultval),
		rm_d(m.size_i(), m.size_j(), defaultval),
		rm_e(m.size_i(), m.size_j(), defaultval),
		rm_f(m.size_i(), m.size_j(), defaultval),
		rm_g(m.size_i(), m.size_j(), defaultval),
		rm_h(m.size_i(), m.size_j(), defaultval);

	
	const size_t overlap = std::abs(dir);
	rotate.setOverlap(overlap);
	
	rotate.setOverlapMode(skepu::Overlap::RowWise);
	
	rotate.setEdgeMode(skepu::Edge::None);
	rotate(rm_a, m, dir);
	skepu::io::cout << "Matrix Row-wise None:    rm_a = " << rm_a << "\n";
	skepu::external(skepu::read(rm_a, m), [&]
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < cols; ++j)
			{
				if (j < overlap || j > cols-overlap-1);//  CHECK(rm_a(i, j) == defaultval); // TODO: Maybe assert this?
				else                                    CHECK(rm_a(i, j) == m(i, j+dir));
			}
	});
	
	rotate.setEdgeMode(skepu::Edge::Cyclic);
	rotate(rm_b, m, dir);
	skepu::io::cout << "Matrix Row-wise Cyclic:    rm_b = " << rm_b << "\n";

	rotate.setEdgeMode(skepu::Edge::Duplicate);
	rotate(rm_c, m, dir);
	skepu::io::cout << "Matrix Row-wise Duplicate  rm_c = " << rm_c << "\n";

	rotate.setEdgeMode(skepu::Edge::Pad);
	rotate.setPad(pad);
	rotate(rm_d, m, dir);
	skepu::io::cout << "Matrix Row-wise Pad:     rm_d = " << rm_d << "\n";


	rotate.setOverlapMode(skepu::Overlap::ColWise);
	
	rotate.setEdgeMode(skepu::Edge::None);
	rotate(rm_e, m, dir);
	skepu::io::cout << "Matrix Col-wise None:    rm_e = " << rm_e << "\n";
	rotate.setEdgeMode(skepu::Edge::Cyclic);
	rotate(rm_f, m, dir);
	skepu::io::cout << "Matrix Col-wise Cyclic:    rm_f = " << rm_f << "\n";

	rotate.setEdgeMode(skepu::Edge::Duplicate);
	rotate(rm_g, m, dir);
	skepu::io::cout << "Matrix Col-wise Duplicate  rm_g = " << rm_g << "\n";

	rotate.setEdgeMode(skepu::Edge::Pad);
	rotate.setPad(pad);
	rotate(rm_h, m, dir);
	skepu::io::cout << "Matrix Col-wise Pad 0:     rm_h = " << rm_h << "\n";
	
}


TEST_CASE("MapOverlap 1D matrix fundamentals")
{
	const size_t size{16};
	
	skepu::Matrix<int> m(size, size);
	skepu::external([&]
	{
		int i = 1;
		for(size_t y = 0; y < size; ++y)
			for(size_t x = 0; x < size; ++x)
				m(y, x) = i++;
		std::cout << "m: " << m <<"\n";
	},
	skepu::write(m));
	
//	SECTION( "Rotate:  0" ) { helper_matrix_rotate(m,  0); } // Invalid for OpenCL backend
	SECTION( "Rotate: -1" ) { helper_matrix_rotate(m, -1); }
	SECTION( "Rotate: +1" ) { helper_matrix_rotate(m, +1); }
	SECTION( "Rotate: -2" ) { helper_matrix_rotate(m, -2); }
	SECTION( "Rotate: +2" ) { helper_matrix_rotate(m, +2); }
	
}

