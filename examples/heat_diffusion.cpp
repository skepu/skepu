#include <skepu>
#include <skepu-lib/io.hpp>

#define ENABLE_1D_EXAMPLE 1
#define ENABLE_2D_EXAMPLE 1
#define ENABLE_3D_EXAMPLE 1
#define ENABLE_4D_EXAMPLE 1
#define ENABLE_DEBUG 1

float heat1D(skepu::Region1D<float> r)
{
	float newval = 0;
	newval += r(-1);
	newval += r( 1);
	newval /= 2;
	return newval;
}

float heat2D(skepu::Region2D<float> r)
{
	float newval = 0;
	newval += r(-1,  0);
	newval += r( 1,  0);
	newval += r( 0, -1);
	newval += r( 0,  1);
	newval /= 4;
	return newval;
}

float heat3D(skepu::Region3D<float> r)
{
	float newval = 0;
	newval += r(-1,  0,  0);
	newval += r( 1,  0,  0);
	newval += r( 0, -1,  0);
	newval += r( 0,  1,  0);
	newval += r( 0,  0, -1);
	newval += r( 0,  0,  1);
	newval /= 6;
	return newval;
}

float heat4D(skepu::Region4D<float> r)
{
	float newval = 0;
	newval += r(-1,  0,  0,  0);
	newval += r( 1,  0,  0,  0);
	newval += r(0,  -1,  0,  0);
	newval += r(0,   1,  0,  0);
	newval += r(0,   0, -1,  0);
	newval += r(0,   0,  1,  0);
	newval += r(0,   0,  0, -1);
	newval += r(0,   0,  0,  1);
	newval /= 8;
	return newval;
}

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		skepu::io::cout << "Usage: " << argv[0] << " dim size iterations backend\n";
		exit(1);
	}
	
	const float dim = atof(argv[1]);
	const float size = atof(argv[2]);
	const float iters = atof(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);

#if ENABLE_1D_EXAMPLE
	if (dim == 1)
	{
		auto update = skepu::MapOverlap(heat1D);
		update.setOverlap(1);
		update.setEdgeMode(skepu::Edge::None);
		update.setUpdateMode(skepu::UpdateMode::RedBlack);
		skepu::Vector<float> domain(size, 0);
		
		domain(0) = 0;
		domain(size-1) = 5;
		
		for (size_t i = 0; i < iters; ++i)
		{
			update(domain, domain);
		}
		
		skepu::io::cout << domain << "\n";
		exit(0);
	}
#endif
	
#if ENABLE_2D_EXAMPLE
	if (dim == 2)
	{
		auto update = skepu::MapOverlap(heat2D);
		update.setOverlap(1, 1);
		update.setEdgeMode(skepu::Edge::None);
		update.setUpdateMode(skepu::UpdateMode::RedBlack);
		skepu::Matrix<float> domain(size, size, 0);
		
		for (size_t i = 0; i < size; ++i)
		{
			domain(i, 0) = 2;
			domain(i, size-1) = 2;
		}
		
		for (size_t i = 0; i < size; ++i)
		{
			domain(0, i) = 0;
			domain(size-1, i) = 5;
		}
		
		for (size_t i = 0; i < iters; ++i)
		{
			update(domain, domain);
		}
		
		skepu::io::cout << domain << "\n";
		exit(0);
	}
#endif
	
#if ENABLE_3D_EXAMPLE
	if (dim == 3)
	{
		auto update = skepu::MapOverlap(heat3D);
		update.setOverlap(1, 1, 1);
		update.setEdgeMode(skepu::Edge::None);
		update.setUpdateMode(skepu::UpdateMode::RedBlack);
		skepu::Tensor3<float> domain(size, size, size, 0);
		
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t j = 0; j < size; ++j)
			{
				domain(0, i, j) = 1;
				domain(size-1, i, j) = 5;
				domain(i, 0, j) = 2;
				domain(i, size-1, j) = 2;
			}
		}
		
		for (size_t i = 0; i < iters; ++i)
		{
			update(domain, domain);
		}
		
		skepu::io::cout << domain << "\n";
		exit(0);
	}
#endif

#if ENABLE_4D_EXAMPLE
	if (dim == 4)
	{
		auto update = skepu::MapOverlap(heat4D);
		update.setOverlap(1, 1, 1, 1);
		update.setEdgeMode(skepu::Edge::None);
		update.setUpdateMode(skepu::UpdateMode::RedBlack);
		skepu::Tensor4<float> domain(size, size, size, size, 0);
		
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t j = 0; j < size; ++j)
			{
				for (size_t k = 0; k < size; ++k)
				{
					domain(0, i, j, k) = 1;
					domain(size-1, i, j, k) = 5;
					domain(i, 0, j, k) = 2;
					domain(i, size-1, j, k) = 2;
				}
			}
		}
		
		for (size_t i = 0; i < iters; ++i)
		{
			update(domain, domain);
		}
		
		skepu::io::cout << domain << "\n";
		exit(0);
	}
#endif
	
	return 0;
}

