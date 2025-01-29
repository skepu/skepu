
#include <skepu>
#include <skepu-lib/io.hpp>

// Outputs
skepu::multiple<
	int,
	float,
	int,
	float
>
complicated_user_function(
	// Index
	skepu::Index1D i1,
//	skepu::Index2D i2,
//	skepu::Index3D i3,
//	skepu::Index4D i4,

	// Random generator stream
	skepu::Random<1> &rand,

	// Element-wise inputs
	int e1,
	float e2,
	float e4,
	int e3,

	// Proxy container inputs
	skepu::Vec<float> p1,
	skepu::Mat<float> p2,
	skepu::Ten3<float> p3,
	skepu::Ten4<float> p4,

	// Uniform inputs / scalar
	int u1,
	float u2,
	int u3
)
{
	// Computations...
	float r = rand.getNormalized();

	// Return
	return skepu::ret(
		i1.i,
		r,
		1,
		1.0
	);
}


int main(int argc, char *argv[])
{
	// Command line inputs
	if (argc < 3)
	{
	  skepu::io::cout << "Usage: " << argv[0] << " size backend\n";
	  exit(1);
	}

	const float size = atof(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);

	// Containers
	skepu::Vector<int> vi1(size), vi2(size), vi3(size), vi4(size), vi5(size), vi6(size);
	skepu::Vector<float>  vf1(size), vf2(size), vf3(size), vf4(size), vf5(size), vf6(size);

	skepu::Matrix<float> m1(size, size), m2(size, size);
	skepu::Tensor3<float> t3d(size, size, size);
	skepu::Tensor4<float> t4d(size, size, size, size);

	// Skeleton instance
	auto complicated_skeleton_instance = skepu::Map(complicated_user_function);
	skepu::PRNG stream(0);
	complicated_skeleton_instance.setPRNG(stream);

	// Skeleton call
	complicated_skeleton_instance(
		// Element-wise outputs
		vi1,
		vf1,
		vi2,
		vf2,

		// Element-wise inputs
		vi3,
		vf3,
		vf4,
		vi4,

		// Random-access containers
		vf5,
		m1,
		t3d,
		t4d,

		// Uniforms / scalars
		15,
		3.14,
		1024

	);

	skepu::io::cout << "Results:\n";
	skepu::io::cout << vi1 << "\n";
	skepu::io::cout << m1 << "\n";

	return 0;
}
