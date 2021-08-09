#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include <skepu>
#include <skepu-lib/io.hpp>

// Particle data structure that is used as an element type.
struct Particle
{
	float x, y, z;
	float vx, vy, vz;
	float m;
};


constexpr float G [[skepu::userconstant]] = 1;
constexpr float DELTA_T [[skepu::userconstant]] = 0.1;

struct Acceleration
{
	float x, y, z;
};

// User function computing the acceleration influence on
// a particle p from another particle pj
Acceleration influence(skepu::Index2D index, Particle pi, Particle pj)
{
	Acceleration acc;

	if (index.row != index.col)
	{
		float rij = sqrt((pi.x - pj.x) * (pi.x - pj.x) + (pi.y - pj.y) * (pi.y - pj.y) + (pi.z - pj.z) * (pi.z - pj.z));
		float dum = G * pi.m * pj.m / pow(rij, 3);

		acc.x = dum * (pi.x - pj.x);
		acc.y = dum * (pi.y - pj.y);
		acc.z = dum * (pi.z - pj.z);
	}
	else
	{
		acc.x = 0;
		acc.y = 0;
		acc.z = 0;
	}
	return acc;
}

// User function computing the sum of two sets of acceleration influences
Acceleration sum(Acceleration lhs, Acceleration rhs)
{
	Acceleration res = lhs;
	res.x += rhs.x;
	res.y += rhs.y;
	res.z += rhs.z;
	return res;
}

// User function updating a particle p with acceleration influence a
Particle update(Particle p, Acceleration a)
{
	Particle res = p;

	res.x += DELTA_T * p.vx + DELTA_T * DELTA_T / 2 * a.x;
	res.y += DELTA_T * p.vy + DELTA_T * DELTA_T / 2 * a.y;
	res.z += DELTA_T * p.vz + DELTA_T * DELTA_T / 2 * a.z;

	res.vx += DELTA_T * a.x;
	res.vy += DELTA_T * a.y;
	res.vz += DELTA_T * a.z;

	return res;
}


// User-function that is used for initializing particles array
Particle init(skepu::Index1D index, size_t np)
{
	int s = index.i;
	int d = np / 2 + 1;
	int i = s % np;
	int j = ((s - i) / np) % np;
	int k = (((s - i) / np) - j) / np;

	Particle p;

	p.x = i - d + 1;
	p.y = j - d + 1;
	p.z = k - d + 1;

	p.vx = 0.0;
	p.vy = 0.0;
	p.vz = 0.0;

	p.m = 1;

	return p;
}



// A helper function to write particle output values to standard output stream.
void save_step(skepu::Vector<Particle> &particles, std::ostream &os = std::cout)
{
	int i = 0;

	os
		<< std::setw(4) << "#" << "  "
		<< std::setw(15) << "x"
		<< std::setw(15) << "y"
		<< std::setw(15) << "z"
		<< std::setw(15) << "vx"
		<< std::setw(15) << "vy"
		<< std::setw(15) << "vz" << "\n"
		<< std::string(96,'=') << "\n";
	for (Particle &p : particles)
	{
		os << std::setw( 4) << i++ << ": "
			<< std::setw(15) << p.x
			<< std::setw(15) << p.y
			<< std::setw(15) << p.z
			<< std::setw(15) << p.vx
			<< std::setw(15) << p.vy
			<< std::setw(15) << p.vz << "\n";
	}
}

//! A helper function to write particle output values to a file.
void save_step(skepu::Vector<Particle> &particles, const std::string &filename)
{
	std::ofstream out(filename);

	if (out.is_open())
		save_step(particles, out);
	else
		std::cerr << "Error: cannot open this file: " << filename << "\n";
}


void nbody(skepu::Vector<Particle> &particles, size_t iterations)
{
	// Skeleton instances
	auto nbody_init = skepu::Map<0>(init);
	auto nbody_influence = skepu::MapPairsReduce<1, 1>(influence, sum);
	auto nbody_update = skepu::Map<2>(update);

	// Intermediate data
	size_t np = particles.size();
	skepu::Vector<Acceleration> accel(np);

	// Particle vector initialization
	size_t cbrt_np = std::cbrt(np);
	nbody_init(particles, cbrt_np);

	// Iterative computation loop
	for (size_t i = 0; i < iterations; ++i)
	{
		nbody_influence(accel, particles, particles);
		nbody_update(particles, particles, accel);
	}
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		skepu::io::cout << "Usage: " << argv[0] << " particles iterations backend\n";
		exit(1);
	}

	// Handle arguments
	const size_t np = std::stoul(argv[1]);
	const size_t iterations = std::stoul(argv[2]);
	auto spec = skepu::BackendSpec{argv[3]};
	skepu::setGlobalBackendSpec(spec);

	// Particle vector
	skepu::Vector<Particle> particles(np);

	nbody(particles, iterations);
	
	// Write out result
	skepu::external(skepu::read(particles), [&]{
		std::stringstream outfile;
		outfile << "output" << spec.type() << ".txt";
		save_step(particles, outfile.str());
	});

	return 0;
}
