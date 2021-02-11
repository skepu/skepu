#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>

#include <skepu>

// Particle data structure that is used as an element type.
struct Particle
{
	float x, y, z;
	float vx, vy, vz;
	float m;
};


constexpr float G [[skepu::userconstant]] = 1;
constexpr float DELTA_T [[skepu::userconstant]] = 0.1;


/*
 * User-function for applying Nbody computation.
 * All elements from parr and a single element (named 'pi') are accessible
 * to produce one output element of the same type.
 */
Particle move(skepu::Index1D index, Particle pi, const skepu::Vec<Particle> parr)
{
	size_t i = index.i;

	float ax = 0.0, ay = 0.0, az = 0.0;
	size_t np = parr.size;

	for (size_t j = 0; j < np; ++j)
	{
		if (i != j)
		{
			Particle pj = parr[j];

			float rij = sqrt((pi.x - pj.x) * (pi.x - pj.x)
			               + (pi.y - pj.y) * (pi.y - pj.y)
			               + (pi.z - pj.z) * (pi.z - pj.z));

			float dum = G * pi.m * pj.m / pow(rij, 3);

			ax += dum * (pi.x - pj.x);
			ay += dum * (pi.y - pj.y);
			az += dum * (pi.z - pj.z);
		}
	}

	Particle newp;
	newp.m = pi.m;

	newp.x = pi.x + DELTA_T * pi.vx + DELTA_T * DELTA_T / 2 * ax;
	newp.y = pi.y + DELTA_T * pi.vy + DELTA_T * DELTA_T / 2 * ay;
	newp.z = pi.z + DELTA_T * pi.vz + DELTA_T * DELTA_T / 2 * az;

	newp.vx = pi.vx + DELTA_T * ax;
	newp.vy = pi.vy + DELTA_T * ay;
	newp.vz = pi.vz + DELTA_T * az;

	return newp;
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
		os << std::setw(4) << i++ << ": "
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
	auto nbody_simulate_step = skepu::Map(move);

	// Itermediate data
	size_t np = particles.size();
	skepu::Vector<Particle> doublebuffer(np);

	// Particle vector initialization
	size_t cbrt_np = std::cbrt(np);
	nbody_init(particles, cbrt_np);

	// Iterative computation loop
	for (size_t i = 0; i < iterations; i += 2)
	{
		nbody_simulate_step(doublebuffer, particles, particles);
		nbody_simulate_step(particles, doublebuffer, doublebuffer);
	}
}

int main(int argc, char *argv[])
{
	if (argc < 4)
	{
		skepu::external([&]{
			std::cout << "Usage: " << argv[0]
				<< " particles iterations backend\n";}
		);
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
	skepu::external(
		skepu::read(particles),
		[&]{
			std::stringstream outfile;
			outfile << "output" << spec.type() << ".txt";
			save_step(particles, outfile.str());
		}
	);
	return 0;
}
