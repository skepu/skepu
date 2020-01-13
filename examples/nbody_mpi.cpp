#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <skepu>

// Particle data structure that is used as an element type.
struct Particle
{
	float x, y, z;
	float vx, vy, vz;
	float m;
};


constexpr float G = 1;
constexpr float delta_t = 0.1;

/*
 * Array user-function that is used for applying nbody computation,
 * All elements from parr and a single element (named 'pi') are accessible
 * to produce one output element of the same type.
 */
Particle move(skepu::Index1D index,
              Particle pi,
              const skepu::Vec<Particle> parr)
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

	pi.x += delta_t * pi.vx + delta_t * delta_t / 2 * ax;
	pi.y += delta_t * pi.vy + delta_t * delta_t / 2 * ay;
	pi.z += delta_t * pi.vz + delta_t * delta_t / 2 * az;

	pi.vx += delta_t * ax;
	pi.vy += delta_t * ay;
	pi.vz += delta_t * az;

	return pi;
}


// Generate user-function that is used for initializing particles array.
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

auto inline
operator<<(std::ostream & os, Particle const & p)
-> std::ostream &
{
	os
		<< std::setw(15) << p.x
		<< std::setw(15) << p.y
		<< std::setw(15) << p.z
		<< std::setw(15) << p.vx
		<< std::setw(15) << p.vy
		<< std::setw(15) << p.vz;

	return os;
}

auto inline
operator<<(std::fstream & os, skepu::Vector<Particle> & ps)
-> std::ostream &
{
	os
		<< std::setw(4) << "#" << "  "
		<< std::setw(15) << "x"
		<< std::setw(15) << "y"
		<< std::setw(15) << "z"
		<< std::setw(15) << "vx"
		<< std::setw(15) << "vy"
		<< std::setw(15) << "vz" << "\n"
		<< std::string(96,'=') << "\n";
	os << std::flush;
	for(size_t i{0}; i < ps.size(); ++i)
		os << std::setw(4) << i << ": " << ps[i] << "\n";

	return os;
}

auto nbody_init = skepu::Map<0>(init);
auto nbody_simulate_step = skepu::Map<1>(move);

void nbody(skepu::Vector<Particle> &particles, size_t iterations)
{
	skepu::Vector<Particle> doublebuffer(particles.size());

	for (size_t i = 0; i < iterations; i += 2)
	{
		nbody_simulate_step(doublebuffer, particles, particles);
		nbody_simulate_step(particles, doublebuffer, doublebuffer);
	}
}

int main(int argc, char** argv)
{
	if(argc != 3 && argc != 4)
	{
		std::cout << "Usage: " << argv[0]
			<< " <particles> <interations> [filename]\n";
		return 0;
	}

	auto rank = skepu::cluster::mpi_rank();
	size_t np = std::stoul(argv[1]);
	size_t iterations = std::stoul(argv[2]);
	skepu::Vector<Particle> particles(np);

	skepu::cluster::barrier();
	if(!rank)
	{
		std::cout
			<< "nbody simulation (StarPU MPI)\n"
			<< "Particles: " << np << "\n"
			<< "Iterations: " << iterations << "\n"
			<< "MPI Ranks: " << skepu::cluster::mpi_size() << "\n"
			<< "StarPU threads: " << skepu::cluster::starpu_ncpus() << "\n"
			<< std::endl;
	}

	nbody_init(particles, np);

	skepu::cluster::barrier();
	auto start = std::chrono::high_resolution_clock::now();
	nbody(particles, iterations);
	skepu::cluster::barrier();
	auto stop = std::chrono::high_resolution_clock::now();

	auto time = stop - start;
	auto hours =
		std::chrono::duration_cast<std::chrono::hours>(time).count();
	auto minutes =
		std::chrono::duration_cast<std::chrono::minutes>(time).count() % 60;
	auto seconds =
		std::chrono::duration_cast<std::chrono::seconds>(time).count() % 60;
	auto milliseconds =
		std::chrono::duration_cast<std::chrono::milliseconds>(time).count() % 1000;

	if(!rank)
		std::cout << "Execution time: "
			<< hours
			<< ":" << minutes
			<< ":" << seconds
			<< "." << milliseconds
			<< std::endl;

	if(argc == 4)
	{
		particles.flush();
		if(argc == 4 && !rank)
		{
			std::string filename{argv[3]};
			std::fstream file(filename, std::ios_base::out);
			assert(file.is_open());
			file << particles;
			file.close();
		}
	}

	return 0;
}
