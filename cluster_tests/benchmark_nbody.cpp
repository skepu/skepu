#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "common.hpp"

#ifdef BENCHMARK_NBODY


namespace benchmark_nbody {

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
Particle move(skepu2::Index1D index,
              Particle pi,
              const skepu2::Vec<Particle> parr)
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
Particle init(skepu2::Index1D index, size_t np)
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

auto nbody_init = skepu2::Map<0>(init);
auto nbody_simulate_step = skepu2::Map<1>(move);

void nbody(skepu2::Vector<Particle> &particles, size_t iterations)
{
	size_t np = particles.size();
	skepu2::Vector<Particle> doublebuffer(particles.size());

	// particle vectors initialization
	nbody_init(particles, np);

	for (size_t i = 0; i < iterations; i += 2)
	{
		nbody_simulate_step(doublebuffer, particles, particles);
		nbody_simulate_step(particles, doublebuffer, doublebuffer);
	}
}

		TEST_CASE("Benchmark nbody")
{
	std::vector<size_t> nps {1,2,4,8};//,16,32,64};
			const size_t iterations = 8;//std::stoul(argv[2]);
			for(auto np : nps) {
					SOFFA_BENCHMARK("nbody.csv", {"nodes", "N"}, \
													{ std::to_string(skepu2::cluster::mpi_size()), \
														std::to_string(np*512)}, "nbody");
				skepu2::Vector<Particle> particles(np*512);
				nbody(particles, iterations);
			}
}
}

#endif
