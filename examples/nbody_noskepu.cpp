#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <vector>

// Particle data structure that is used as an element type.
struct Particle
{
	float x, y, z;
	float vx, vy, vz;
	float m;
};


#define G 1
#define DELTA_T 0.1


// Computes a single Nbody iteration
void nbody_simulate_step(std::vector<Particle> &parr_out, std::vector<Particle> &parr_in)
{
	size_t np = parr_in.size();
	for (size_t i = 0; i < np; ++i)
	{
		Particle &pi = parr_in[i];
		float ax = 0.0, ay = 0.0, az = 0.0;

		for (size_t j = 0; j < np; ++j)
		{
			if (i != j)
			{
				Particle &pj = parr_in[j];
				std::cout << "---" << i << " " << j << "------\n";
				std::cout << 0 << " " << pi.x << " " << pi.y << " " << pi.z << "\n";
				std::cout << 0 << " " << pj.x << " " << pj.y << " " << pj.z << "\n";

				float rij = sqrt((pi.x - pj.x) * (pi.x - pj.x)
				               + (pi.y - pj.y) * (pi.y - pj.y)
				               + (pi.z - pj.z) * (pi.z - pj.z));

				float dum = G * pi.m * pj.m / pow(rij, 3);
				std::cout << dum << " " << ax << " " << ay << " " << az << "\n";

				ax += dum * (pi.x - pj.x);
				ay += dum * (pi.y - pj.y);
				az += dum * (pi.z - pj.z);
			}
		}

		Particle &newp = parr_out[i];
		newp.m = pi.m;
		
		std::cout << ax << " " << ay << " " << az << "\n";

		newp.x = pi.x + DELTA_T * pi.vx + DELTA_T * DELTA_T / 2 * ax;
		newp.y = pi.y + DELTA_T * pi.vy + DELTA_T * DELTA_T / 2 * ay;
		newp.z = pi.z + DELTA_T * pi.vz + DELTA_T * DELTA_T / 2 * az;

		newp.vx = pi.vx + DELTA_T * ax;
		newp.vy = pi.vy + DELTA_T * ay;
		newp.vz = pi.vz + DELTA_T * az;
	}
}


// Initializes particle array
void nbody_init(std::vector<Particle> &parr)
{
	size_t np = std::cbrt(parr.size());
	for (size_t s = 0; s < parr.size(); ++s)
	{
		Particle &p = parr[s];
		int d = np / 2 + 1;
		int i = s % np;
		int j = ((s - i) / np) % np;
		int k = (((s - i) / np) - j) / np;

		p.x = i - d + 1;
		p.y = j - d + 1;
		p.z = k - d + 1;

		p.vx = 0.0;
		p.vy = 0.0;
		p.vz = 0.0;

		p.m = 1;
		parr[s] = p;
	}
}


// A helper function to write particle output values to standard output stream.
void save_step(std::vector<Particle> &particles, std::ostream &os = std::cout)
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
void save_step(std::vector<Particle> &particles, const std::string &filename)
{
	std::ofstream out(filename);

	if (out.is_open())
		save_step(particles, out);
	else
		std::cerr << "Error: cannot open this file: " << filename << "\n";
}


void nbody(std::vector<Particle> &particles, size_t iterations)
{
	std::vector<Particle> doublebuffer(particles.size());

	// Particle vector initialization
	nbody_init(particles);

	// Iterative computation loop
	for (size_t i = 0; i < iterations; i += 2)
	{
		nbody_simulate_step(doublebuffer, particles);
		nbody_simulate_step(particles, doublebuffer);
	}
}


int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0]
				<< " particles iterations\n";
		exit(1);
	}

	const size_t np = std::stoul(argv[1]);
	const size_t iterations = std::stoul(argv[2]);

	// Particle vector
	std::vector<Particle> particles(np);

	nbody(particles, iterations);
	
	save_step(particles, "outputSERIAL.txt");
	return 0;
}
