#include <iostream>
#include <cmath>

#include <skepu>
#include <skepu-lib/io.hpp>

#define RAND_MAX_LOC 100

typedef struct _dim3
{
	size_t x;
	size_t y;
	size_t z;
} dimension3;

struct Atom
{
	float charge;
	float x, y, z;
};

std::ostream &operator<<(std::ostream &o, Atom &a)
{
	o << "Atom: [ x: " << a.x << " y: " << a.y << " z: " << a.z << " charge: " << a.charge << " ]\n";
	return o;
}


// ***********************
// ***********************
// Please note down the difference that here we use "sqrt" function instead of "sqrtf" function used in CUDA code
// ***********************
// ***********************

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
// Note: this implementation uses precomputed and unrolled
// loops of dy*dy + dz*dz values for increased FP arithmetic intensity
// per FP load.  The X coordinate portion of the loop is unrolled by
// four, allowing the same dy^2 + dz^2 values to be reused four times,
// increasing the ratio of FP arithmetic relative to FP loads, and
// eliminating some redundant calculations.
// This version implement's Kahan's compensated summation method to
// increase doubleing point accuracy for the large number of summed
// potential values.
//
//
float coulombPotential_f(skepu::Index2D index, float energygridItem, const skepu::Vec<Atom> atoms, float gridspacing)
{
	float coory = gridspacing * index.col;
	float coorx = gridspacing * index.row;
	
	float energyvalx1 = 0;
	float energycomp1 = 0;
	size_t numatoms = atoms.size;
	for (size_t atomid = 0; atomid < numatoms; atomid ++)
	{
		float dy = coory - atoms.data[atomid].y;
		float dx = coorx - atoms.data[atomid].x;
		float s = atoms.data[atomid].charge / sqrt(dx*dx + dy*dy);
		
		float y = s - energycomp1;
		float t = energyvalx1 + y;
		energycomp1 = (t - energyvalx1)  - y;
		energyvalx1 = t;
	}
	return energygridItem + energyvalx1;
}


// Random atom: a charge in [-1, 1) and a position in the unit cube
Atom random_atom(skepu::Random<4> &random)
{
	Atom a;
	a.charge = random.getNormalized() * 2.0 - 1.0;
	a.x = random.getNormalized();
	a.y = random.getNormalized();
	a.z = random.getNormalized();
	return a;
}

// Function to initialize atoms
void initatoms(skepu::Vector<Atom> &atombuf)
{
	auto init = skepu::Map<0>(random_atom);
	skepu::PRNG prng(0);
	init.setPRNG(prng);
	init(atombuf);
}


const float gridspacing = 0.1;
const size_t matrixSize = 10;

auto columbicPotential = skepu::Map<1>(coulombPotential_f);

void coulombic(skepu::Matrix<float> &energy_out, skepu::Vector<Atom> &atombuf, skepu::BackendSpec *spec = nullptr)
{
	size_t atomcount = atombuf.size();
	
	if (spec)
		columbicPotential.setBackend(*spec);
		
	skepu::Matrix<float> grid_in(matrixSize, matrixSize);
	
	columbicPotential(energy_out, grid_in, atombuf, gridspacing);
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		skepu::io::cout << "Usage: " << argv[0] << " input_size backend\n";
		exit(1);
	}
	
	const size_t atomcount = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec); // (also for the initialization: data that a GPU backend has written cannot be used directly by another one)
	
	skepu::Vector<Atom> atoms(atomcount);
	skepu::Matrix<float> energy_out(matrixSize, matrixSize);
	
	// allocate and initialize atom coordinates and charges
	initatoms(atoms);
	
	coulombic(energy_out, atoms, &spec);
	
	// can print and compare. output is exactly same for cpu, openmp, cuda for 1 and 2 gpus.
	skepu::io::cout << "Energy out: " << energy_out << "\n";
	
	return 0;
}
