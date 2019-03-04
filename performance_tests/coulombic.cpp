#include <iostream>
#include <cmath>

#include <skepu2.hpp>
#include "performance_tests_common.hpp"

const float gridspacing = 0.1;
const size_t matrixSize = 10;

const size_t N = 100000;
std::string application = "Coulombic";

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
float coulombPotential_f(skepu2::Index2D index, float energygridItem, const skepu2::Vec<Atom> atoms, float gridspacing)
{
	float coory = gridspacing * index.col;
	float coorx = gridspacing * index.row;
	
//	std::cout << "X: " << coorx << " Y: " << coory << "\n";
	
	float energyvalx1 = 0;
	float energycomp1 = 0;
	size_t numatoms = atoms.size;
	for (size_t atomid = 0; atomid < numatoms; atomid ++)
	{
		float dy = coory - atoms.data[atomid].y;
		float dx = coorx - atoms.data[atomid].x;
		float s = atoms.data[atomid].charge / sqrt(dx*dx + dy*dy);
		
	//	std::cout << "s: " << s << "\n";
		
		float y = s - energycomp1;
		float t = energyvalx1 + y;
		energycomp1 = (t - energyvalx1)  - y;
		energyvalx1 = t;
	}
	return energygridItem + energyvalx1;
}


// Function to initialize atoms
void initatoms(skepu2::Vector<Atom> &atombuf, dimension3 volsize, double gridspacing)
{
	srand(0);
	
	// compute grid dimensions in angstroms
	dimension3 size;
	size.x = gridspacing * volsize.x;
	size.y = gridspacing * volsize.y;
	size.z = gridspacing * volsize.z;
	
	size_t count = atombuf.size();
	for (size_t i = 0; i < count; i++)
	{
		Atom a;
		a.charge = ((rand() / (double) RAND_MAX) * 2.0) - 1.0;  // charge
		a.x = (rand() / (double) RAND_MAX);// * size.x;
		a.y = (rand() / (double) RAND_MAX);// * size.y;
		a.z = (rand() / (double) RAND_MAX);// * size.z;
		atombuf[i] = a;
	}
}



auto columbicPotential = skepu2::Map<1>(coulombPotential_f);

double coulombic() {
	skepu2::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		skepu2::Vector<Atom> atoms(N);
		dimension3 volsize {2048, 2048, 1};
		initatoms(atoms, volsize, gridspacing);
		
		skepu2::Matrix<float> grid_in(matrixSize, matrixSize);
		skepu2::Matrix<float> energy_out(matrixSize, matrixSize);
		
		timer.start();
		columbicPotential(energy_out, grid_in, atoms, gridspacing);
		timer.stop();
	}
	return timer.getMedianTime();
}



constexpr auto benchmarkFunc = coulombic;

void setBackend(const skepu2::BackendSpec& spec) {
	columbicPotential.setBackend(spec);
}

void tune() {
	skepu2::backend::tuner::hybridTune(columbicPotential, 16, 1, 32, 1024);
	columbicPotential.resetBackend();
}

int main(int argc, char* argv[]) {
	std::vector<double> times;
	
	std::cout << application << ": Running CPU backend" << std::endl;
	skepu2::BackendSpec specCPU(skepu2::Backend::Type::CPU);
	setBackend(specCPU);
	double cpuTime = benchmarkFunc();
	
	std::cout << application << ": Running OpenMP backend" << std::endl;
	skepu2::BackendSpec specOpenMP(skepu2::Backend::Type::OpenMP);
	specOpenMP.setCPUThreads(16);
	setBackend(specOpenMP);
	times.push_back(benchmarkFunc());
	
	std::cout << application << ": Running CUDA GPU backend" << std::endl;
	skepu2::BackendSpec specGPU(skepu2::Backend::Type::CUDA);
	specGPU.setDevices(1);
	setBackend(specGPU);
	times.push_back(benchmarkFunc());
	
	
	std::cout << application << ": Running Oracle" << std::endl;
	double bestOracleTime = 100000000.0;
	size_t bestOracleRatio = 9999999;
	std::vector<double> oracleTimes;
	
	for(size_t ratio = 0; ratio <= 100; ratio += 5) {
		double percentage = (double)ratio / 100.0;
		skepu2::BackendSpec spec(skepu2::Backend::Type::Hybrid);
		spec.setDevices(1);
		spec.setCPUThreads(16);
		spec.setCPUPartitionRatio(percentage);
		
		setBackend(spec);
		double time = benchmarkFunc();
		
		oracleTimes.push_back(time);
		if(time < bestOracleTime) {
			bestOracleTime = time;
			bestOracleRatio = ratio;
		}
		std::cout << "Ratio: " << percentage << " gave time: " << time << std::endl;
	}
	times.push_back(bestOracleTime);
	std::cout << "Optimal ratio was: " << bestOracleRatio << std::endl;
	
	std::cout << application << ": Running Hybrid backend" << std::endl;
	tune();
	times.push_back(benchmarkFunc());
	
	appendPerformanceResult("speedup_openmp.dat", application, cpuTime/times[0]);
	appendPerformanceResult("speedup_cuda.dat", application, cpuTime/times[1]);
	appendPerformanceResult("speedup_oracle.dat", application, cpuTime/times[2]);
	appendPerformanceResult("speedup_hybrid.dat", application, cpuTime/times[3]);
	
	return 0;
}
