#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include <skepu2.hpp>
#include "performance_tests_common.hpp"


const size_t iterations = 4;
const size_t N = 100;
std::string application = "NBody";


// Particle data structure that is used as an element type.
struct Particle
{
	float x, y, z;
	float vx, vy, vz;
	float m;
};


constexpr float G [[skepu::userconstant]] = 1;
constexpr float delta_t [[skepu::userconstant]] = 0.1;


/*
 * Array user-function that is used for applying nbody computation,
 * All elements from parr and a single element (named 'pi') are accessible
 * to produce one output element of the same type.
 */
Particle move(skepu2::Index1D index, Particle pi, const skepu2::Vec<Particle> parr)
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



// A helper function to write particle output values to standard output stream.
void save_step(skepu2::Vector<Particle> &particles, std::ostream &os = std::cout)
{
	int i = 0;
	for (Particle &p : particles)
	{
		os << std::setw( 4) << i++
			<< std::setw(15) << p.x
			<< std::setw(15) << p.y
			<< std::setw(15) << p.z
			<< std::setw(15) << p.vx
			<< std::setw(15) << p.vy
			<< std::setw(15) << p.vz << "\n";
	}
}

//! A helper function to write particle output values to a file.
void save_step(skepu2::Vector<Particle> &particles, const std::string &filename)
{
	std::ofstream out(filename);
	
	if (out.is_open())
		save_step(particles, out);
	else
		std::cerr << "Error: cannot open this file: " << filename << "\n";
}


auto nbody_init = skepu2::Map<0>(init);
auto nbody_simulate_step = skepu2::Map<1>(move);

double nbody() {
	skepu2::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		skepu2::Vector<Particle> particles(N);
		skepu2::Vector<Particle> doublebuffer(N);
		
		timer.start();
		// particle vectors initialization
		nbody_init(particles, N);
		
		for (size_t i = 0; i < iterations; i += 2)
		{
			nbody_simulate_step(doublebuffer, particles, particles);
			nbody_simulate_step(particles, doublebuffer, doublebuffer);
		}
		timer.stop();
	}
	return timer.getMedianTime();
}


constexpr auto benchmarkFunc = nbody;

void setBackend(const skepu2::BackendSpec& spec) {
	nbody_init.setBackend(spec);
	nbody_simulate_step.setBackend(spec);
}

void tune() {
	skepu2::backend::tuner::hybridTune(nbody_init);
	skepu2::backend::tuner::hybridTune(nbody_simulate_step);
	nbody_init.resetBackend();
	nbody_simulate_step.resetBackend();
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

