/*
 * Taylor series calculation, natural log(1+x)  sum(1:N) (((-1)^(i+1))/i)*x^i
 */

#include <iostream>
#include <cmath>

#include <skepu>
#include "performance_tests_common.hpp"


const float x = 0.556;
const size_t N = 30000;
std::string application = "Taylor";


float nth_term(skepu::Index1D index, float x)
{
	int k = index.i + 1;
	float temp_x = pow(x, k);
	int sign = (k % 2 == 0) ? -1 : 1;
	return sign * temp_x / k;
}

float plus(float a, float b)
{
	return a + b;
}

auto taylor = skepu::MapReduce<0>(nth_term, plus);

double taylor_approx() {
	skepu::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		taylor.setDefaultSize(N);
		timer.start();
		taylor(x - 1);
		timer.stop();
	}
	return timer.getMedianTime();
}


constexpr auto benchmarkFunc = taylor_approx;

void setBackend(const skepu::BackendSpec& spec) {
	taylor.setBackend(spec);
}

void tune() {
	skepu::backend::tuner::hybridTune(taylor);
	taylor.resetBackend();
}

int main(int argc, char* argv[]) {
	std::vector<double> times;
	
	std::cout << application << ": Running CPU backend" << std::endl;
	skepu::BackendSpec specCPU(skepu::Backend::Type::CPU);
	setBackend(specCPU);
	double cpuTime = benchmarkFunc();
	
	std::cout << application << ": Running OpenMP backend" << std::endl;
	skepu::BackendSpec specOpenMP(skepu::Backend::Type::OpenMP);
	specOpenMP.setCPUThreads(16);
	setBackend(specOpenMP);
	times.push_back(benchmarkFunc());
	
	std::cout << application << ": Running CUDA GPU backend" << std::endl;
	skepu::BackendSpec specGPU(skepu::Backend::Type::CUDA);
	specGPU.setDevices(1);
	setBackend(specGPU);
	times.push_back(benchmarkFunc());
	
	
	std::cout << application << ": Running Oracle" << std::endl;
	double bestOracleTime = 100000000.0;
	size_t bestOracleRatio = 9999999;
	std::vector<double> oracleTimes;
	
	for(size_t ratio = 0; ratio <= 100; ratio += 5) {
		double percentage = (double)ratio / 100.0;
		skepu::BackendSpec spec(skepu::Backend::Type::Hybrid);
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
