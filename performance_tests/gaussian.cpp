#include <iostream>
#include <utility>
#include <cfloat>

#include <skepu2.hpp>
#include "performance_tests_common.hpp"

const size_t N = 700000;
std::string application = "Gaussian";


float gauss_kernel(int overlap, size_t stride, const float* v) {
	float gaussianWeights[] = { 0.382925, 0.24173, 0.060598, 0.005977, 0.000229, 0.000003 };
	float res = v[0]*gaussianWeights[0];
	for(int i = 1; i <= overlap; ++i) {
		res += v[i*stride]*gaussianWeights[i];
		res += v[-i*stride]*gaussianWeights[i];
	}
	return res;
}


auto gauss = skepu2::MapOverlap(gauss_kernel);

double gaussianBlur1D() {
	skepu2::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		skepu2::Vector<float> a(N), out(N);
		a.randomize(0, 3);
		gauss.setOverlap(5);
		
		timer.start();
		gauss(out, a);
		timer.stop();
	}
	return timer.getMedianTime();
}


constexpr auto benchmarkFunc = gaussianBlur1D;

void setBackend(const skepu2::BackendSpec& spec) {
	gauss.setBackend(spec);
}

void tune() {
	skepu2::backend::tuner::hybridTune(gauss);
	gauss.resetBackend();
}

int main(int argc, char *argv[]) {
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

