/*
 * Matrix-Matrix multiplication
 */

#include <iostream>
#include <cmath>

#include <skepu2.hpp>
#include "performance_tests_common.hpp"


const size_t N = 90;
std::string application = "mmmult";


template<typename T>
T arr(skepu2::Index2D idx, const skepu2::Mat<T> lhs, const skepu2::Mat<T> rhs)
{
	T res = 0;
	for (size_t i = 0; i < lhs.cols; ++i)
		res += lhs.data[idx.row * lhs.cols + i] * rhs.data[i * rhs.cols + idx.col];
	return res;
}


auto mmprod = skepu2::Map<0>(arr<float>);

double mmmult() {
	skepu2::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		skepu2::Matrix<float> lhs(N, N);
		skepu2::Matrix<float> rhs(N, N);
		skepu2::Matrix<float> res(N, N);
		lhs.randomize();
		rhs.randomize();
		
		timer.start();
		mmprod(res, lhs, rhs);
		timer.stop();
	}
	return timer.getMedianTime();
}


constexpr auto benchmarkFunc = mmmult;

void setBackend(const skepu2::BackendSpec& spec) {
	mmprod.setBackend(spec);
}

void tune() {
	skepu2::backend::tuner::hybridTune(mmprod, 16, 1, 16, 512);
	mmprod.resetBackend();
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