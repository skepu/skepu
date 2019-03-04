#include <iostream>
#include <vector>
#include <skepu2.hpp>
#include "tests_common.hpp"

int custom_max(int a, int b) {
	if(a % 2 == 0 && b % 2 == 0)
		return 0;
	else if(a % 2 == 1 && b % 2 == 1)
		return a > b ? a : b;
	else if (a % 2 == 1)
		return a;
	else 
		return b;
}

int mult(int a, int b) {
	return a * b;
}

int add(int a, int b) {
	return a + b;
}

auto skeleton1D = skepu2::Reduce(custom_max);
auto skeleton2D = skepu2::Reduce(mult, add);

double runTestVector(const size_t size, const std::string& backend) {
	skepu2::Timer timer;
	
	skeleton1D.setStartValue(1);
	
	
	for(size_t iteration = 0; iteration < NUM_REPEATS; ++iteration) {
		skepu2::Vector<int> in1(size);
		in1.randomize();
		
		timer.start();
		skeleton1D(in1);
		timer.stop();
	}
	std::cout << "Test Vector. Backend: " << backend << " Load: " << size << " Time: " << timer.getMedianTime() << std::endl;
	
	return timer.getMedianTime();
}


double runTestMatrix1D(const size_t size, const std::string& backend) {
	size_t sqSize = size/100; //sqrt(size);
	skepu2::Timer timer;
	
	skeleton1D.setStartValue(1);
	
	for(size_t iteration = 0; iteration < NUM_REPEATS; ++iteration) {
		skepu2::Matrix<int> in1(200, sqSize);
		skepu2::Vector<int> res(200);
		in1.randomize();
		timer.start();
		skeleton1D(res, in1);
		timer.stop();
	}
	std::cout << "Test Matrix 1D. Backend: " << backend << " Load: 200x" << sqSize << " Time: " << timer.getMedianTime() << std::endl;
	
	return timer.getMedianTime();
}


double runTestMatrix2D(const size_t size, const std::string& backend) {
	size_t sqSize = size/100; //sqrt(size);
	skepu2::Timer timer;
	
	skeleton2D.setStartValue(1);
	
	for(size_t iteration = 0; iteration < NUM_REPEATS; ++iteration) {
		skepu2::Matrix<int> in1(200, sqSize);
		in1.randomize();
		timer.start();
		int _res = skeleton2D(in1);
		timer.stop();
	}
	std::cout << "Test Matrix 2D. Backend: " << backend << " Load: 200x" << sqSize << " Time: " << timer.getMedianTime() << std::endl;
	
	return timer.getMedianTime();
}



int main(int argc, char* argv[]) {
	std::vector<size_t> problemSizes;
	std::vector<double> cpuTimes;
	std::vector<double> gpuTimes;
	std::vector<double> hybridTimes;
	
	for(size_t i = 100000; i <= 4000000; i += 100000)
		problemSizes.push_back(i);
	
	auto TEST_FUNCTION = runTestVector;
// 	auto TEST_FUNCTION = runTestMatrix1D;
// 	auto TEST_FUNCTION = runTestMatrix2D;
	
	printInfo("Running OpenMP CPU backend");
	skepu2::BackendSpec specCPU(skepu2::Backend::Type::OpenMP);
	specCPU.setCPUThreads(16);
	skeleton1D.setBackend(specCPU);
	skeleton2D.setBackend(specCPU);
	for(size_t size : problemSizes) {
		double time = TEST_FUNCTION(size, "OpenMP");
		cpuTimes.push_back(time);
	}
	
	printInfo("Running CUDA GPU backend");
	skepu2::BackendSpec specGPU(skepu2::Backend::Type::CUDA);
	specGPU.setDevices(1);
	skeleton1D.setBackend(specGPU);
	skeleton2D.setBackend(specGPU);
	for(size_t size : problemSizes) {
		double time = TEST_FUNCTION(size, "CUDA");
		gpuTimes.push_back(time);
	}
	
	printInfo("Running Hybrid backend");
// 	skepu2::BackendSpec specHybrid(skepu2::Backend::Type::Hybrid);
// 	specHybrid.setCPUThreads(16);
// 	specHybrid.setDevices(1);
// 	skeleton1D.setBackend(specHybrid);
// 	skeleton2D.setBackend(specHybrid);
	
	skepu2::backend::tuner::hybridTune(skeleton1D, 16, 1, 50000, 4000000);
	skeleton1D.resetBackend();
// 	skepu2::backend::tuner::hybridTune(skeleton2D, 16, 1, 64, 4096);
// 	skeleton2D.resetBackend();
	for(size_t size : problemSizes) {
 		float percentage = 0.91;
// 		specHybrid.setCPUPartitionRatio(percentage);
		hybridTimes.push_back(TEST_FUNCTION(size, "Hybrid"));
	}
	
	savePerformanceTest("times_reduce.csv", problemSizes, cpuTimes, gpuTimes, hybridTimes);
		
	printOk("Test ended successfully");
	return 0;
}

