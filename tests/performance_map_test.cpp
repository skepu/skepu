#include <iostream>
#include <vector>
#include <skepu>
#include "tests_common.hpp"

int add(int a, int b) {
	return (a*a + b*b);
}

auto skeleton = skepu::Map<2>(add);

double runTest(const size_t size, const std::string& backend) {
	skepu::Timer timer;
	
	for(size_t iteration = 0; iteration < NUM_REPEATS; ++iteration) {
		skepu::Vector<int> in1(size);
		skepu::Vector<int> in2(size);
		skepu::Vector<int> out(size);
		in1.randomize();
		in2.randomize();
		
		timer.start();
		skeleton(out, in1, in2);
		timer.stop();
	}
	std::cout << "Backend: " << backend << " Load: " << size << " Time: " << timer.getMedianTime() << std::endl;
	
// 	for(size_t i = 0; i < size; ++i) {
// 		if(add(in1[i], in2[i]) != out[i]) {
// 			std::cout << "ERROR: Mismatch at index " << i << std::endl;
// 		}
// 	}
	
	return timer.getMedianTime();
}

int main(int argc, char* argv[]) {
	std::vector<size_t> problemSizes;
	std::vector<double> cpuTimes;
	std::vector<double> gpuTimes;
	std::vector<double> hybridTimes;
	
	for(size_t i = 100000; i <= 4000000; i += 100000)
		problemSizes.push_back(i);
	
	printInfo("Running OpenMP CPU backend");
	skepu::BackendSpec specCPU(skepu::Backend::Type::OpenMP);
	specCPU.setCPUThreads(16);
	skeleton.setBackend(specCPU);
	for(size_t size : problemSizes) {
		double time = runTest(size, "OpenMP");
		cpuTimes.push_back(time);
	}
	
	printInfo("Running CUDA GPU backend");
	skepu::BackendSpec specGPU(skepu::Backend::Type::CUDA);
	specGPU.setDevices(1);
	skeleton.setBackend(specGPU);
	for(size_t size : problemSizes) {
		double time = runTest(size, "CUDA");
		gpuTimes.push_back(time);
	}
	
	printInfo("Running Hybrid backend");
// 	skepu::BackendSpec specHybrid(skepu::Backend::Type::Hybrid);
// 	specHybrid.setCPUThreads(16);
// 	specHybrid.setDevices(1);
// 	skeleton.setBackend(specHybrid);

	skepu::backend::tuner::hybridTune(skeleton, 16, 1, 50000, 4000000);
	skeleton.resetBackend();
	for(size_t size : problemSizes) {
 		float percentage = 0.45;
// 		specHybrid.setCPUPartitionRatio(percentage);
		hybridTimes.push_back(runTest(size, "Hybrid"));
	}
	
	savePerformanceTest("times_map.csv", problemSizes, cpuTimes, gpuTimes, hybridTimes);
		
	printOk("Test ended successfully");
	return 0;
}

