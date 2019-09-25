#include <iostream>
#include <skepu>
#include "tests_common.hpp"

const size_t NUM_ELEMENTS = 6553600;
const size_t NUM_TESTS = 5;

int mult(int a, int b) {
	return a * b;
}

int add(int a, int b) {
	return a + b;
}


int sum(int overlap, size_t stride, const int* v) {
	int res = 0;
	for(int i = -overlap; i <= overlap; ++i) {
		res += v[i*stride];
	}
	return res;
}

// auto skeleton = skepu::Reduce(add);
// auto skeleton = skepu::MapReduce<2>(mult, add);

// auto skeleton = skepu::Map<2>(add);
auto skeleton = skepu::MapOverlap(sum);
// auto skeleton = skepu::Scan(add);

double runSingleTest(std::string str, size_t size) {
	skepu::Timer timer;
	for(size_t test = 0; test < NUM_TESTS; ++test) {
		skepu::Vector<int> in1(size);
// 		skepu::Vector<int> in2(size);
		skepu::Vector<int> out(size);
		in1.randomize();
// 		in2.randomize();
		
		timer.start();
		skeleton(out, in1);
		timer.stop();
	}
	return timer.getMedianTime();
}

#if 0
void testOneImplementation() {
	skepu::BackendSpec openMPBackend(skepu::Backend::Type::OpenMP);
	openMPBackend.setCPUThreads(15);
	runSingleTest("OpenMP", 3912843/2, openMPBackend);
	
	skepu::BackendSpec openCLBackend(skepu::Backend::Type::CUDA);
	openCLBackend.setDevices(1);
	runSingleTest("OpenCL", 3912843/2+1, openCLBackend);
	
	skepu::BackendSpec hybridBackend(skepu::Backend::Type::Hybrid);
	hybridBackend.setCPUThreads(16);
	hybridBackend.setDevices(1);
	hybridBackend.setCPUPartitionRatio(0.5);
	runSingleTest("Hybrid", 3912843, hybridBackend); // 4000000
}
#endif

void testTuning() {
	printInfo("Running oracle");
	
	skeleton.setOverlap(20);
	
	double bestOracleTime = 100000000.0;
	size_t bestOracleRatio = 9999999;
	std::vector<double> oracleTimes;
	
	for(size_t ratio = 0; ratio <= 100; ratio += 5) {
		double percentage = (double)ratio / 100.0;
		percentage = percentage;
		std::cout << "Set ratio to " << percentage << std::endl;
		skepu::BackendSpec spec(skepu::Backend::Type::Hybrid);
		spec.setDevices(1);
		spec.setCPUThreads(16);
		spec.setCPUPartitionRatio(percentage);
		
		skeleton.setBackend(spec);
		double time = runSingleTest("Oracle", NUM_ELEMENTS);
		
		oracleTimes.push_back(time);
		std::cout << time << ", ";
		if(time < bestOracleTime) {
			bestOracleTime = time;
			bestOracleRatio = ratio;
		}
	}
	
	std::cout << std::endl;
	
	printInfo("Running CPU");
	skepu::BackendSpec cpuSpec(skepu::Backend::Type::CPU);
	skeleton.setBackend(cpuSpec);
	double cpuTime = runSingleTest("CPU", NUM_ELEMENTS);
	
	printInfo("Running OpenMP");
	skepu::BackendSpec openmpSpec(skepu::Backend::Type::OpenMP);
	openmpSpec.setCPUThreads(16);
	skeleton.setBackend(openmpSpec);
	double openmpTime = runSingleTest("OpenMP", NUM_ELEMENTS);
	
	
	printInfo("Running CUDA/OpenCL");
	skepu::BackendSpec gpuSpec(skepu::Backend::Type::CUDA);
	gpuSpec.setDevices(1);
	skeleton.setBackend(gpuSpec);
	double gpuTime = runSingleTest("CUDA/OpenCL", NUM_ELEMENTS);
	
	
	printInfo("Running auto-tuner");
	skepu::backend::tuner::hybridTune(skeleton);

	skeleton.resetBackend();
	printInfo("Test auto-tuner");
	double hybridTime = runSingleTest("Hybrid", NUM_ELEMENTS);
	
	
	
	std::cout << "CPU seq time was: " << cpuTime << std::endl;
	std::cout << "OpenMP time was: " << openmpTime << std::endl;
	std::cout << "CUDA/OpenCL time was: " << gpuTime << std::endl;
	std::cout << "Oracle time was: " << bestOracleTime << " at partition ratio " << bestOracleRatio << "%" << std::endl;
	std::cout << "Hybrid time was: " << hybridTime << std::endl;
	
	std::cout << std::endl;
	
	double openmpSpeedup = cpuTime/openmpTime;
	double gpuSpeedup = cpuTime/gpuTime;
	double oracleSpeedup = cpuTime/bestOracleTime;
	double hybridSpeedup = cpuTime/hybridTime;
	
	std::cout << "OpenMP speedup was: " << openmpSpeedup << std::endl;
	std::cout << "CUDA/OpenCL speedup was: " << gpuSpeedup << std::endl;
	std::cout << "Oracle speedup was: " << oracleSpeedup << std::endl;
	std::cout << "Hybrid speedup was: " << hybridSpeedup << std::endl;
	std::cout << std::endl;
	
	
	std::cout << "(Application," << openmpSpeedup << ") " << std::endl;
	std::cout << "(Application," << gpuSpeedup << ") " << std::endl;
	std::cout << "(Application," << oracleSpeedup << ") " << std::endl;
	std::cout << "(Application," << hybridSpeedup << ") " << std::endl;
	
	printOk("Test ended");
}


int main(int argc, char* argv[]) {
	testTuning();
	return 0;
}
