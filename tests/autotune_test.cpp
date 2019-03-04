#include <iostream>
#include <skepu2.hpp>
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

// auto skeleton = skepu2::Reduce(add);
// auto skeleton = skepu2::MapReduce<2>(mult, add);

// auto skeleton = skepu2::Map<2>(add);
auto skeleton = skepu2::MapOverlap(sum);
// auto skeleton = skepu2::Scan(add);

double runSingleTest(std::string str, size_t size) {
	skepu2::Timer timer;
	for(size_t test = 0; test < NUM_TESTS; ++test) {
		skepu2::Vector<int> in1(size);
// 		skepu2::Vector<int> in2(size);
		skepu2::Vector<int> out(size);
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
	skepu2::BackendSpec openMPBackend(skepu2::Backend::Type::OpenMP);
	openMPBackend.setCPUThreads(15);
	runSingleTest("OpenMP", 3912843/2, openMPBackend);
	
	skepu2::BackendSpec openCLBackend(skepu2::Backend::Type::CUDA);
	openCLBackend.setDevices(1);
	runSingleTest("OpenCL", 3912843/2+1, openCLBackend);
	
	skepu2::BackendSpec hybridBackend(skepu2::Backend::Type::Hybrid);
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
		skepu2::BackendSpec spec(skepu2::Backend::Type::Hybrid);
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
	skepu2::BackendSpec cpuSpec(skepu2::Backend::Type::CPU);
	skeleton.setBackend(cpuSpec);
	double cpuTime = runSingleTest("CPU", NUM_ELEMENTS);
	
	printInfo("Running OpenMP");
	skepu2::BackendSpec openmpSpec(skepu2::Backend::Type::OpenMP);
	openmpSpec.setCPUThreads(16);
	skeleton.setBackend(openmpSpec);
	double openmpTime = runSingleTest("OpenMP", NUM_ELEMENTS);
	
	
	printInfo("Running CUDA/OpenCL");
	skepu2::BackendSpec gpuSpec(skepu2::Backend::Type::CUDA);
	gpuSpec.setDevices(1);
	skeleton.setBackend(gpuSpec);
	double gpuTime = runSingleTest("CUDA/OpenCL", NUM_ELEMENTS);
	
	
	printInfo("Running auto-tuner");
	skepu2::backend::tuner::hybridTune(skeleton);

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
