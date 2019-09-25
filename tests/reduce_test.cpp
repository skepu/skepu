#include <iostream>
#include <skepu>
#include "tests_common.hpp"

const size_t NUM_ELEMENTS = 10000;

int mult(int a, int b) {
	return a * b;
}

int sum(int a, int b) {
	return a + b;
}

int main(int argc, char* argv[]) {
	auto sum_1d = skepu::Reduce(sum);
// 	sum_1d.setStartValue(1284);
	skepu::Vector<int> in(NUM_ELEMENTS);
	skepu::Matrix<int> inMat(NUM_ELEMENTS, NUM_ELEMENTS);
	in.randomize();
	inMat.randomize();
	
	skepu::Vector<int> seqRes1DMat(NUM_ELEMENTS);
	skepu::Vector<int> hybridRes1DMat(NUM_ELEMENTS);
	
	std::cout <<"##### Running 1D test #####" << std::endl;
	
	printInfo("Running sequential CPU backend");
	skepu::BackendSpec spec1(skepu::Backend::Type::CPU);
	sum_1d.setBackend(spec1);
	int seqRes1DVec = sum_1d(in);
	sum_1d(seqRes1DMat, inMat);
	
	printInfo("Running hybrid execution backend");
	skepu::BackendSpec spec2(skepu::Backend::Type::Hybrid);
	spec2.setCPUThreads(16);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.5);
	sum_1d.setBackend(spec2);
	
	int hybridRes1DVec = sum_1d(in);
	
	if(seqRes1DVec != hybridRes1DVec) {
		printFail("Mismatch in 1D vector results, CPU gave: " + std::to_string(seqRes1DVec) + ", hybrid gave: " + std::to_string(hybridRes1DVec));
		exit(1);
	}
	printOk("Test in 1D vector ended successfully");
	
	
	sum_1d(hybridRes1DMat, inMat);
	for(size_t i = 0; i < NUM_ELEMENTS; ++i) {
		if(seqRes1DMat[i] != hybridRes1DMat[i]) {
			printFail("Mismatch in 1D matrix results, at index: " + std::to_string(i) +  ". CPU gave: " + std::to_string(seqRes1DMat[i]) + ", hybrid gave: " + std::to_string(hybridRes1DMat[i]));
			exit(1);
		}
	}
	printOk("Test in 1D matrix ended successfully");
	
	
	std::cout << "##### Running 2D test #####" << std::endl;
	auto sum_2d = skepu::Reduce(sum, mult);
// 	sum_2d.setStartValue(48839);
	
	printInfo("Running sequential CPU backend");
	sum_2d.setBackend(spec1);
	int seqRes = sum_2d(inMat);
	
	printInfo("Running hybrid execution backend");
	sum_2d.setBackend(spec2);
	int hybridRes = sum_2d(inMat);
	
	if(seqRes != hybridRes) {
		printFail("Mismatch in 2D results, CPU gave: " + std::to_string(seqRes) + ", hybrid gave: " + std::to_string(hybridRes));
		exit(1);
	}
	printOk("Test in 2D ended successfully");
	
	return 0;
}
