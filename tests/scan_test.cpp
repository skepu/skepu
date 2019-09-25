#include <iostream>
#include <skepu>
#include "tests_common.hpp"

const size_t NUM_ELEMENTS = 10000;

int add(int a, int b) {
	return a + b;
}

int main(int argc, char* argv[]) {
	auto partial_add = skepu::Scan(add);
	
	printInfo("##### Running Inclusive test #####");
	partial_add.setScanMode(skepu::ScanMode::Inclusive);
	
	skepu::Vector<int> in(NUM_ELEMENTS);
	in.randomize();
	
	skepu::Vector<int> outSeq(NUM_ELEMENTS);
	skepu::Vector<int> outHybrid(NUM_ELEMENTS);
	
	printInfo("Running sequential CPU backend");
	skepu::BackendSpec spec1(skepu::Backend::Type::CPU);
	partial_add.setBackend(spec1);
	partial_add(outSeq, in);
	
	printInfo("Running hybrid execution backend");
	skepu::BackendSpec spec2(skepu::Backend::Type::Hybrid);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.5);
	partial_add.setBackend(spec2);
	partial_add(outHybrid, in);
	
	for(size_t i = 0; i < NUM_ELEMENTS; ++i) {
		if(outSeq[i] != outHybrid[i]) {
			printFail("Mismatch in results, at index " + std::to_string(i) + " Sequential is:" + std::to_string(outSeq[i]) + " Hybrid is:" + std::to_string(outHybrid[i]));
			exit(1);
		}
	}
	printOk("Inclusive test ended successfully");
	
	
	
	printInfo("##### Running Exclusive test #####");
	partial_add.setScanMode(skepu::ScanMode::Exclusive);
	partial_add.setStartValue(42);
	
	skepu::Vector<int> outSeqEx(NUM_ELEMENTS);
	skepu::Vector<int> outHybridEx(NUM_ELEMENTS);
	
	printInfo("Running sequential CPU backend");
	partial_add.setBackend(spec1);
	partial_add(outSeqEx, in);
	
	printInfo("Running hybrid execution backend");
	partial_add.setBackend(spec2);
	partial_add(outHybridEx, in);
	
	for(size_t i = 0; i < NUM_ELEMENTS; ++i) {
		if(outSeqEx[i] != outHybridEx[i]) {
			printFail("Mismatch in results, at index " + std::to_string(i) + " Sequential is:" + std::to_string(outSeqEx[i]) + " Hybrid is:" + std::to_string(outHybridEx[i]) + " diff:" + std::to_string(outHybridEx[i] - outSeqEx[i]));
			exit(1);
		}
	}
	printOk("Exclusive test ended successfully");
	
	return 0;
}
