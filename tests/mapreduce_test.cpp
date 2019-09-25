#include <iostream>
#include <skepu>
#include "tests_common.hpp"

const size_t NUM_ELEMENTS = 10000;


int mult(int a, int b) {
	return a * b;
}


int add(int a, int b) {
	return a + b;
}



void testNormal() {
	printInfo("Running regular dotproduct test");
	auto dot_prod = skepu::MapReduce<2>(mult, add);
	dot_prod.setStartValue(10);
	
	skepu::Vector<int> in1(NUM_ELEMENTS);
	skepu::Vector<int> in2(NUM_ELEMENTS);
	in1.randomize();
	in2.randomize();
	
	printInfo("Running sequential CPU backend");
	skepu::BackendSpec spec1(skepu::Backend::Type::CPU);
	dot_prod.setBackend(spec1);
	int seqRes = dot_prod(in1, in2);
	
	printInfo("Running hybrid execution backend");
	skepu::BackendSpec spec2(skepu::Backend::Type::Hybrid);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.5);
	dot_prod.setBackend(spec2);
	int hybridRes = dot_prod(in1, in2);
	
	if(seqRes != hybridRes) {
		printFail("Mismatch in results, CPU gave: " + std::to_string(seqRes) + ", hybrid gave: " + std::to_string(hybridRes));
		exit(1);
	}
	printOk("Test ended successfully");
}


int gen(skepu::Index1D idx, int elem) {
	return idx.i;
}


void testIndex() {
	printInfo("Running Index1D test");
	auto seq_sum = skepu::MapReduce<1>(gen, add);
	seq_sum.setStartValue(10);
	
	skepu::Vector<int> in(NUM_ELEMENTS);
	in.randomize();
	
	printInfo("Running sequential CPU backend");
	skepu::BackendSpec spec1(skepu::Backend::Type::CPU);
	seq_sum.setBackend(spec1);
	int seqRes = seq_sum(in);
	
	printInfo("Running hybrid execution backend");
	skepu::BackendSpec spec2(skepu::Backend::Type::Hybrid);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.5);
	seq_sum.setBackend(spec2);
	int hybridRes = seq_sum(in);
	
	if(seqRes != hybridRes) {
		printFail("Mismatch in results, CPU gave: " + std::to_string(seqRes) + ", hybrid gave: " + std::to_string(hybridRes));
		exit(1);
	}
	printOk("Test ended successfully");
}


int main(int argc, char* argv[]) {
	testNormal();
	testIndex();
	return 0;
}
