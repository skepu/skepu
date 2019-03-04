#include <iostream>
#include <skepu2.hpp>
#include "tests_common.hpp"

const size_t NUM_ELEMENTS = 10000;
const size_t NUM_ELEMENTS_MAT = 100;

int add(int a, int b) {
	return a + b;
}


int initMatrixElem(skepu2::Index2D index, int a) {
	return index.col*a + index.row;
}

auto elementwise_add = skepu2::Map<2>(add);

auto init_matrix = skepu2::Map<1>(initMatrixElem);

void vectorTest() {
	skepu2::Vector<int> in1(NUM_ELEMENTS);
	skepu2::Vector<int> in2(NUM_ELEMENTS);
	in1.randomize();
	in2.randomize();
	
	skepu2::Vector<int> outSeq(NUM_ELEMENTS);
	skepu2::Vector<int> outHybrid(NUM_ELEMENTS);
	
	printInfo("Running sequential CPU backend");
	skepu2::BackendSpec spec1(skepu2::Backend::Type::CPU);
	elementwise_add.setBackend(spec1);
	elementwise_add(outSeq, in1, in2);
	
	printInfo("Running hybrid execution backend");
	skepu2::BackendSpec spec2(skepu2::Backend::Type::Hybrid);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.5);
	elementwise_add.setBackend(spec2);
	elementwise_add(outHybrid, in1, in2);

	for(size_t i = 0; i < NUM_ELEMENTS; ++i) {
		if(outSeq[i] != outHybrid[i]) {
			printFail("Mismatch in results, at index " + std::to_string(i) + " Got: " + std::to_string(outHybrid[i]) + " Expected: " + std::to_string(outSeq[i]));
			exit(1);
		}
	}
	printOk("Test ended successfully");
}

void matrixTest() {
	skepu2::Matrix<int> in1(NUM_ELEMENTS_MAT, NUM_ELEMENTS_MAT);
	skepu2::Matrix<int> in2(NUM_ELEMENTS_MAT, NUM_ELEMENTS_MAT);
	in1.randomize();
	in2.randomize();
	
	skepu2::Matrix<int> outSeq(NUM_ELEMENTS_MAT, NUM_ELEMENTS_MAT);
	skepu2::Matrix<int> outHybrid(NUM_ELEMENTS_MAT, NUM_ELEMENTS_MAT);
	
	printInfo("Running sequential CPU backend");
	skepu2::BackendSpec spec1(skepu2::Backend::Type::CPU);
	elementwise_add.setBackend(spec1);
	elementwise_add(outSeq, in1, in2);
	
	printInfo("Running hybrid execution backend");
	skepu2::BackendSpec spec2(skepu2::Backend::Type::Hybrid);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.2);
	elementwise_add.setBackend(spec2);
	elementwise_add(outHybrid, in1, in2);

	for(size_t i = 0; i < NUM_ELEMENTS_MAT; ++i) {
		for(size_t j = 0; j < NUM_ELEMENTS_MAT; ++j) {
			if(outSeq(i, j) != outHybrid(i, j)) {
				printFail("Mismatch in results, at index (" + std::to_string(i) + ", " + std::to_string(j) + ") Got: " + std::to_string(outHybrid(i, j)) + " Expected: " + std::to_string(outSeq(i, j)));
				exit(1);
			}
		}
	}
	printOk("Test ended successfully");
	
}


void matrixIndex2DTest() {
	skepu2::Matrix<int> in1(NUM_ELEMENTS_MAT, NUM_ELEMENTS_MAT);
	in1.randomize();
	
	skepu2::Matrix<int> outSeq(NUM_ELEMENTS_MAT, NUM_ELEMENTS_MAT);
	skepu2::Matrix<int> outHybrid(NUM_ELEMENTS_MAT, NUM_ELEMENTS_MAT);
	
	printInfo("Running sequential CPU backend");
	skepu2::BackendSpec spec1(skepu2::Backend::Type::CPU);
	init_matrix.setBackend(spec1);
	init_matrix(outSeq, in1);
	
	printInfo("Running hybrid execution backend");
	skepu2::BackendSpec spec2(skepu2::Backend::Type::Hybrid);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.2);
	init_matrix.setBackend(spec2);
	init_matrix(outHybrid, in1);

	for(size_t i = 0; i < NUM_ELEMENTS_MAT; ++i) {
		for(size_t j = 0; j < NUM_ELEMENTS_MAT; ++j) {
			if(outSeq(i, j) != outHybrid(i, j)) {
				printFail("Mismatch in results, at index (" + std::to_string(i) + ", " + std::to_string(j) + ") Got: " + std::to_string(outHybrid(i, j)) + " Expected: " + std::to_string(outSeq(i, j)));
				exit(1);
			}
		}
	}
	printOk("Test ended successfully");
	
}

int main(int argc, char* argv[]) {
	vectorTest();
	matrixTest();
	matrixIndex2DTest();
	return 0;
}
