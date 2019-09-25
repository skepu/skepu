#include <iostream>
#include <skepu>
#include "tests_common.hpp"

const size_t NUM_ELEMENTS = 30;


int sum(int overlap, size_t stride, const int* v) {
	int res = 0;
	for(int i = -overlap; i <= overlap; ++i) {
		res += v[i*stride];
	}
	return res;
}


int main(int argc, char* argv[]) {
	auto skeleton = skepu::MapOverlap(sum);
	
	skepu::Vector<int> in(NUM_ELEMENTS);
	skepu::Matrix<int> inMat(NUM_ELEMENTS, NUM_ELEMENTS);
	in.randomize(0, 19);
	inMat.randomize(0, 19);
	
	skepu::Vector<int> seqRes(NUM_ELEMENTS);
	skepu::Vector<int> hybridRes(NUM_ELEMENTS);
	skepu::Matrix<int> seqMatRes(NUM_ELEMENTS, NUM_ELEMENTS);
	skepu::Matrix<int> hybridMatRes(NUM_ELEMENTS, NUM_ELEMENTS);
	skepu::Matrix<int> hybridMatRes2(NUM_ELEMENTS, NUM_ELEMENTS);
	
	skeleton.setOverlap(2);
	skeleton.setPad(42);
	skeleton.setEdgeMode(skepu::Edge::Pad);

	std::cout <<"##### Running Vector test #####" << std::endl;
	printInfo("Running sequential CPU backend");
	skepu::BackendSpec spec1(skepu::Backend::Type::CPU);
	skeleton.setBackend(spec1);
	skeleton(seqRes, in);
	
	printInfo("Running hybrid execution backend");
	skepu::BackendSpec spec2(skepu::Backend::Type::Hybrid);
	spec2.setDevices(1);
	spec2.setCPUPartitionRatio(0.5);
	skeleton.setBackend(spec2);
	skeleton(hybridRes, in);
	
	int counter = 0;
	for(size_t i = 0; i < NUM_ELEMENTS; ++i) {
		if(seqRes[i] != hybridRes[i]) {
			printFail("Mismatch in results for vector MapOverlap, on index: " + std::to_string(i) + " CPU gave:" + std::to_string(seqRes[i]) + " Hybrid gave:" + std::to_string(hybridRes[i]));
			if(counter++ > 10)
				break;
		}
	}
	if(counter == 0)
		printOk("MapOverlap Vector test ended successfully");
	
	std::cout <<"##### Running Row wise Matrix test #####" << std::endl;
	skeleton.setOverlapMode(skepu::Overlap::RowWise);
	skeleton.setBackend(spec1);
	skeleton(seqMatRes, inMat);
	
	skeleton.setBackend(spec2);
	skeleton(hybridMatRes, inMat);
	counter = 0;
	for(size_t r = 0; r < NUM_ELEMENTS; ++r) {
		for(size_t c = 0; c < NUM_ELEMENTS; ++c) {
			if(seqMatRes(r, c) != hybridMatRes(r, c)) {
				printFail("Mismatch in results for matrix MapOverlap RowWise, on index: " + std::to_string(r) + "," + std::to_string(c) + " CPU gave:" + std::to_string(seqMatRes(r, c)) + " Hybrid gave:" + std::to_string(hybridMatRes2(r, c)));
				if(counter++ > 10)
					break;
			}
		}
	}
	
	if(counter == 0)
		printOk("MapOverlap Row wise Matrix test ended successfully");
	
#if 0
	// This is commented out until ColWise MapOverlap for Hybrid is fixed
	std::cout <<"##### Running Column wise Matrix test #####" << std::endl;
	
	skeleton.setOverlapMode(skepu::Overlap::ColWise);
	skeleton.setBackend(spec1);
	skeleton(seqMatRes, inMat);
	
	skeleton.setBackend(spec2);
	skeleton(hybridMatRes2, inMat);
	counter = 0;
	for(size_t r = 0; r < NUM_ELEMENTS; ++r) {
		for(size_t c = 0; c < NUM_ELEMENTS; ++c) {
			if(seqMatRes(r, c) != hybridMatRes2(r, c)) {
				printFail("Mismatch in results for matrix MapOverlap ColWise, on index: " + std::to_string(r) + "," + std::to_string(c) + " CPU gave:" + std::to_string(seqMatRes(r, c)) + " Hybrid gave:" + std::to_string(hybridMatRes2(r, c)));
				if(counter++ > 10)
					break;
			}
		}
	}
	printErrorMatrix(hybridMatRes2, seqMatRes);
	if(counter == 0)
		printOk("MapOverlap Row wise Matrix test ended successfully");
#endif
	return 0;
}
