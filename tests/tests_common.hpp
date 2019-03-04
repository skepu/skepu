#pragma once

#include <skepu2.hpp>
#include <iostream>

const size_t NUM_REPEATS = 7;

void printOk(const std::string& msg) {
	std::cout << "[\033[1;32m OK \033[0m] " << msg << std::endl;
}


void printFail(const std::string& msg) {
	std::cout << "[\033[1;31mFAIL\033[0m] " << msg << std::endl;
}

void printInfo(const std::string& msg) {
	std::cout << "[INFO]" << msg << std::endl;
}

void printErrorMatrix(const skepu2::Matrix<int>& hybrid, const skepu2::Matrix<int>& seq) {
	for(size_t r = 0; r < hybrid.total_rows(); ++r) {
		for(size_t c = 0; c < hybrid.total_cols(); ++c) {
			if(hybrid(r, c) != seq(r, c))
				std::cout << "\033[1;31m" << hybrid(r, c) << "\033[0m ";
			else 
				std::cout << "\033[1;32mY\033[0m ";
		}
		std::cout << std::endl;
	}
}


void savePerformanceTest(const std::string& filename, const std::vector<size_t>& problemSizes, const std::vector<double>& cpuTimes, 
						 const std::vector<double>& gpuTimes, const std::vector<double>& hybridTimes) {
	printInfo("Write performance test result to file: " + filename);
	std::ofstream stream;
	stream.open(filename, std::ios::out | std::ios::trunc);
	
	stream << "x,cpu,gpu,hybrid" << std::endl;
	for(size_t i = 0; i < problemSizes.size(); ++i) {
		stream << problemSizes[i] << "," << (cpuTimes[i]/1000.0) << "," << (gpuTimes[i]/1000.0) << "," << (hybridTimes[i]/1000.0) << std::endl;
	}
	stream.close();
}
