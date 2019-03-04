/*!
 *  PPMCC stands for "Pearson product-moment correlation coefficient"
 *  In statistics, the Pearson coefficient of correlation is a measure by the
 *  linear dependence between two variables X and Y. The mathematical
 *  expression of the Pearson coefficient of correlation is as follows:
 *   r = ( (n*sum(X.Y)-sum(X)*sum(Y))/((n*sum(X^2)-(sum(X))^2)*(n*sum(Y^2)-(sum(Y))^2)) )
 */

#include <iostream>
#include <cmath>

#include <skepu2.hpp>
#include "performance_tests_common.hpp"


const size_t N = 250000;
std::string application = "PPMCC";

// Unary user-function used for mapping
template<typename T>
T square(T a)
{
	return a * a;
}

// Binary user-function used for mapping
template<typename T>
T mult(T a, T b)
{
	return a * b;
}

// User-function used for reduction
template<typename T>
T plus(T a, T b)
{
	return a + b;
}


using T = float;

// Skeleton definitions
auto sum = skepu2::Reduce(plus<T>);
auto dotProduct = skepu2::MapReduce<2>(mult<T>, plus<T>);
auto sumSquare = skepu2::MapReduce<1>(square<T>, plus<T>);

T ppmcc() {
	skepu2::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		// Vector operands
		skepu2::Vector<T> x(N), y(N);
		x.randomize(1, 3);
		y.randomize(2, 4);

		timer.start();
		T sumX = sum(x);
		T sumY = sum(y);
		
		T res = (N * dotProduct(x, y) - sumX * sumY)
			/ sqrt((N * sumSquare(x) - pow(sumX, 2)) * (N * sumSquare(y) - pow(sumY, 2)));
		timer.stop();
	}
	return timer.getMedianTime();
}


constexpr auto benchmarkFunc = ppmcc;

void setBackend(const skepu2::BackendSpec& spec) {
	sum.setBackend(spec);
	dotProduct.setBackend(spec);
	sumSquare.setBackend(spec);
}

void tune() {
	skepu2::backend::tuner::hybridTune(sum);
	skepu2::backend::tuner::hybridTune(dotProduct);
	skepu2::backend::tuner::hybridTune(sumSquare);
	sum.resetBackend();
	dotProduct.resetBackend();
	sumSquare.resetBackend();
}

int main(int argc, char* argv[]) {
	std::vector<double> times;
	
	std::cout << application << ": Running CPU backend" << std::endl;
	skepu2::BackendSpec specCPU(skepu2::Backend::Type::CPU);
	setBackend(specCPU);
	double cpuTime = benchmarkFunc();
	
	std::cout << application << ": Running OpenMP backend" << std::endl;
	skepu2::BackendSpec specOpenMP(skepu2::Backend::Type::OpenMP);
	specOpenMP.setCPUThreads(16);
	setBackend(specOpenMP);
	times.push_back(benchmarkFunc());
	
	std::cout << application << ": Running CUDA GPU backend" << std::endl;
	skepu2::BackendSpec specGPU(skepu2::Backend::Type::CUDA);
	specGPU.setDevices(1);
	setBackend(specGPU);
	times.push_back(benchmarkFunc());
	
	
	std::cout << application << ": Running Oracle" << std::endl;
	double bestOracleTime = 100000000.0;
	size_t bestOracleRatio = 9999999;
	std::vector<double> oracleTimes;
	
	for(size_t ratio = 0; ratio <= 100; ratio += 5) {
		for(size_t ratio2 = 0; ratio2 <= 100; ratio2 += 5) {
			for(size_t ratio3 = 0; ratio3 <= 100; ratio3 += 5) {
				double percentage = (double)ratio / 100.0;
				skepu2::BackendSpec spec(skepu2::Backend::Type::Hybrid);
				spec.setDevices(1);
				spec.setCPUThreads(16);
				spec.setCPUPartitionRatio(percentage);
				
				double percentage2 = (double)ratio2 / 100.0;
				skepu2::BackendSpec spec2(skepu2::Backend::Type::Hybrid);
				spec2.setDevices(1);
				spec2.setCPUThreads(16);
				spec2.setCPUPartitionRatio(percentage2);
				
				double percentage3 = (double)ratio3 / 100.0;
				skepu2::BackendSpec spec3(skepu2::Backend::Type::Hybrid);
				spec3.setDevices(1);
				spec3.setCPUThreads(16);
				spec3.setCPUPartitionRatio(percentage3);
				
// 				setBackend(spec);
				sum.setBackend(spec);
				dotProduct.setBackend(spec2);
				sumSquare.setBackend(spec3);
				double time = benchmarkFunc();
				
				oracleTimes.push_back(time);
				if(time < bestOracleTime) {
					bestOracleTime = time;
					bestOracleRatio = ratio;
				}
				std::cout << "Ratio: " << percentage << " gave time: " << time << std::endl;
			}
		}
	}
	times.push_back(bestOracleTime);
	std::cout << "Optimal ratio was: " << bestOracleRatio << std::endl;
	
	std::cout << application << ": Running Hybrid backend" << std::endl;
	tune();
	times.push_back(benchmarkFunc());
	
	appendPerformanceResult("speedup_openmp.dat", application, cpuTime/times[0]);
	appendPerformanceResult("speedup_cuda.dat", application, cpuTime/times[1]);
	appendPerformanceResult("speedup_oracle.dat", application, cpuTime/times[2]);
	appendPerformanceResult("speedup_hybrid.dat", application, cpuTime/times[3]);
	
	return 0;
}

