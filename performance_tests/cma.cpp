#include <iostream>
#include <skepu>
#include "performance_tests_common.hpp"


const size_t N = 500000;
std::string application = "CMA";

template<typename T>
T sum(T a, T b)
{
	return a + b;
}

template<typename T, typename U>
U avg(skepu::Index1D index, T sum)
{
	return (U)sum / (index.i + 1);
}


auto prefix_sum = skepu::Scan(sum<int>);
auto average = skepu::Map<1>(avg<int, float>);

double cma() {
	skepu::Timer timer;
	for(size_t test = 0; test < NUM_REPEATS; ++test) {
		skepu::Vector<int> in(N);
		skepu::Vector<float> out(N);
		in.randomize(0, 10);
		
		timer.start();
		prefix_sum(in, in);
		average(out, in);
		timer.stop();
	}
	return timer.getMedianTime();
}


constexpr auto benchmarkFunc = cma;

void setBackend(const skepu::BackendSpec& spec) {
	prefix_sum.setBackend(spec);
	average.setBackend(spec);
}

void tune() {
	skepu::backend::tuner::hybridTune(prefix_sum);
	prefix_sum.resetBackend();
	skepu::backend::tuner::hybridTune(average);
	average.resetBackend();
}

int main(int argc, char *argv[]) {
	std::vector<double> times;
	
	std::cout << application << ": Running CPU backend" << std::endl;
	skepu::BackendSpec specCPU(skepu::Backend::Type::CPU);
	setBackend(specCPU);
	double cpuTime = benchmarkFunc();
	
	std::cout << application << ": Running OpenMP backend" << std::endl;
	skepu::BackendSpec specOpenMP(skepu::Backend::Type::OpenMP);
	specOpenMP.setCPUThreads(16);
	setBackend(specOpenMP);
	times.push_back(benchmarkFunc());
	
	std::cout << application << ": Running CUDA GPU backend" << std::endl;
	skepu::BackendSpec specGPU(skepu::Backend::Type::CUDA);
	specGPU.setDevices(1);
	setBackend(specGPU);
	times.push_back(benchmarkFunc());
	
	
	std::cout << application << ": Running Oracle" << std::endl;
	double bestOracleTime = 100000000.0;
	size_t bestOracleRatio = 9999999;
	std::vector<double> oracleTimes;
	
	for(size_t ratio = 0; ratio <= 100; ratio += 5) {
		for(size_t ratio2 = 0; ratio2 <= 100; ratio2 += 5) {
			double percentage = (double)ratio / 100.0;
			skepu::BackendSpec spec(skepu::Backend::Type::Hybrid);
			spec.setDevices(1);
			spec.setCPUThreads(16);
			spec.setCPUPartitionRatio(percentage);
			
			double percentage2 = (double)ratio2 / 100.0;
			skepu::BackendSpec spec2(skepu::Backend::Type::Hybrid);
			spec2.setDevices(1);
			spec2.setCPUThreads(16);
			spec2.setCPUPartitionRatio(percentage2);
			
// 			setBackend(spec);
			
			prefix_sum.setBackend(spec);
			average.setBackend(spec2);
			double time = benchmarkFunc();
			
			oracleTimes.push_back(time);
			if(time < bestOracleTime) {
				bestOracleTime = time;
				bestOracleRatio = ratio;
			}
			std::cout << "Ratio: " << percentage << " gave time: " << time << std::endl;
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
