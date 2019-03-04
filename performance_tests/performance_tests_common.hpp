

const size_t NUM_REPEATS = 7;


void appendPerformanceResult(const std::string& filename, const std::string& problemName, double time) {
	std::cout << "Append performance test result to file: " << filename << std::endl;;
	std::ofstream stream;
	stream.open(filename, std::ios::out | std::ios::app);
	stream  << problemName << " " << time << std::endl;
	stream.close();
}
