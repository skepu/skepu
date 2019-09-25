#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <vector>
#include <tuple>

#ifdef SKEPU_MPI_STARPU
#include <starpu.h>
#include <skepu>
#endif

namespace soffa {
		class profiler
		{
		private:
				std::chrono::high_resolution_clock::time_point _start;

				std::string _filename;
				std::vector<std::string> _header;
				std::vector<std::string> _args;
				std::string _description;
		public:
				inline profiler(std::string filename = "",
												std::vector<std::string> header = {},
												std::vector<std::string> args = {},
												std::string description = "") :
						_filename(filename),
						_header(header),
						_args(args),
						_description(description)
						{
								using namespace std::chrono;
#ifdef SKEPU_MPI_STARPU
								if (skepu::cluster::mpi_rank() != 0) {
										_filename = "";
										_description = "";
								}
								starpu_task_wait_for_all();
								//	MPI_Barrier(MPI_COMM_WORLD);
#endif
								_start = high_resolution_clock::now();
						}

				inline ~profiler() {
#ifdef SKEPU_MPI_STARPU
						starpu_task_wait_for_all();
						//MPI_Barrier(MPI_COMM_WORLD);
#endif
								using namespace std::chrono;
								auto diff = high_resolution_clock::now() - _start;
								auto t = duration_cast<duration<double>>(diff).count();


								if (_filename != "") {
										bool is_new_file;
										{
												std::ifstream f(_filename.c_str());
												is_new_file = !f.good();
										}
										std::ofstream os(_filename.c_str(), std::ios_base::app);
										if (is_new_file)
										{
												for(const auto & h : _header) {
														os << h << ", ";
												}
												os << 't' << std::endl;
										}
										for(const auto & arg : _args) {
												os << arg << ", ";
										}
										os << t << std::endl;
								}
								if (_description.size()) {
								std::cout << _description << " >>>\t";
										for (size_t i {}; i < _args.size(); ++i) {
												std::cout << _header[i] << ":" << _args[i] << '\t';
										}
										std::cout << "time(s): " << t << std::endl;
								}
				}

		};
}

#define SOFFA_BENCHMARK(...) soffa::profiler __soffa_profiler(__VA_ARGS__)
