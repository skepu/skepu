#include "common.hpp"
#ifdef BENCHMARK_REDUCE

namespace benchmark_reduce {
		size_t seq_init_impl(skepu::Index1D index) {
				return index.i;
		}
		auto seq_init = skepu::Map<0>(seq_init_impl);


		size_t sum_impl(size_t x, size_t y) {
				return x + y;	
		}
		auto sum_reduce = skepu::Reduce(sum_impl);

		TEST_CASE("Benchmark reduce on skepu::Vector") {
				std::vector<size_t> ns {1,8,64,512,4096,32768,262144};
			const size_t iterations = 32;//std::stoul(argv[2]);
			for(auto n : ns) {
					n *= 1024;
					size_t res {};
				  skepu::Vector<size_t> v(n);
				  seq_init(v);
					{
						SOFFA_BENCHMARK("reduce.csv", {"nodes", "N"}, \
														{std::to_string(skepu::cluster::mpi_size()), \
														 std::to_string(n)}, "reduce");
						for(size_t i {}; i < iterations; ++i) {
								res += sum_reduce(v);
						}
					}
					REQUIRE ( res == iterations*((n*(n-1))/2) );
			}
	}
}

#endif
