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
			std::vector<size_t> ns {1,2,4,8};
			for(auto n : ns) {
					n *= 1000000;
					size_t res {};
				  skepu::Vector<size_t> v(n);
				  seq_init(v);
					{
						SOFFA_BENCHMARK("reduce.csv", {"nodes", "N"}, \
														{std::to_string(skepu::cluster::mpi_size()), \
														 std::to_string(n)}, "reduce");
						res += sum_reduce(v);
					}
					REQUIRE(res == ((n*(n-1))/2));
			}
	}
}

#endif
