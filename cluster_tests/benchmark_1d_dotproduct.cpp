#include "common.hpp"
#ifdef BENCHMARK_1D_DOTPRODUCT

namespace benchmark_1d_dotproduct
{
		template<typename T>
		T mult(T a, T b)
		{
				return a * b;
		}

		template<typename T>
		T add(T a, T b)
		{
				return a + b;
		}
		auto dotprod = skepu::MapReduce<2>(mult<size_t>, add<size_t>);
		float dotproduct(skepu::Vector<size_t> &a, skepu::Vector<size_t> &b);

		TEST_CASE("Benchmark 1D dotproduct")
		{
				std::vector<size_t> ns {1,2,4,8,16,32,64};//,128,256,512,1024,2048};
				size_t iterations = 1;
				for(auto n : ns)
				{
						size_t res {};
						{
								skepu::Vector<size_t> a(1024*n);
								skepu::Vector<size_t> b(1024*n);
								SOFFA_BENCHMARK("1d_dotproduct.csv", {"nodes", "N"}, \
																{std::to_string(skepu::cluster::mpi_size()), std::to_string(1024*n)}, \
										"dotprod");
								for (size_t i {}; i < iterations; ++i) {
										res = dotprod(a,b);
								}
						}
						size_t expected {};
						for (size_t i {}; i < 1024*n; ++i) {
								expected += i*i;
						}

						REQUIRE(res == expected);
				}
		}
}

#endif
