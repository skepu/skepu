#include "common.hpp"
#ifdef TEST_REDUCE

namespace test_reduce
{
	size_t seq_init_impl(skepu::Index2D index) {
		return index.col;
	}
	auto seq_init = skepu::Map<0>(seq_init_impl);


	size_t sum_impl(size_t x, size_t y) {
		return x + y;
	}
	auto sum_reduce = skepu::Reduce(sum_impl);

	TEST_CASE("Test Reduce") {

		FOR_N {
			skepu::Vector<size_t> res(n);
			skepu::Matrix<size_t> m({n,n});
			seq_init(m);

			SECTION("rowwise reduction") {
				sum_reduce.setReduceMode(skepu::ReduceMode::RowWise);
				sum_reduce(res, m);
				for(size_t i {}; i < n; ++i) {
					CHECK ( res[i] == (n*(n-1))/2);
				}
			}

			SECTION("colwise reduction") {
				sum_reduce.setReduceMode(skepu::ReduceMode::ColWise);
				sum_reduce(res, m);
				for(size_t i {}; i < n; ++i) {
					CHECK ( res[i] == n*i );
				}
			}

			SECTION("elwise reduction") {
					auto rank = skepu::cluster::mpi_rank();
					auto r = sum_reduce(m);
					auto expected = (n*n*(n-1))/2;

					CHECK ( expected == r );
			}
		}
	}
}

#endif
