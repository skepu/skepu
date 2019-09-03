#include "common.hpp"
#ifdef TEST_REDUCE

namespace test_reduce
{
	size_t seq_init_impl(skepu2::Index2D index) {
		return index.col;
	}
	auto seq_init = skepu2::Map<0>(seq_init_impl);


	size_t sum_impl(size_t x, size_t y) {
		return x + y;
	}
	auto sum_reduce = skepu2::Reduce(sum_impl);

	TEST_CASE("Test Reduce") {

		FOR_N {
			skepu2::Vector<size_t> res(n);
			skepu2::Matrix<size_t> m({n,n});
			seq_init(m);


			SECTION("rowwise reduction") {
				sum_reduce.setReduceMode(skepu2::ReduceMode::RowWise);
				sum_reduce(res, m);
				for(size_t i {}; i < n; ++i) {
					CHECK ( res[i] == (n*(n-1))/2);
				}
			}

			SECTION("colwise reduction") {
				sum_reduce.setReduceMode(skepu2::ReduceMode::ColWise);
				sum_reduce(res, m);
				for(size_t i {}; i < n; ++i) {
					CHECK ( res[i] == n*i );
				}
			}

			SECTION("elwise reduction") {
					auto r = sum_reduce(m);
					auto expected = (n*(n-1))/2;
					expected *= n;

					CHECK ( expected == r );
			}
		}
	}
}

#endif
