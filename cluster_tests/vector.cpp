#include "common.hpp"
#ifdef TEST_VECTOR

namespace test_vector
{
	size_t seq_init_impl(skepu::Index1D index) {
		return index.i;
	}
	auto seq_init = skepu::Map<0>(seq_init_impl);

	TEST_CASE("basic skepu::Vector<T> functionality") {
		FOR_N {
			skepu::Vector<size_t> v(n, 5); // Init to 5
			CHECK( v.size() == n );
			CHECK( v[0] == 5 );

			seq_init(v);
			//v.flush();

			for (size_t i {}; i < n; ++i) {
				CHECK ( v[i] == i );
			}
		}
	}
}

#endif
