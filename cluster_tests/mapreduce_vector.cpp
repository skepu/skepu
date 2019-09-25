#include "common.hpp"

#ifdef TEST_MAP_VECTOR

namespace test_map_vector
{
		size_t seq_init_impl(skepu::Index1D index) {
				return index.i;
		}
		auto seq_init = skepu::Map<0>(seq_init_impl);

		size_t add_one_impl(size_t x) {
					return x + 1;
		}
		auto add_one = skepu::Map<1>(add_one_impl);

		size_t add_impl(size_t x, size_t y) {
					return x + y;
		}
		auto add = skepu::Map<2>(add_impl);

		size_t add_k_impl(size_t x, size_t k) {
					return x + k;
		}
		auto add_k = skepu::Map<1>(add_k_impl);

		size_t add_container_impl(size_t x, const skepu::Vec<size_t> xs) {
					size_t tot = 0;
					for (size_t i = 0; i < xs.size; ++i){
							tot += xs[i];
					}
					return x + tot;
		}
		auto add_container = skepu::Map<1>(add_container_impl);


		TEST_CASE( "Map on skepu::Vector" ) {
			FOR_N {
				skepu::Vector<size_t> v(n);
				skepu::Vector<size_t> res(n);
				seq_init(v);
				SECTION( "indexed maps on vector" ) {
							CHECK( v[n/2] == n/2 );
							CHECK( v[0]   == 0 );
							CHECK( v[n-1]   == n-1 );
				}

				SECTION( "simple map" ) {
								add_one(res, v);

								CHECK( v[n/2] == n/2 );
								CHECK( v[0]   == 0 );
								CHECK( v[n-1]   == n-1 );

								CHECK( res[n/2] == 1 + n/2 );
								CHECK( res[0]   == 1 + 0 );
								CHECK( res[n-1]   == 1 + n-1 );

								for(size_t i {}; i < n; ++i) {
										CHECK( res[i] == i+1 );
								}
				}

				SECTION( "map with argument" ) {
								add_k(res, v, 5);

								CHECK( res[n/2] == 5 + n/2 );
								CHECK( res[0]   == 5 + 0 );
								CHECK( res[n-1]   == 5 + n-1 );
				}

				SECTION( "map with two elementwise" ) {
								add(res, v, v);

								CHECK( res[n/2]/2 ==  n/2 );
								CHECK( res[0]/2   ==  0 );
								CHECK( res[n-1]/2   ==  n-1 );
				}

				SECTION( "map with container argument" ) {
							add_container(res, v, v);

							size_t k = (n*(n-1))/2;
							CHECK( res[n/2] == k + n/2 );
							CHECK( res[0]   == k + 0 );
							CHECK( res[n-1]   == k + n-1 );

							// Change container argument to enforce new broadcast
							add_one(v,v);
							k += n;

							skepu::Vector<size_t> w(n);
							seq_init(w);
							add_container(res, w, v);

							CHECK( res[n/2] == k + n/2 );
							CHECK( res[0]   == k + 0 );
							CHECK( res[n-1]   == k + n-1 );
				}

				SECTION ( "map with iterator arguments misaligned" ) {
								add_one(res.begin(), res.end() - 1, v.begin() + 1);
								for(size_t i {}; i < n; ++i) {
									CHECK( v[i] == i );
								}
								for(size_t i {}; i < n - 1; ++i)
								{
									CHECK( res[i] == 2 + i);
								}
								CHECK( res[n-1] == 0 );
				}

				SECTION ( "map with iterator arguments somewhat aligned" ) {
								add_one(res.begin() + 1, res.end(), v.begin() + 1);
								CHECK( v[n/2] == n/2 );
								CHECK( v[0]   == 0 );
								CHECK( v[n-1]   == n-1 );

								CHECK( res[n/2] == 1 + n/2 );
								CHECK( res[0]   == 0 );
								CHECK( res[n-1]   == 1 + n-1 );
				 }

				} // FORN

				SECTION ( "completely mismatched vector alignment")  {
								skepu::Vector<size_t> a(1389); seq_init(a.begin(), a.end());
								skepu::Vector<size_t> b(2789); seq_init(b.begin(), b.end());
								skepu::Vector<size_t> res(1131);
							seq_init(res);

							add(res.begin() + 5, res.end() - 6, a.begin() + 37, b.begin() + 999);

								for(size_t i {}; i < res.size() - 5 - 6; ++i) {
									CHECK ( res[i + 5] == a[i + 37] + b[i + 999] );
								}
				}
		}
}

#endif
