#include "common.hpp"

#ifdef TEST_STARPU_MATRIX_CONTAINER

#ifdef SKEPU_MPI_STARPU
namespace test_starpu_matrix_container {
		TEST_CASE ( "basic starpu_matrix_container functionality" ) {
				for (size_t n : {33, 32, 17, 129}){
						skepu::cluster::starpu_matrix_container<int> a(n, n/2);
						CHECK(a.size() == n * (n/2));
						CHECK(a.height() == n);
						CHECK(a.width() == n/2);
						CHECK(starpu_matrix_get_nx(a.get_block(0,0)) != 0);
						for (size_t row {}; row < n; ++row) {
								for (size_t col {}; col < n/2; ++col) {
										a.set(row, col, row*100 + col);
								}
						}
						for (size_t row {}; row < n; ++row) {
								for (size_t col {}; col < n/2; ++col) {
										CHECK(a(row, col) == row*100 + col);
								}
						}
				}
		}
		TEST_CASE ( "'allgather' functionality" ) {
				for (size_t n : {33, 32, 17, 129}){
						skepu::cluster::starpu_matrix_container<int> a(n, n/2);
						// Set data
						for (size_t row {}; row < n; ++row) {
								for (size_t col {}; col < n/2; ++col) {
										a.set(row, col, row*100 + col);
								}
						}
						auto handle = a.allgather();

						starpu_data_acquire(handle, STARPU_R);
						int* data = (int*)starpu_matrix_get_local_ptr(handle);
						int nx = starpu_matrix_get_nx(handle);
						int ny = starpu_matrix_get_ny(handle);
						int ld = starpu_matrix_get_local_ld(handle);

						for (size_t row {}; row < ny; ++row) {
							for (size_t col {}; col < nx; ++col) {
								CHECK(data[col + row*ld] == a(row, col));
							}
						}
						starpu_data_release(handle);
				}
		}
}
#endif

#endif
