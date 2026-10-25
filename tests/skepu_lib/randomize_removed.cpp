// Vector::randomize, Matrix::randomize and Matrix::randomizeReal are removed (skepu-lib/randomize.hpp has the random fills): a call is a static
// assertion that says so. This file must not compile (see randomize_removed in CMakeLists.txt).
#include <skepu>

int main()
{
#if defined(CASE_VECTOR)
	skepu::Vector<int> v(4);
	v.randomize(0, 9);
#elif defined(CASE_MATRIX)
	skepu::Matrix<int> m(2, 2);
	m.randomize(0, 9);
#elif defined(CASE_MATRIX_REAL)
	skepu::Matrix<float> m(2, 2);
	m.randomizeReal(0, 1);
#endif
	return 0;
}
