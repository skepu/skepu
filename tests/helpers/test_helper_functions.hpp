#ifndef SKEPU_TEST_HELPER_FUNCTIONS
#define SKEPU_TEST_HELPER_FUNCTIONS

#include <cmath>
#define PRECISION 1E-3
template <typename T>
void is_similar(T a, T b)
{
	CHECK((
		(std::isnan(a) && std::isnan(b)) ||
		(std::isinf(a) && std::isinf(b)) ||
		(a == Approx(b).epsilon(PRECISION)))
	);
}

#endif // SKEPU_TEST_HELPER_FUNCTIONS