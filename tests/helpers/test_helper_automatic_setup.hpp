#ifndef SKEPU_TEST_HELPER_AUTOMATIC_SETUP
#define SKEPU_TEST_HELPER_AUTOMATIC_SETUP

#ifdef SKEPU_OPENMP
#include <iostream>
inline bool automaticallySetCPUThreadsForTesting()
{
	std::cout << "OpenMP Detected, setting CPU threads to 8\n";
	auto spec = skepu::BackendSpec{"openmp"};
	spec.setCPUThreads(8);
	skepu::setGlobalBackendSpec(spec);
	return true;
}
inline bool automaticallySetCPUThreadsForTesting_dummy_variable = automaticallySetCPUThreadsForTesting();
#endif // SKEPU_OPENMP

#endif // SKEPU_TEST_HELPER_AUTOMATIC_SETUP