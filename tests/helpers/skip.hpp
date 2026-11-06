// skip.hpp — for a test that finds out at run time that it cannot run (no working device, a device too small): says why and exits with the code ctest
// reports as skipped (SKIP_RETURN_CODE 77). What the host lacks at configure time goes through skepu_gate and REQUIRES.
#pragma once

#include <cstdlib>
#include <iostream>

#define SKEPU_TEST_SKIP(reason) \
	do { std::cout << "SKIPPED: " << reason << std::endl; std::exit(77); } while (0)
