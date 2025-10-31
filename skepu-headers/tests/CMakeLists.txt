find_package(Catch2 2 REQUIRED)
add_subdirectory(catch2_main)

# Use add_mpi_test(<name> <command>) to get a parallel mpi test added.
# Will make sure that mpirun is called on the executable and that STARPU_NCPU
# is set to one to reduce execution time. (One thread OpenMP execution within
# SkePU StarPU MPI backend.
macro(add_mpi_test name command)
	add_test(NAME ${name}
		COMMAND ${MPIEXEC_EXECUTABLE} ./${command})
	set_tests_properties(${name} PROPERTIES ENVIRONMENT
		STARPU_NCPU=1)
endmacro()

####################
# Add tests below. #
####################
# Utilities
add_subdirectory(external)

# Containers
add_subdirectory(matrix)
add_subdirectory(vector)
add_subdirectory(tensor3)
add_subdirectory(tensor4)

# Skeletons
add_subdirectory(map)
add_subdirectory(reduce)
add_subdirectory(scan)
