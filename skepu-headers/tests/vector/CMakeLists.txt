add_executable(vector vector.cpp)
target_link_libraries(vector PRIVATE SkePU catch2_main)

add_test(NAME vector_test
	COMMAND vector)

if(SKEPU_HEADERS_MPI)
	add_executable(starpu_vector starpu_vector.cpp)
	target_link_libraries(starpu_vector
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

	add_test(NAME starpu_vector_seq
		COMMAND starpu_vector)
	add_mpi_test(starpu_vector_par starpu_vector)
endif()
