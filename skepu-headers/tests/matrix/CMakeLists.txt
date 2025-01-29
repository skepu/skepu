if(SKEPU_HEADERS_MPI)
	add_executable(starpu_matrix starpu_matrix.cpp)
	target_link_libraries(starpu_matrix
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

	add_test(NAME starpu_matrix_seq
		COMMAND starpu_matrix)
	add_mpi_test(starpu_matrix_par starpu_matrix)
endif()
