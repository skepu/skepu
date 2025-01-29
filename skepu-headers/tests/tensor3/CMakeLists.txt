if(SKEPU_HEADERS_MPI)
	add_executable(starpu_tensor3 starpu_tensor3.cpp)
	target_link_libraries(starpu_tensor3
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

	add_test(NAME starpu_tensor3_seq
		COMMAND starpu_tensor3)
	add_mpi_test(starpu_tensor3_par starpu_tensor3)
endif()
