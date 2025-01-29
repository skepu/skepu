if(SKEPU_HEADERS_MPI)
	add_executable(starpu_tensor4 starpu_tensor4.cpp)
	target_link_libraries(starpu_tensor4
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)

	add_test(NAME starpu_tensor4_seq
		COMMAND starpu_tensor4)
	add_mpi_test(starpu_tensor4_par starpu_tensor4)
endif()
