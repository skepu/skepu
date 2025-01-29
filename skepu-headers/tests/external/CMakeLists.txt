add_executable(external external.cpp)
target_link_libraries(external
	PRIVATE
		catch2_main SkePU::SkePU)
add_test(external external)

if(SKEPU_HEADERS_MPI)
	add_executable(starpu_external starpu_external.cpp)
	target_link_libraries(starpu_external
		PRIVATE
			catch2_main
			SkePU::SkePU
			MPI::MPI_CXX
			PkgConfig::STARPU
			OpenMP::OpenMP_CXX)
	add_test(NAME starpu_external_seq
		COMMAND starpu_external)
	add_mpi_test(starpu_external_par starpu_external)
endif()
