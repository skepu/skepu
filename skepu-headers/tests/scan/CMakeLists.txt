if(SKEPU_HEADERS_MPI)
	add_executable(starpu_scan starpu_scan.cpp)
	target_link_libraries(starpu_scan
		PRIVATE
			catch2_main
			SkePU::SkePU
			MPI::MPI_CXX
			PkgConfig::STARPU
			OpenMP::OpenMP_CXX)
	add_test(NAME starpu_scan_seq
		COMMAND starpu_scan)
	add_mpi_test(starpu_scan_par starpu_scan)
endif()
