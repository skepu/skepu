if(SKEPU_HEADERS_MPI)
	add_executable(starpu_map starpu_map.cpp)
	target_link_libraries(starpu_map
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_map_seq
		COMMAND starpu_map)
	add_mpi_test(starpu_map_par starpu_map)

	add_executable(starpu_mapreduce starpu_mapreduce.cpp)
	target_link_libraries(starpu_mapreduce
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_mapreduce_seq
		COMMAND starpu_mapreduce)
	add_mpi_test(starpu_mapreduce_par starpu_mapreduce)

	add_executable(starpu_mappairs starpu_mappairs.cpp)
	target_link_libraries(starpu_mappairs
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_mappairs_seq
		COMMAND starpu_mappairs)
	add_mpi_test(starpu_mappairs_par starpu_mappairs)

	add_executable(starpu_mappairsreduce starpu_mappairsreduce.cpp)
	target_link_libraries(starpu_mappairsreduce
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_mappairsreduce_seq
		COMMAND starpu_mappairsreduce)
	add_mpi_test(starpu_mappairsreduce_par starpu_mappairsreduce)

	add_executable(starpu_mapoverlap_1d starpu_mapoverlap_1d.cpp)
	target_link_libraries(starpu_mapoverlap_1d
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_mapoverlap_1d_seq
		COMMAND starpu_mapoverlap_1d)
	add_mpi_test(starpu_mapoverlap_1d_par starpu_mapoverlap_1d)

	add_executable(starpu_mapoverlap_2d starpu_mapoverlap_2d.cpp)
	target_link_libraries(starpu_mapoverlap_2d
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_mapoverlap_2d_seq
		COMMAND starpu_mapoverlap_2d)
	add_mpi_test(starpu_mapoverlap_2d_par starpu_mapoverlap_2d)

	add_executable(starpu_mapoverlap_3d starpu_mapoverlap_3d.cpp)
	target_link_libraries(starpu_mapoverlap_3d
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_mapoverlap_3d_seq
		COMMAND starpu_mapoverlap_3d)
	add_mpi_test(starpu_mapoverlap_3d_par starpu_mapoverlap_3d)

	add_executable(starpu_mapoverlap_4d starpu_mapoverlap_4d.cpp)
	target_link_libraries(starpu_mapoverlap_4d
		PRIVATE
			catch2_main SkePU::SkePU MPI::MPI_CXX PkgConfig::STARPU OpenMP::OpenMP_CXX)
	add_test(NAME starpu_mapoverlap_4d_seq
		COMMAND starpu_mapoverlap_4d)
	add_mpi_test(starpu_mapoverlap_4d_par starpu_mapoverlap_4d)

	if(SKEPU_HEADERS_CUDA)
		add_executable(starpu_map_cu starpu_map.cu)
		target_link_libraries(starpu_map_cu
			PRIVATE
				catch2_main
				SkePU::SkePU
				MPI::MPI_CXX
				PkgConfig::STARPU
				OpenMP::OpenMP_CXX)
		add_test(NAME starpu_map_seq_cu
			COMMAND starpu_map_cu)
		add_mpi_test(starpu_map_par_cu starpu_map_cu)

		add_executable(starpu_mappairs_cu starpu_map.cu)
		target_link_libraries(starpu_mappairs_cu
			PRIVATE
				catch2_main
				SkePU::SkePU
				MPI::MPI_CXX
				PkgConfig::STARPU
				OpenMP::OpenMP_CXX)
		add_test(NAME starpu_mappairs_seq_cu
			COMMAND starpu_mappairs_cu)
		add_mpi_test(starpu_mappairs_par_cu starpu_map_cu)
	endif()
endif()
