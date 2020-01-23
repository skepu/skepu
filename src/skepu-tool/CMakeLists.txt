set(LLVM_LINK_COMPONENTS Support)

add_clang_executable(skepu-tool
  skepu.cpp
  visitor.cpp
  data_structures.cpp
  code_gen.cpp
  mapreduce_cl.cpp
  mapreduce_cu.cpp
  map_cl.cpp
  map_cu.cpp
  scan_cl.cpp
  scan_cu.cpp
  reduce_cl.cpp
  reduce_cu.cpp
  mapoverlap_cl.cpp
  mapoverlap_cu.cpp
  mappairs_cl.cpp
  mappairs_cu.cpp
  call_cl.cpp
  call_cu.cpp)

clang_target_link_libraries(skepu-tool
	PRIVATE
		clangTooling
		clangBasic
		clangAST
		clangASTMatchers)

install(TARGETS skepu-tool
	EXPORT skepu-toolTargets
	RUNTIME
	DESTINATION bin)
install(EXPORT skepu-toolTargets
	DESTINATION lib/cmake/SkePU
	NAMESPACE SkePU::)
