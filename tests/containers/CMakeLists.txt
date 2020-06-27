skepu_add_executable(lifecycle_test SKEPUSRC lifecycle.cpp)
target_link_libraries(lifecycle_test PRIVATE catch2_main)
add_test(lifecycle lifecycle_test)


