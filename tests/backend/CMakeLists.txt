skepu_add_executable(selection_test OpenMP SKEPUSRC selection.cpp)
target_link_libraries(selection_test PRIVATE catch2_main)
add_test(selection selection_test)


