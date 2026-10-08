add_library(catch2_main STATIC catch2_main.cpp)
target_include_directories(catch2_main PRIVATE
  ${CMAKE_SOURCE_DIR}/external/catch2
)
