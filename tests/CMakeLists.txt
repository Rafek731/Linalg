cmake_minimum_required(VERSION 3.14)
project(Linalg CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Fetch Google Test directly from GitHub
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
# For Windows MSVC compatibility:
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(matrix_tests test_matrix.cpp)

target_include_directories(matrix_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link Google Test to your executable
target_link_libraries(matrix_tests PRIVATE GTest::gtest_main)

# 3. Enable CTest discovery (Allows you to run tests from IDEs easily)
include(GoogleTest)
gtest_discover_tests(matrix_tests)