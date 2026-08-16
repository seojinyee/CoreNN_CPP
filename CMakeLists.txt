cmake_minimum_required(VERSION 3.20)
# Define project name and declare it's a C++ project
project(CoreNN_CPP CXX)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)
# Compiler flags: Optimize for speed (-O3) and enable AVX2/FMA instructions for SIMD
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-tree-vectorize -mavx2 -mfma")

# Automatically fetch and integrate GoogleTest framework from GitHub
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Create the core deep learning static library
add_library(corenn_library
    src/storage.cpp
    src/tensor.cpp
    src/nn.cpp
    src/optim.cpp
)
# Expose headers in include/ directory
target_include_directories(corenn_library PUBLIC include)

# Create testing executable and link GoogleTest
enable_testing()
add_executable(corenn_test tests/test_corenn.cpp)
target_link_libraries(corenn_test PRIVATE corenn_library GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(corenn_test)

# Create the main executable to run the XOR training example
add_executable(train_benchmark train.cpp)
target_link_libraries(train_benchmark PRIVATE corenn_library)
