# Standalone host regression for the same OpenMP wiring as surogate-common.
# No CUDA compiler, CUDA headers, driver or GPU is needed for this contract.
cmake_minimum_required(VERSION 3.25)
project(SurogateTrainingOpenMPContract LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/training_openmp.cmake)
add_library(training-openmp-probe SHARED probe.cpp)
surogate_enable_training_openmp(training-openmp-probe)

add_executable(training-openmp-contract check.cpp)
target_link_libraries(training-openmp-contract PRIVATE training-openmp-probe)

enable_testing()
add_test(NAME training-openmp-contract COMMAND training-openmp-contract)
# An explicit user/runtime thread limit can prevent a second worker. Report that
# as skipped, not as evidence that multithreaded execution was exercised.
set_tests_properties(training-openmp-contract PROPERTIES SKIP_RETURN_CODE 77)
