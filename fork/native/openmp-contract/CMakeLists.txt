# Standalone host regression for the same OpenMP wiring as surogate-common.
# No CUDA compiler, CUDA headers, driver or GPU is needed for this contract.
cmake_minimum_required(VERSION 3.25)
project(SurogateTrainingOpenMPContract LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# The csrc/CMakeLists.txt hook itself must give surogate-common this wiring (configure-time check).
include(${CMAKE_CURRENT_SOURCE_DIR}/hook_contract.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/training_openmp.cmake)
find_package(Threads REQUIRED)
# team.cpp compiles surogate_fork/omp_team.h with the same helper as surogate-common's cpu_adamw.cpp.
add_library(training-openmp-probe SHARED probe.cpp team.cpp)
surogate_enable_training_openmp(training-openmp-probe)
target_include_directories(training-openmp-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(training-openmp-contract check.cpp)
target_link_libraries(training-openmp-contract PRIVATE training-openmp-probe)

add_executable(training-openmp-team team_check.cpp)
target_link_libraries(training-openmp-team PRIVATE training-openmp-probe Threads::Threads)

enable_testing()
add_test(NAME training-openmp-contract COMMAND training-openmp-contract)
# An explicit user/runtime thread limit can prevent a second worker. Report that
# as skipped, not as evidence that multithreaded execution was exercised.
set_tests_properties(training-openmp-contract PROPERTIES SKIP_RETURN_CODE 77)
# Team sizes under CPU pinning, the serial threshold and the ordered norm (surogate_fork/omp_team.h). Checks
# that need more CPUs than the process may use print "skipped" and the rest still run.
add_test(NAME training-openmp-team COMMAND training-openmp-team)
