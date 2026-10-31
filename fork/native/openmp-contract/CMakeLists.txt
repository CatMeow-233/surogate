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
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/cpu_step_kernels.cmake)
find_package(Threads REQUIRED)
# team.cpp compiles surogate_fork/omp_team.h with the same helper as surogate-common's cpu_adamw.cpp, and the
# library carries the production cpu_step_kernels.cpp with its own flags (ledgers T14, T15), at -O3 as in the
# trainer's Release build, so that the checks run the vectorized clones.
add_library(training-openmp-probe SHARED probe.cpp team.cpp)
surogate_enable_training_openmp(training-openmp-probe)
target_include_directories(training-openmp-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
surogate_fork_add_cpu_step_kernels(training-openmp-probe)
set_property(SOURCE "${SUROGATE_FORK_CPU_STEP_KERNELS_SRC}" APPEND PROPERTY COMPILE_OPTIONS -O3)

add_executable(training-openmp-contract check.cpp)
target_link_libraries(training-openmp-contract PRIVATE training-openmp-probe)

add_executable(training-openmp-team team_check.cpp)
target_link_libraries(training-openmp-team PRIVATE training-openmp-probe Threads::Threads)

add_executable(cpu-step-contract cpu_step_check.cpp)
target_include_directories(cpu-step-contract PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
# Its reference loops are compiled as surogate-common's Release build compiles cpu_adamw.cpp (-O3, the compiler's
# default contraction), so that a baseline ISA with FMA fuses them as it fuses upstream's (ledger T15).
set_source_files_properties(cpu_step_check.cpp PROPERTIES COMPILE_OPTIONS -O3)
target_link_libraries(cpu-step-contract PRIVATE training-openmp-probe Threads::Threads)

enable_testing()
add_test(NAME training-openmp-contract COMMAND training-openmp-contract)
# An explicit user/runtime thread limit can prevent a second worker. Report that
# as skipped, not as evidence that multithreaded execution was exercised.
set_tests_properties(training-openmp-contract PROPERTIES SKIP_RETURN_CODE 77)
# Team sizes under CPU pinning, the serial threshold and the ordered norm (surogate_fork/omp_team.h). Checks
# that need more CPUs than the process may use print "skipped" and the rest still run.
add_test(NAME training-openmp-team COMMAND training-openmp-team)
# The host loops of the cpu_training step against the upstream loops, bit for bit, and the host-gradient
# overwrite protocol against the upstream memset (ledgers T13-T15).
add_test(NAME cpu-step-contract COMMAND cpu-step-contract)
