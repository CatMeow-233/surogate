# The host loops of the cpu_training step (fork ledgers T14, T15): fork/native/src/cpu_step_kernels.cpp. Its
# bit-for-bit contract with the upstream loops (surogate_fork/cpu_step_kernels.h) needs two flags on that file
# only: -fno-math-errno, so std::sqrt vectorizes, and -ffp-contract=off, so the AVX2 and x86-64-v4 clones do not
# fuse multiply-adds. surogate_fork_configure_common() (surogate-common) and the CPU contract test
# (openmp-contract) both call this, so both compile the same source the same way.
get_filename_component(SUROGATE_FORK_CPU_STEP_KERNELS_SRC "${CMAKE_CURRENT_LIST_DIR}/../src/cpu_step_kernels.cpp" ABSOLUTE)

# Call from the directory that defines `target`: source properties are directory-scoped.
function(surogate_fork_add_cpu_step_kernels target)
    target_sources(${target} PRIVATE "${SUROGATE_FORK_CPU_STEP_KERNELS_SRC}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties("${SUROGATE_FORK_CPU_STEP_KERNELS_SRC}" PROPERTIES
            COMPILE_OPTIONS "-fno-math-errno;-ffp-contract=off")
    else()
        message(FATAL_ERROR "cpu_step_kernels.cpp needs GCC or Clang flags (-fno-math-errno -ffp-contract=off)")
    endif()
endfunction()
