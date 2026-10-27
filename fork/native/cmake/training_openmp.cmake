# The CPU optimizer lives in surogate-common, independently of the serving
# encoder. Linking OpenMP only to the encoder leaves the training pragmas inert.
function(surogate_enable_training_openmp target)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    # The imported CXX target scopes its compile flags to CXX, so -fopenmp is
    # not passed directly to nvcc when the target also contains CUDA sources.
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
endfunction()
