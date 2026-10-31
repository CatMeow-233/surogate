# Fork-owned native wiring. csrc/CMakeLists.txt includes this file once, from
# the fork's two-line hook after the surogate-common link libraries; every other
# fork native change is made here or under fork/native, never in csrc/.
get_filename_component(SUROGATE_FORK_NATIVE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
include("${CMAKE_CURRENT_LIST_DIR}/training_openmp.cmake")

# The fork include root stays PRIVATE so that consumers of surogate-common do
# not pick it up; only targets that compile fork code get it. Fork sources that
# upstream code calls (T3's expert weight gradients) are compiled into the
# library, so the fork tests link against the code production runs.
function(surogate_fork_configure_common target)
    surogate_enable_training_openmp(${target})
    target_include_directories(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
    target_sources(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/src/moe_weight_grad.cpp")
endfunction()

# The test targets are defined after the hook, so they are attached once the
# csrc directory has been processed (cmake_language(DEFER) needs CMake 3.19).
# The fork's test cases live in fork/native/tests and join the upstream test
# executables, so ctest and --list-tests see the same names as before.
function(_surogate_fork_attach_tests)
    if(TARGET integration-tests)
        target_include_directories(integration-tests PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
        target_sources(integration-tests PRIVATE
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-moe-grouped-gemm.cu"
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-lora-fused-dx.cu"
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-fp8-quant.cpp"
        )
    endif()
    if(TARGET dsl-run-state-tests)
        target_include_directories(dsl-run-state-tests PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
        target_sources(dsl-run-state-tests PRIVATE
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-dispatch-pp-host-copy.cpp"
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-tied-lm-head.cpp"
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-doc-mask-staging.cpp"
        )
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL _surogate_fork_attach_tests)

# The training extension module (_surogate, PYTHON_BINDING builds) is defined after the hook too.
# T17: py_train.cpp's train_step_graphed includes surogate_fork/doc_mask_staging.h, a header-only
# file, so the module needs only the fork include root (PRIVATE, like surogate-common's).
function(_surogate_fork_attach_binding)
    if(TARGET _surogate)
        target_include_directories(_surogate PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL _surogate_fork_attach_binding)
