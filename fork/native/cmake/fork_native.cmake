# Fork-owned native wiring. csrc/CMakeLists.txt includes this file once, from
# the fork's two-line hook after the surogate-common link libraries; every other
# fork native change is made here or under fork/native, never in csrc/.
get_filename_component(SUROGATE_FORK_NATIVE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
include("${CMAKE_CURRENT_LIST_DIR}/training_openmp.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cpu_step_kernels.cmake")

# The fork include root stays PRIVATE so that consumers of surogate-common do
# not pick it up; only targets that compile fork code get it. Fork sources that
# upstream code calls (T3's expert weight gradients, T14/T15's host loops of the
# cpu_training step, T21's CPU AdamW checkpoint state, T24's NVFP4 Four Over Six
# kernels) are compiled into the library, so the fork tests link against the code
# production runs.
function(surogate_fork_configure_common target)
    surogate_enable_training_openmp(${target})
    target_include_directories(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
    target_sources(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/src/moe_weight_grad.cpp")
    target_sources(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/src/cpu_adamw_checkpoint.cpp")
    target_sources(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/src/nvfp4_4o6.cu")   # T24: the 4/6 kernels
    surogate_fork_add_cpu_step_kernels(${target})
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
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-nvfp4-4o6.cu"
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
# py_train.cpp's train_step_graphed includes two header-only fork files, surogate_fork/doc_mask_staging.h
# (T17) and surogate_fork/full_step_warmup.h (T20), so the module needs only the fork include root
# (PRIVATE, like surogate-common's).
function(_surogate_fork_attach_binding)
    if(TARGET _surogate)
        target_include_directories(_surogate PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL _surogate_fork_attach_binding)

# The serve engine's op library is defined after the hook too, inside if(SUROGATE_BUILD_SERVE), and
# exists only in a serve build. T16: the small-T decode units of the GQA shapes the fork registers in
# csrc/src/serve/ops/kernel/gqa_attention_geometry.cuh join sinfer_ops beside upstream's, and the
# fork include root gives its wrapper and prompt launcher surogate_fork/gqa_fork_shapes.h. Without
# them a serve build fails: the registry declares each unit extern (an undefined reference at link)
# and the two sources include the header.
function(_surogate_fork_attach_serve)
    if(TARGET sinfer_ops)
        target_include_directories(sinfer_ops PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
        target_sources(sinfer_ops PRIVATE
            "${SUROGATE_FORK_NATIVE_DIR}/serve/gqa_attention_decode_gqa128_32q8.cu"
            "${SUROGATE_FORK_NATIVE_DIR}/serve/gqa_attention_decode_gqa128_40q8.cu"
            "${SUROGATE_FORK_NATIVE_DIR}/serve/gqa_attention_decode_gqa128_32q32.cu"
        )
    endif()
    # T16's GPU cases (serve tests only). Upstream's test_gqa_attention.cpp reads the fork header
    # (its registration contract exempts the KV pair the fork withholds with the switch off), so
    # sinfer_gqa_attention_test gets the fork include directory first. sinfer_gqa_fork_shapes_test
    # compiles that harness with the three fork shapes' cases (tests/test-fork-gqa-shapes.cpp); it
    # copies sinfer_gqa_attention_test's include directories, options and libraries, lands beside
    # it and joins serve-tests. ctest runs it with SUROGATE_SERVE_GQA_FORK_SHAPES=0 and =1, and
    # upstream's own test with =1 (its unset run is upstream's sinfer_gqa_attention_test).
    if(TARGET sinfer_gqa_attention_test)
        target_include_directories(sinfer_gqa_attention_test PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
        add_executable(sinfer_gqa_fork_shapes_test EXCLUDE_FROM_ALL
            "${SUROGATE_FORK_NATIVE_DIR}/tests/test-fork-gqa-shapes.cpp")
        foreach(property INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES)
            get_target_property(value sinfer_gqa_attention_test ${property})
            if(value)
                set_property(TARGET sinfer_gqa_fork_shapes_test PROPERTY ${property} "${value}")
            endif()
        endforeach()
        get_target_property(test_dir sinfer_gqa_attention_test BINARY_DIR)
        set_target_properties(sinfer_gqa_fork_shapes_test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${test_dir}")
        if(TARGET serve-tests)
            add_dependencies(serve-tests sinfer_gqa_fork_shapes_test)
        endif()
        add_test(NAME sinfer_gqa_fork_shapes_test COMMAND sinfer_gqa_fork_shapes_test)
        add_test(NAME sinfer_gqa_fork_shapes_on_test COMMAND sinfer_gqa_fork_shapes_test)
        add_test(NAME sinfer_gqa_attention_fork_shapes_on_test COMMAND sinfer_gqa_attention_test)
        set_tests_properties(sinfer_gqa_fork_shapes_test sinfer_gqa_fork_shapes_on_test
            sinfer_gqa_attention_fork_shapes_on_test PROPERTIES SKIP_RETURN_CODE 77)
        set_tests_properties(sinfer_gqa_fork_shapes_test PROPERTIES
            ENVIRONMENT "SUROGATE_SERVE_GQA_FORK_SHAPES=0")
        set_tests_properties(sinfer_gqa_fork_shapes_on_test sinfer_gqa_attention_fork_shapes_on_test
            PROPERTIES ENVIRONMENT "SUROGATE_SERVE_GQA_FORK_SHAPES=1")
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL _surogate_fork_attach_serve)
