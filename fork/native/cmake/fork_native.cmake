# Fork-owned native wiring. csrc/CMakeLists.txt includes this file once, from
# the fork's two-line hook after the surogate-common link libraries; every other
# fork native change is made here or under fork/native, never in csrc/.
get_filename_component(SUROGATE_FORK_NATIVE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
include("${CMAKE_CURRENT_LIST_DIR}/training_openmp.cmake")

# The fork include root stays PRIVATE so that consumers of surogate-common do
# not pick it up; only targets that compile fork code get it.
function(surogate_fork_configure_common target)
    surogate_enable_training_openmp(${target})
    target_include_directories(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
endfunction()

# The test targets are defined after the hook, so they are attached once the
# csrc directory has been processed (cmake_language(DEFER) needs CMake 3.19).
function(_surogate_fork_attach_tests)
    if(TARGET integration-tests)
        target_include_directories(integration-tests PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL _surogate_fork_attach_tests)
