# Configure-time contract for the fork's hook in csrc/CMakeLists.txt (ledger T5).
#
# Reads the hook lines from csrc/CMakeLists.txt (the include of fork_native.cmake and the
# surogate_fork_* calls), checks that they come after add_library(surogate-common), and evaluates
# them as written on a stand-in surogate-common target that is never built. The configure fails
# unless the stand-in then links OpenMP::OpenMP_CXX and has the fork include root. The ctest of
# this project shows that OpenMP::OpenMP_CXX runs a multi-threaded team, so together they cover
# csrc hook -> surogate_fork_configure_common() -> surogate_enable_training_openmp().
# Losing the call line, or the OpenMP or include call in fork_native.cmake, fails here;
# fork/dev check (the [[exact_diff]] entry in the ledger) also fails on the first.

get_filename_component(_hook_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(_hook_file "${_hook_root}/csrc/CMakeLists.txt")
set(_hook_doc "see fork/native/README.md (挂接约定) and the T5 entry in fork/docs/sync.md")
file(READ "${_hook_file}" _hook_text)

string(REGEX MATCHALL "[^\n]*(fork/native/cmake/fork_native\\.cmake|surogate_fork_[a-z_]+[ \t]*\\()[^\n]*"
       _hook_lines "${_hook_text}")
list(LENGTH _hook_lines _hook_count)
if(_hook_count EQUAL 0)
    message(FATAL_ERROR "fork hook contract: ${_hook_file} has no fork hook line "
                        "(include of fork_native.cmake, surogate_fork_configure_common); ${_hook_doc}")
endif()

string(REGEX MATCH "add_library\\([ \t\n]*surogate-common[ \t\n)]" _hook_target "${_hook_text}")
string(FIND "${_hook_text}" "${_hook_target}" _hook_target_at)
list(GET _hook_lines 0 _hook_first)
string(FIND "${_hook_text}" "${_hook_first}" _hook_first_at)
if(NOT _hook_target OR _hook_first_at LESS _hook_target_at)
    message(FATAL_ERROR "fork hook contract: the hook in ${_hook_file} must come after "
                        "add_library(surogate-common ...); ${_hook_doc}")
endif()

# The hook as written, with the csrc directory for the variables csrc/CMakeLists.txt sees there.
string(JOIN "\n" _hook_code ${_hook_lines})
foreach(_hook_var CMAKE_CURRENT_SOURCE_DIR CMAKE_CURRENT_LIST_DIR CMAKE_SOURCE_DIR PROJECT_SOURCE_DIR)
    string(REPLACE "\${${_hook_var}}" "${_hook_root}/csrc" _hook_code "${_hook_code}")
endforeach()

add_library(surogate-common STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_LIST_DIR}/probe.cpp")
cmake_language(EVAL CODE "${_hook_code}")

string(JOIN "\n    " _hook_shown ${_hook_lines})
get_target_property(_hook_libs surogate-common LINK_LIBRARIES)
if(NOT "OpenMP::OpenMP_CXX" IN_LIST _hook_libs)
    message(FATAL_ERROR "fork hook contract: after the hook lines of ${_hook_file}\n    ${_hook_shown}\n"
                        "surogate-common does not link OpenMP::OpenMP_CXX. The "
                        "surogate_fork_configure_common(surogate-common) call, or the "
                        "surogate_enable_training_openmp() call in fork/native/cmake/fork_native.cmake, is missing; "
                        "${_hook_doc}")
endif()

file(REAL_PATH "${_hook_root}/fork/native/include" _hook_want)
get_target_property(_hook_dirs surogate-common INCLUDE_DIRECTORIES)
set(_hook_found FALSE)
foreach(_hook_dir IN LISTS _hook_dirs)
    file(REAL_PATH "${_hook_dir}" _hook_real)
    if(_hook_real STREQUAL _hook_want)
        set(_hook_found TRUE)
    endif()
endforeach()
if(NOT _hook_found)
    message(FATAL_ERROR "fork hook contract: after the hook lines of ${_hook_file}\n    ${_hook_shown}\n"
                        "surogate-common does not have the include root ${_hook_want}. The "
                        "target_include_directories() call in surogate_fork_configure_common() is missing; ${_hook_doc}")
endif()
message(STATUS "fork hook contract: csrc/CMakeLists.txt gives surogate-common OpenMP::OpenMP_CXX and "
               "fork/native/include")
