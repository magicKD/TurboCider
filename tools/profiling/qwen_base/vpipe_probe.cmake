# Injected after project(vpipe), deferred until its targets exist. The
# reference checkout is read-only; every generated file stays in -B.
function(tc_add_vpipe_reference_probe)
  add_library(qwen_vpipe_reference SHARED "${TC_VPIPE_PROBE_SOURCE}")
  target_compile_features(qwen_vpipe_reference PRIVATE cxx_std_20)
  target_compile_options(qwen_vpipe_reference PRIVATE -O2 -Wall -Wextra -Werror)
  target_include_directories(qwen_vpipe_reference PRIVATE
    "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}")
  target_link_libraries(qwen_vpipe_reference PRIVATE vpipe_objects vpipe_link
    ${VPIPE_MC_KERNEL_OBJS})
endfunction()
cmake_language(DEFER CALL tc_add_vpipe_reference_probe)
