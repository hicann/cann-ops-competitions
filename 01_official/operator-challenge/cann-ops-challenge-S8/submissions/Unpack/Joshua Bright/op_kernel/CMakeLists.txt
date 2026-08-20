# set custom compile options
add_ops_compile_options(ALL OPTIONS -O3)
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()
add_kernels_compile()
