cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    bench_addrelu.asc
)

add_executable(demo_sweep
    bench_sweep.asc
)

foreach(tgt demo demo_sweep)
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    )
endforeach()
