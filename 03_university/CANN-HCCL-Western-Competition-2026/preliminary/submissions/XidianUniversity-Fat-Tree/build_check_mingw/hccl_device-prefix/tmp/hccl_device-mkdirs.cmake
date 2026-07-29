# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "E:/HCCL/HCCLCOM/allreduce/op_kernel_aicpu")
  file(MAKE_DIRECTORY "E:/HCCL/HCCLCOM/allreduce/op_kernel_aicpu")
endif()
file(MAKE_DIRECTORY
  "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/device_build"
  "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/hccl_device-prefix"
  "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/hccl_device-prefix/tmp"
  "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/hccl_device-prefix/src/hccl_device-stamp"
  "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/hccl_device-prefix/src"
  "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/hccl_device-prefix/src/hccl_device-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/hccl_device-prefix/src/hccl_device-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "E:/HCCL/HCCLCOM/allreduce/build_check_mingw/hccl_device-prefix/src/hccl_device-stamp${cfgdir}") # cfgdir has leading slash
endif()
