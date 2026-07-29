/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl_env.h"

namespace aclco::test
{

/**
 * Simple RAII device buffer for host-side tests.
 * Uses aclrtMalloc/aclrtFree and supports (async) H2D/D2H copies.
 */
template <typename T>
class DeviceBuffer
{
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST)
  {
    Resize(count, policy);
  }

  ~DeviceBuffer()
  {
    Reset();
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
  {
    MoveFrom(other);
  }
  
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
  {
    if (this != &other) {
      Reset();
      MoveFrom(other);
    }
    return *this;
  }

  void Resize(std::size_t count, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST)
  {
    Reset();
    count_ = count;
    if (count_ == 0) {
      return;
    }
    void* p = nullptr;
    uint32_t ret = aclrtMalloc(&p, Bytes(), policy);
    CheckAcl(ret, "aclrtMalloc");
    ptr_ = static_cast<T*>(p);
  }

  void Reset()
  {
    if (ptr_) {
      (void)aclrtFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* Data() noexcept
  {
    return ptr_;
  }
  
  T const* Data() const noexcept
  {
    return ptr_;
  }
  
  std::size_t Size() const noexcept
  {
    return count_;
  }
  
  std::size_t Bytes() const noexcept
  {
    return count_ * sizeof(T);
  }

  void MemsetZero(aclrtStream stream = nullptr)
  {
    if (!ptr_ || count_ == 0) {
      return;
    }
    CheckAcl(aclrtMemsetAsync(ptr_, Bytes(), 0, Bytes(), stream), "aclrtMemsetAsync");
  }

  void CopyFromHostAsync(T const* host, std::size_t count, aclrtStream stream)
  {
    if (count != count_) {
      throw std::runtime_error("DeviceBuffer::CopyFromHostAsync count mismatch");
    }
    if (count_ == 0) {
      return;
    }
    CheckAcl(aclrtMemcpyAsync(ptr_, Bytes(), host, Bytes(), ACL_MEMCPY_HOST_TO_DEVICE, stream),
              "aclrtMemcpyAsync H2D");
  }

  void CopyToHostAsync(T* host, std::size_t count, aclrtStream stream) const
  {
    if (count != count_) {
      throw std::runtime_error("DeviceBuffer::CopyToHostAsync count mismatch");
    }
    if (count_ == 0) {
      return;
    }
    CheckAcl(aclrtMemcpyAsync(host, Bytes(), ptr_, Bytes(), ACL_MEMCPY_DEVICE_TO_HOST, stream),
              "aclrtMemcpyAsync D2H");
  }

  std::vector<T> CopyToHost(aclrtStream stream) const
  {
    std::vector<T> host(count_);
    if (count_ == 0) {
      return host;
    }
    CopyToHostAsync(host.data(), count_, stream);
    Sync(stream);
    return host;
  }

 private:
  void MoveFrom(DeviceBuffer& other) noexcept
  {
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  T* ptr_{nullptr};
  std::size_t count_{0};
};

} // namespace aclco::test