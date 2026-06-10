#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ErfTilingData {
    uint32_t length = 0U;
    uint32_t blockLength = 0U;
    uint32_t blockDim = 1U;

    void set_totalLength(uint32_t value)
    {
        length = value;
    }

    uint32_t get_totalLength() const
    {
        return length;
    }

    void set_blockLength(uint32_t value)
    {
        blockLength = value;
    }

    uint32_t get_blockLength() const
    {
        return blockLength;
    }

    void set_blockDim(uint32_t value)
    {
        blockDim = value;
    }

    uint32_t get_blockDim() const
    {
        return blockDim;
    }

    std::size_t GetDataSize() const
    {
        return sizeof(ErfTilingData);
    }

    void SaveToBuffer(void *buffer, std::size_t capacity) const
    {
        if (buffer == nullptr || capacity < sizeof(ErfTilingData)) {
            return;
        }
        std::memcpy(buffer, this, sizeof(ErfTilingData));
    }
};

namespace optiling {
using ::ErfTilingData;
}  // namespace optiling
