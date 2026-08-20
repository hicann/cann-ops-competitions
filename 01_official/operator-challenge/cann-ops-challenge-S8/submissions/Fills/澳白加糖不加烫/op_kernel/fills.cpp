#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_BYTES = 128 * 1024;

enum FillsDType : uint32_t {
    FILL_DTYPE_FLOAT16 = 0,
    FILL_DTYPE_BF16 = 1,
    FILL_DTYPE_FLOAT = 2,
    FILL_DTYPE_INT32 = 3,
    FILL_DTYPE_INT16 = 4,
    FILL_DTYPE_UINT8 = 5,
    FILL_DTYPE_INT8 = 6,
};

template <typename T>
class KernelFillGm {
public:
    __aicore__ inline KernelFillGm() = default;

    __aicore__ inline void Init(GM_ADDR output, uint64_t totalSize, T value)
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint64_t blockSize = (totalSize + blockNum - 1) / blockNum;
        const uint64_t blockOffset = blockSize * blockIdx;

        if (blockOffset >= totalSize) {
            this->blockLength = 0;
        } else {
            const uint64_t remaining = totalSize - blockOffset;
            this->blockLength = remaining < blockSize ? remaining : blockSize;
        }

        this->fillValue = value;
        if (this->blockLength > 0) {
            this->outputGm.SetGlobalBuffer((__gm__ T*)output + blockOffset, this->blockLength);
        }
        uint64_t tileBytes = TILE_BYTES;
        const uint64_t blockBytes = this->blockLength * sizeof(T);
        if (blockBytes > 0 && blockBytes < tileBytes) {
            tileBytes = ((blockBytes + 31) / 32) * 32;
        }
        this->tileLength = tileBytes / sizeof(T);
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        this->pipe.InitBuffer(this->outBuf, this->tileLength * sizeof(T));
        this->FillByCopy(0, this->blockLength);
    }

private:
    __aicore__ inline void CopyAlignedOrOverlapTail(uint64_t offset, uint32_t len, const LocalTensor<T>& outLocal)
    {
        constexpr uint32_t BLOCK_BYTES = 32;
        constexpr uint32_t BLOCK_ELEMENTS = BLOCK_BYTES / sizeof(T);
        const uint32_t copyBytes = static_cast<uint32_t>(len * sizeof(T));
        if ((copyBytes % BLOCK_BYTES) == 0) {
            DataCopy<T>(this->outputGm[offset], outLocal, len);
            return;
        }

        const uint32_t prefixBytes = (copyBytes / BLOCK_BYTES) * BLOCK_BYTES;
        if (prefixBytes > 0) {
            DataCopy<T>(this->outputGm[offset], outLocal, static_cast<uint32_t>(prefixBytes / sizeof(T)));
        }

        if (offset + len >= BLOCK_ELEMENTS) {
            DataCopy<T>(this->outputGm[offset + len - BLOCK_ELEMENTS], outLocal, BLOCK_ELEMENTS);
            return;
        }

        DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
        DataCopyPad<T>(this->outputGm[offset], outLocal, copyParams);
    }

    __aicore__ inline void FillByCopy(uint64_t offset, uint64_t length)
    {
        if (length == 0) {
            return;
        }

        const uint32_t duplicateLength =
            static_cast<uint32_t>(length < this->tileLength ? length : this->tileLength);
        LocalTensor<T> outLocal = this->outBuf.template Get<T>();
        Duplicate<T>(outLocal, this->fillValue, duplicateLength);

        for (uint64_t pos = 0; pos < length; pos += this->tileLength) {
            const uint64_t remaining = length - pos;
            const uint32_t len = static_cast<uint32_t>(remaining < this->tileLength ? remaining : this->tileLength);
            this->CopyAlignedOrOverlapTail(offset + pos, len, outLocal);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> outBuf;
    GlobalTensor<T> outputGm;
    T fillValue;
    uint64_t blockLength = 0;
    uint32_t tileLength = 0;
};

class KernelFills8Bit {
public:
    __aicore__ inline KernelFills8Bit() = default;

    __aicore__ inline void Init(GM_ADDR output, uint64_t totalSize, uint32_t byteValue)
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint64_t blockSize = (totalSize + blockNum - 1) / blockNum;
        const uint64_t blockOffset = blockSize * blockIdx;

        if (blockOffset >= totalSize) {
            this->blockLength = 0;
        } else {
            const uint64_t remaining = totalSize - blockOffset;
            this->blockLength = remaining < blockSize ? remaining : blockSize;
        }

        const uint32_t bytePattern = byteValue & 0xFF;
        this->fillValue = bytePattern | (bytePattern << 8) | (bytePattern << 16) | (bytePattern << 24);
        if (this->blockLength > 0) {
            this->outputGm.SetGlobalBuffer((__gm__ uint8_t*)output + blockOffset, this->blockLength);
        }
        if (this->blockLength > 0 && this->blockLength < TILE_BYTES) {
            this->tileBytes = ((this->blockLength + 31) / 32) * 32;
        } else {
            this->tileBytes = TILE_BYTES;
        }
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }

        this->pipe.InitBuffer(this->outBuf, this->tileBytes);
        this->FillBytesByCopy(0, this->blockLength);
    }

private:
    __aicore__ inline void CopyAlignedOrOverlapTail(
        uint64_t offset, uint32_t len, const LocalTensor<uint8_t>& outLocal)
    {
        constexpr uint32_t BLOCK_BYTES = 32;
        if ((len % BLOCK_BYTES) == 0) {
            DataCopy<uint8_t>(this->outputGm[offset], outLocal, len);
            return;
        }

        const uint32_t prefixLen = (len / BLOCK_BYTES) * BLOCK_BYTES;
        if (prefixLen > 0) {
            DataCopy<uint8_t>(this->outputGm[offset], outLocal, prefixLen);
        }

        if (offset + len >= BLOCK_BYTES) {
            DataCopy<uint8_t>(this->outputGm[offset + len - BLOCK_BYTES], outLocal, BLOCK_BYTES);
            return;
        }

        DataCopyExtParams copyParams{1, len, 0, 0, 0};
        DataCopyPad<uint8_t>(this->outputGm[offset], outLocal, copyParams);
    }

    __aicore__ inline void FillBytesByCopy(uint64_t offset, uint64_t length)
    {
        if (length == 0) {
            return;
        }

        const uint64_t duplicateBytes = length < this->tileBytes ? length : this->tileBytes;
        const uint32_t packedDuplicateLength =
            static_cast<uint32_t>((duplicateBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        LocalTensor<uint32_t> outLocal = this->outBuf.Get<uint32_t>();
        Duplicate<uint32_t>(outLocal, this->fillValue, packedDuplicateLength);
        const LocalTensor<uint8_t> outBytes = outLocal.ReinterpretCast<uint8_t>();

        for (uint64_t pos = 0; pos < length; pos += this->tileBytes) {
            const uint64_t remaining = length - pos;
            const uint32_t len = static_cast<uint32_t>(remaining < this->tileBytes ? remaining : this->tileBytes);
            this->CopyAlignedOrOverlapTail(offset + pos, len, outBytes);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> outBuf;
    GlobalTensor<uint8_t> outputGm;
    uint32_t fillValue = 0;
    uint64_t blockLength = 0;
    uint32_t tileBytes = TILE_BYTES;
};

template <typename T>
__aicore__ inline void RunFillGm(GM_ADDR output, uint64_t size, T value)
{
    KernelFillGm<T> op;
    op.Init(output, size, value);
    op.Process();
}

__aicore__ inline void RunFills8Bit(GM_ADDR output, uint64_t size, uint32_t byteValue)
{
    KernelFills8Bit op;
    op.Init(output, size, byteValue);
    op.Process();
}

extern "C" __global__ __aicore__ void fills(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    (void)input;
    (void)workspace;

    switch (tiling_data.dtype) {
        case FILL_DTYPE_FLOAT16:
            RunFillGm<half>(output, tiling_data.size, static_cast<half>(tiling_data.value));
            break;
        case FILL_DTYPE_BF16:
            RunFillGm<uint16_t>(output, tiling_data.size, static_cast<uint16_t>(tiling_data.bf16_value));
            break;
        case FILL_DTYPE_FLOAT:
            RunFillGm<float>(output, tiling_data.size, tiling_data.value);
            break;
        case FILL_DTYPE_INT32:
            RunFillGm<int32_t>(output, tiling_data.size, tiling_data.int_value);
            break;
        case FILL_DTYPE_INT16:
            RunFillGm<int16_t>(output, tiling_data.size, static_cast<int16_t>(tiling_data.int_value));
            break;
        case FILL_DTYPE_UINT8:
            RunFills8Bit(output, tiling_data.size, tiling_data.byte_value);
            break;
        case FILL_DTYPE_INT8:
            RunFills8Bit(output, tiling_data.size, tiling_data.byte_value);
            break;
        default:
            break;
    }
}
