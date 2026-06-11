// Kernel侧核函数实现
#include "kernel_operator.h"

#include "addcmul_tiling.h"
#include "tiling_key_addcmul.h"

using namespace AscendC;

template <typename T>
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y,
                                const AddcmulTilingData &tiling) {
        this->tiling = tiling;
        if (tiling.totalLength == 0) return;

        // Calculate this core's block index and params
        blockIdx = GetBlockIdx();
        blockNum = (tiling.totalLength + tiling.blockFormer - 1) / tiling.blockFormer;
        blockLen = (blockIdx == blockNum - 1 && tiling.blockTail > 0)
                    ? tiling.blockTail : tiling.blockFormer;
        blockOffset = blockIdx * tiling.blockFormer;

        // Set GlobalTensor buffers
        // For broadcast, wrap offsets within each input's actual element count.
        // Set buffer at GM base (position 0) and compute per-tile offsets in ProcessTile.
        if (tiling.needBroadcast) {
            // Broadcast inputs: start at GM base, wrap at their own total
            x1Gm.SetGlobalBuffer((__gm__ T *)x1, tiling.x1Total);
            x2Gm.SetGlobalBuffer((__gm__ T *)x2, tiling.x2Total);
            inputDataGm.SetGlobalBuffer((__gm__ T *)input_data, tiling.inTotal);
        } else {
            // Linear mode: all inputs have same size, use per-block offset
            x1Gm.SetGlobalBuffer((__gm__ T *)x1 + blockOffset, blockLen);
            x2Gm.SetGlobalBuffer((__gm__ T *)x2 + blockOffset, blockLen);
            inputDataGm.SetGlobalBuffer((__gm__ T *)input_data + blockOffset, blockLen);
        }
        yGm.SetGlobalBuffer((__gm__ T *)y + blockOffset, blockLen);
        valueGm.SetGlobalBuffer((__gm__ T *)value, 1);

        // Initialize pipe and allocate UB buffers
        // bufX1 and bufX2 are TBuf, not LocalTensor, for InitBuffer compatibility
        uint32_t ubBytes = tiling.ubFormer * sizeof(T);
        if (ubBytes > 0) {
            pipe.InitBuffer(bufX1, ubBytes);
            pipe.InitBuffer(bufX2, ubBytes);
            // Buffers for float16 upcast path (fp32 computation buffers)
            if constexpr (std::is_same_v<T, half>) {
                pipe.InitBuffer(bufFpX1, tiling.ubFormer * sizeof(float));
                pipe.InitBuffer(bufFpX2, tiling.ubFormer * sizeof(float));
                pipe.InitBuffer(bufFpIn, tiling.ubFormer * sizeof(float));
            // Buffers for int8 cast path (half computation buffers)
            } else if constexpr (std::is_same_v<T, int8_t>) {
                pipe.InitBuffer(bufHalfX1, tiling.ubFormer * sizeof(half));
                pipe.InitBuffer(bufHalfX2, tiling.ubFormer * sizeof(half));
                pipe.InitBuffer(bufHalfIn, tiling.ubFormer * sizeof(half));
            }
        }
        // Value buffer (single element)
        pipe.InitBuffer(bufValue, 32);
    }

    __aicore__ inline void Process() {
        if (tiling.totalLength == 0) return;

        // Determine this core's loop parameters
        bool isTailBlock = (blockIdx == blockNum - 1);
        uint32_t loopCount = isTailBlock ? tiling.tailLoopNum : tiling.loopNum;
        uint32_t lastTileLen = isTailBlock ? tiling.tailUbTail : tiling.ubTail;
        uint32_t offset = 0;

        // Pre-copy value scalar from GM to UB (single element, shared across all tiles)
        {
            DataCopyExtParams valParams{1u, static_cast<uint32_t>(sizeof(T)), 0u, 0u, 0u};
            DataCopyPadExtParams<T> valPadParams{false, 0u, 0u, T(0)};
            LocalTensor<T> valLocal = bufValue.Get<T>();
            DataCopyPad(valLocal, valueGm, valParams, valPadParams);
            PipeBarrier<PIPE_ALL>();
        }

        T scalarValue = bufValue.Get<T>().GetValue(0);

        // Main tile loop
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? lastTileLen : tiling.ubFormer;
            if (curLen == 0) curLen = tiling.ubFormer;
            ProcessTile(offset, curLen, scalarValue);
            offset += curLen;
        }
    }

private:
    // --- Per-input GM read position (handles broadcast wrapping) ---
    __aicore__ inline uint32_t GetReadPos(uint32_t tileOffset, uint32_t inputTotal) const {
        if (tiling.needBroadcast && inputTotal < tiling.totalLength) {
            uint32_t outputPos = blockOffset + tileOffset;
            return outputPos % inputTotal;
        }
        return tileOffset;  // linear: GM buffer already offset by blockOffset
    }

    // --- Per-tile compute: dispatch by dtype ---
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t curLen, T scalarValue) {
        DataCopyExtParams params{1u, static_cast<uint32_t>(curLen * sizeof(T)), 0u, 0u, 0u};
        DataCopyPadExtParams<T> padParams{false, 0u, 0u, T(0)};

        // Compute per-input read positions (handles broadcast wrapping)
        uint32_t x1Pos = GetReadPos(offset, tiling.x1Total);
        uint32_t x2Pos = GetReadPos(offset, tiling.x2Total);
        uint32_t inPos = GetReadPos(offset, tiling.inTotal);
        (void)inPos;  // used below

        // ----- float16: upcast to float32 for numerical stability -----
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<half> x1Half = bufX1.Get<half>();
            LocalTensor<half> x2Half = bufX2.Get<half>();
            LocalTensor<float> fpX1 = bufFpX1.Get<float>();
            LocalTensor<float> fpX2 = bufFpX2.Get<float>();
            LocalTensor<float> fpIn = bufFpIn.Get<float>();
            LocalTensor<half> outHalf = bufX1.Get<half>();  // reuse bufX1 for output

            // Phase 1: x1 -> half -> cast to float
            DataCopyPad(x1Half, x1Gm[x1Pos], params, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast<float>(fpX1, x1Half, RoundMode::CAST_NONE, curLen);

            // Phase 2: x2 -> half -> cast to float
            DataCopyPad(x2Half, x2Gm[x2Pos], params, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast<float>(fpX2, x2Half, RoundMode::CAST_NONE, curLen);

            // Phase 3-4: product = x1 * x2 (float); scaled = product * value (float)
            Mul<float>(fpX1, fpX1, fpX2, curLen);
            Muls<float>(fpX1, fpX1, static_cast<float>(scalarValue), curLen);

            // Phase 5: input_data -> half -> cast to float
            DataCopyPad(x2Half, inputDataGm[inPos], params, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast<float>(fpIn, x2Half, RoundMode::CAST_NONE, curLen);

            // Phase 6: result = input_data + scaled (float)
            Add<float>(fpIn, fpIn, fpX1, curLen);

            // Phase 7: cast result back to half and copy out
            Cast<half>(outHalf, fpIn, RoundMode::CAST_ROUND, curLen);
            PipeBarrier<PIPE_V>();
            DataCopyExtParams outParams{1u, static_cast<uint32_t>(curLen * sizeof(half)), 0u, 0u, 0u};
            DataCopyPad(yGm[offset], outHalf, outParams);

        // ----- int8: cast to half for computation (Mul/Add don't support int8_t) -----
        } else if constexpr (std::is_same_v<T, int8_t>) {
            LocalTensor<int8_t> x1Int8 = bufX1.Get<int8_t>();
            LocalTensor<int8_t> x2Int8 = bufX2.Get<int8_t>();
            LocalTensor<half> hX1 = bufHalfX1.Get<half>();
            LocalTensor<half> hX2 = bufHalfX2.Get<half>();
            LocalTensor<half> hIn = bufHalfIn.Get<half>();
            LocalTensor<int8_t> outInt8 = bufX1.Get<int8_t>();  // reuse bufX1 for output

            // Phase 1: x1 -> int8 -> cast to half
            DataCopyPad(x1Int8, x1Gm[x1Pos], params, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast<half>(hX1, x1Int8, RoundMode::CAST_NONE, curLen);

            // Phase 2: x2 -> int8 -> cast to half
            DataCopyPad(x2Int8, x2Gm[x2Pos], params, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast<half>(hX2, x2Int8, RoundMode::CAST_NONE, curLen);

            // Phase 3-4: product = x1 * x2 (half); scaled = product * value (half)
            Mul<half>(hX1, hX1, hX2, curLen);
            Muls<half>(hX1, hX1, static_cast<half>(scalarValue), curLen);

            // Phase 5: input_data -> int8 -> cast to half
            DataCopyPad(x2Int8, inputDataGm[inPos], params, padParams);
            PipeBarrier<PIPE_ALL>();
            Cast<half>(hIn, x2Int8, RoundMode::CAST_NONE, curLen);

            // Phase 6: result = input_data + scaled (half)
            Add<half>(hIn, hIn, hX1, curLen);

            // Phase 7: cast result back to int8 and copy out
            Cast<int8_t>(outInt8, hIn, RoundMode::CAST_RINT, curLen);
            PipeBarrier<PIPE_V>();
            DataCopyExtParams outParams{1u, static_cast<uint32_t>(curLen * sizeof(int8_t)), 0u, 0u, 0u};
            DataCopyPad(yGm[offset], outInt8, outParams);

        // ----- float / int32: direct compute -----
        } else {
            LocalTensor<T> x1Local = bufX1.Get<T>();
            LocalTensor<T> x2Local = bufX2.Get<T>();

            // Step 1: CopyIn x1 and x2 concurrently
            DataCopyPad(x1Local, x1Gm[x1Pos], params, padParams);
            DataCopyPad(x2Local, x2Gm[x2Pos], params, padParams);
            PipeBarrier<PIPE_ALL>();

            // Step 2: product = x1 * x2 (x1Local = product)
            Mul<T>(x1Local, x1Local, x2Local, curLen);

            // Step 3: scaled = product * value (x1Local = scaled)
            Muls<T>(x1Local, x1Local, scalarValue, curLen);

            // Step 4: CopyIn input_data (reuse x2Local)
            DataCopyPad(x2Local, inputDataGm[inPos], params, padParams);
            PipeBarrier<PIPE_ALL>();

            // Step 5: result = input_data + scaled
            Add<T>(x2Local, x2Local, x1Local, curLen);
            PipeBarrier<PIPE_V>();

            // Step 6: CopyOut result to GM
            DataCopyExtParams outParams{1u, static_cast<uint32_t>(curLen * sizeof(T)), 0u, 0u, 0u};
            DataCopyPad(yGm[offset], x2Local, outParams);
        }
    }

private:
    TPipe pipe;
    // Global tensors
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> inputDataGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<T> yGm;
    // UB buffers (TBuf for InitBuffer compatibility)
    TBuf<TPosition::VECCALC> bufX1;       // x1 -> product -> scaled (direct) or input (half/int8)
    TBuf<TPosition::VECCALC> bufX2;       // x2 -> input_data -> result (direct) or input (half/int8)
    TBuf<TPosition::VECCALC> bufValue;    // value scalar
    TBuf<TPosition::VECCALC> bufFpX1;     // float x1 for half upcast
    TBuf<TPosition::VECCALC> bufFpX2;     // float x2 for half upcast
    TBuf<TPosition::VECCALC> bufFpIn;     // float input_data for half upcast
    TBuf<TPosition::VECCALC> bufHalfX1;   // half x1 for int8 cast
    TBuf<TPosition::VECCALC> bufHalfX2;   // half x2 for int8 cast
    TBuf<TPosition::VECCALC> bufHalfIn;   // half input_data for int8 cast
    // Block parameters
    AddcmulTilingData tiling;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t blockLen;
    uint32_t blockOffset;
};

// Kernel entry point
template <typename DT_INPUT_DATA>
 __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddcmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddcmulTilingData, tiling_data, tiling);
    KernelAddcmul<DT_INPUT_DATA> op;
    op.Init(input_data, x1, x2, value, y, tiling_data);
    op.Process();
}
