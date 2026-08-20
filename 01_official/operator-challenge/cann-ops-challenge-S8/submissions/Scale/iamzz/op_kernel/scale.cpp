#ifndef __CCE_KT_TEST__
#include "kernel_operator.h"
#else
#include "tikcpp_test_mock.h"
#endif

#include <cstdint>
#include <type_traits>

using namespace AscendC;

#include "scale_broadcast_3_midaxis.h"

// Device implementation for:
//   output = input * scale + bias(optional)
// Shapes are already validated on the host side. The kernel receives flattened
// sizes plus flags that select one of three paths:
//   1. vectorScale: scale/bias line up elementwise in flattened order.
//   2. expandedBroadcast: duplicate scale/bias into UB vectors per suffix group.
//   3. scalar broadcast: apply one scalar scale/bias value to each suffix group.
template <typename T>
class KernelScale {
public:
    __aicore__ inline KernelScale() = default;

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR scale, GM_ADDR bias,
                                GM_ADDR output, uint32_t scaleElemCount,
                                uint32_t suffixElemCount, uint32_t baseElems,
                                uint32_t formerElems, uint32_t baseGroups,
                                uint32_t formerGroups, uint32_t tileSize,
                                uint32_t expandedGroupStrideValue,
                                uint8_t hasBiasValue,
                                uint8_t useExpandedBroadcastValue,
                                uint8_t useRelaxedExpandedFp16Value,
                                uint8_t useGroupAlignedExpandedSplitValue,
                                TPipe *pipeIn) {
        pipe = pipeIn;
        // Tiling values are copied to members so Process() and the tile helpers
        // can stay parameter-light. expandedGroupStride falls back to the real
        // suffix size when the host did not request padded expanded broadcast.
        scaleCount = scaleElemCount;
        suffixCount = suffixElemCount;
        expandedGroupStride = expandedGroupStrideValue == 0U ? suffixElemCount : expandedGroupStrideValue;
        hasBias = hasBiasValue != 0;
        vectorScale = suffixCount == 1U;
        expandedBroadcast = useExpandedBroadcastValue != 0;
        relaxedExpandedFp16 = useRelaxedExpandedFp16Value != 0;
        groupAlignedExpandedSplit = useGroupAlignedExpandedSplitValue != 0;
        const uint32_t blockIdx = GetBlockIdx();
        if (groupAlignedExpandedSplit) {
            // Expanded grouped mode assigns whole suffix groups to each core.
            // This keeps every tile group-aligned and avoids partial-group
            // scalar cleanup in the hot path.
            const uint32_t formerGroupOffset = blockIdx < formerGroups ? blockIdx : formerGroups;
            const uint32_t groupCount = baseGroups + (blockIdx < formerGroups ? 1U : 0U);
            const uint32_t startGroup = baseGroups * blockIdx + formerGroupOffset;
            elemCount = groupCount * suffixCount;
            startOffset = startGroup * suffixCount;
        } else {
            // Generic mode splits flattened elements across cores. Blocks may
            // start or end inside a suffix group, so Process() handles those
            // boundary fragments with scalar tiles.
            const uint32_t formerOffset = blockIdx < formerElems ? blockIdx : formerElems;
            elemCount = baseElems + (blockIdx < formerElems ? 1U : 0U);
            startOffset = baseElems * blockIdx + formerOffset;
        }
        if (elemCount == 0 || suffixCount == 0) {
            elemCount = 0;
            tileElems = 0;
            return;
        }

        inputGm.SetGlobalBuffer((__gm__ T *)input + startOffset, elemCount);
        outputGm.SetGlobalBuffer((__gm__ T *)output + startOffset, elemCount);
        scaleGm.SetGlobalBuffer((__gm__ T *)scale, scaleCount);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ T *)bias, scaleCount);
        }

        // DataCopyPad handles non-aligned tails; the extra 32 bytes protects UB
        // queue allocations for alignment padding. floatTileElems is rounded to
        // 8 floats so temporary FP32 buffers remain vector friendly.
        tileElems = tileSize == 0 ? 1U : tileSize;
        floatTileElems = ((tileElems + 7U) / 8U) * 8U;
        pipe->InitBuffer(inputQueue, 1, tileElems * sizeof(T) + 32);
        // scale/bias queues are needed only when the tile carries full vectors.
        // Scalar broadcast reads scale/bias directly from GM with GetValue().
        if (vectorScale || expandedBroadcast) {
            pipe->InitBuffer(scaleQueue, 1, tileElems * sizeof(T) + 32);
            if (hasBias) {
                pipe->InitBuffer(biasQueue, 1, tileElems * sizeof(T) + 32);
            }
        }
        pipe->InitBuffer(outputQueue, 1, tileElems * sizeof(T) + 32);
        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            const uint32_t tmpTensorCount = (vectorScale || expandedBroadcast) ? 3U : 2U;
            if constexpr (std::is_same<T, half>::value) {
                // Relaxed expanded FP16 intentionally computes in FP16 for the
                // inferred contest family to save UB and improve throughput.
                // Other FP16/BF16 paths use FP32 temporaries for accuracy.
                if (!(relaxedExpandedFp16 && expandedBroadcast)) {
                    pipe->InitBuffer(tmpBuf, floatTileElems * sizeof(float) * tmpTensorCount);
                }
            } else {
                pipe->InitBuffer(tmpBuf, floatTileElems * sizeof(float) * tmpTensorCount);
            }
        }
    }

    __aicore__ inline void Process() {
        if (elemCount == 0) {
            return;
        }

        if (vectorScale) {
            ProcessVectorScale();
            return;
        }
        if (expandedBroadcast) {
            if (groupAlignedExpandedSplit) {
                ProcessGroupAlignedExpandedBroadcastScale();
            } else {
                ProcessExpandedBroadcastScale();
            }
            return;
        }

        uint32_t localOffset = 0;
        while (localOffset < elemCount) {
            // Generic broadcast path. A scale/bias value applies to one suffix
            // group, so each loop segment stops at the next group boundary.
            const uint32_t globalOffset = startOffset + localOffset;
            const uint32_t scaleIdx = (globalOffset / suffixCount) % scaleCount;
            const uint32_t offsetInGroup = globalOffset % suffixCount;
            uint32_t segmentElems = suffixCount - offsetInGroup;
            const uint32_t remainingElems = elemCount - localOffset;
            if (segmentElems > remainingElems) {
                segmentElems = remainingElems;
            }

            const T scaleValue = scaleGm.GetValue(scaleIdx);
            const T biasValue = hasBias ? biasGm.GetValue(scaleIdx) : static_cast<T>(0);
            uint32_t segmentOffset = 0;
            while (segmentOffset < segmentElems) {
                const uint32_t remaining = segmentElems - segmentOffset;
                const uint32_t currentSize = tileElems < remaining ? tileElems : remaining;
                ProcessScalarTile(localOffset + segmentOffset, currentSize, scaleValue, biasValue);
                segmentOffset += currentSize;
            }
            localOffset += segmentElems;
        }
    }

private:
    __aicore__ inline float ScalarToFloat(T value) {
        // bfloat16_t does not reliably static_cast to float in device code, so
        // use the AscendC conversion helper.
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            return ToFloat(value);
        } else {
            return static_cast<float>(value);
        }
    }

    __aicore__ inline void ComputeElementwiseTile(LocalTensor<T> outputLocal,
                                                  LocalTensor<T> readyInput,
                                                  LocalTensor<T> readyScale,
                                                  LocalTensor<T> readyBias,
                                                  uint32_t count) {
        if constexpr (std::is_same<T, float>::value) {
            // FP32 can compute directly in the output tensor.
            Mul(outputLocal, readyInput, readyScale, count);
            if (hasBias) {
                Add(outputLocal, outputLocal, readyBias, count);
            }
        } else if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            if constexpr (std::is_same<T, half>::value) {
                if (relaxedExpandedFp16 && expandedBroadcast) {
                    // Fast FP16 expanded path: scale/bias are already materialized
                    // per element, so direct vector math avoids FP32 temp traffic.
                    Mul(outputLocal, readyInput, readyScale, count);
                    if (hasBias) {
                        Add(outputLocal, outputLocal, readyBias, count);
                    }
                    return;
                }
            }
            LocalTensor<float> inputFloat = tmpBuf.Get<float>();
            LocalTensor<float> scaleFloat = inputFloat[floatTileElems];
            LocalTensor<float> outputFloat = scaleFloat[floatTileElems];
            Cast(inputFloat, readyInput, RoundMode::CAST_NONE, count);
            Cast(scaleFloat, readyScale, RoundMode::CAST_NONE, count);
            if (relaxedExpandedFp16 && expandedBroadcast) {
                // Kept for completeness when a non-FP16 type reaches the relaxed
                // expanded path. Bias is used as the destination of MulAddDst,
                // then cast back to output dtype.
                if (hasBias) {
                    Cast(outputFloat, readyBias, RoundMode::CAST_NONE, count);
                    MulAddDst(outputFloat, inputFloat, scaleFloat, count);
                } else {
                    Mul(outputFloat, inputFloat, scaleFloat, count);
                }
                Cast(outputLocal, outputFloat, RoundMode::CAST_RINT, count);
                return;
            }

            Mul(outputFloat, inputFloat, scaleFloat, count);
            Cast(outputLocal, outputFloat, RoundMode::CAST_RINT, count);
            if (hasBias) {
                // Match the two-step precision behavior used by the contest
                // reference for reduced types: round multiplication to output
                // dtype, then add bias through FP32 and round again.
                Cast(inputFloat, outputLocal, RoundMode::CAST_NONE, count);
                Cast(scaleFloat, readyBias, RoundMode::CAST_NONE, count);
                Add(outputFloat, inputFloat, scaleFloat, count);
                Cast(outputLocal, outputFloat, RoundMode::CAST_RINT, count);
            }
        }
    }

    __aicore__ inline void ProcessVectorScale() {
        uint32_t localOffset = 0;
        while (localOffset < elemCount) {
            // suffixCount == 1: scale/bias index advances with every input
            // element and wraps at scaleCount.
            const uint32_t globalOffset = startOffset + localOffset;
            const uint32_t scaleOffset = globalOffset % scaleCount;
            uint32_t segmentElems = scaleCount - scaleOffset;
            const uint32_t remainingElems = elemCount - localOffset;
            if (segmentElems > remainingElems) {
                segmentElems = remainingElems;
            }

            uint32_t segmentOffset = 0;
            while (segmentOffset < segmentElems) {
                const uint32_t remaining = segmentElems - segmentOffset;
                const uint32_t currentSize = tileElems < remaining ? tileElems : remaining;
                ProcessVectorTile(localOffset + segmentOffset, scaleOffset + segmentOffset, currentSize);
                segmentOffset += currentSize;
            }
            localOffset += segmentElems;
        }
    }

    __aicore__ inline void ProcessExpandedBroadcastScale() {
        const uint32_t groupsPerTile = expandedGroupStride == 0U ? 0U : tileElems / expandedGroupStride;
        if (groupsPerTile <= 1U) {
            // Expanded mode is useful only when it can process multiple full
            // groups at once. Otherwise fall back to scalar broadcast.
            expandedBroadcast = false;
            Process();
            return;
        }

        uint32_t localOffset = 0;
        while (localOffset < elemCount) {
            const uint32_t globalOffset = startOffset + localOffset;
            const uint32_t offsetInGroup = globalOffset % suffixCount;
            const uint32_t remainingElems = elemCount - localOffset;

            if (offsetInGroup != 0U) {
                // Element-based core splitting can start inside a group. Handle
                // that head fragment with the scalar path, then resume expanded
                // processing on a group boundary.
                uint32_t headElems = suffixCount - offsetInGroup;
                if (headElems > remainingElems) {
                    headElems = remainingElems;
                }
                const uint32_t scaleIdx = (globalOffset / suffixCount) % scaleCount;
                const T scaleValue = scaleGm.GetValue(scaleIdx);
                const T biasValue = hasBias ? biasGm.GetValue(scaleIdx) : static_cast<T>(0);
                ProcessScalarTile(localOffset, headElems, scaleValue, biasValue);
                localOffset += headElems;
                continue;
            }

            if (remainingElems < suffixCount) {
                // Tail fragment smaller than a complete group cannot use the
                // multi-block expanded copy pattern.
                const uint32_t scaleIdx = (globalOffset / suffixCount) % scaleCount;
                const T scaleValue = scaleGm.GetValue(scaleIdx);
                const T biasValue = hasBias ? biasGm.GetValue(scaleIdx) : static_cast<T>(0);
                ProcessScalarTile(localOffset, remainingElems, scaleValue, biasValue);
                break;
            }

            uint32_t currentGroups = remainingElems / suffixCount;
            if (currentGroups > groupsPerTile) {
                currentGroups = groupsPerTile;
            }
            ProcessExpandedTile(localOffset, currentGroups);
            localOffset += currentGroups * suffixCount;
        }
    }

    __aicore__ inline void ProcessGroupAlignedExpandedBroadcastScale() {
        const uint32_t groupsPerTile = expandedGroupStride == 0U ? 0U : tileElems / expandedGroupStride;
        if (groupsPerTile <= 1U) {
            // Defensive fallback if host tiling selected grouped mode but UB
            // cannot hold multiple groups.
            groupAlignedExpandedSplit = false;
            expandedBroadcast = false;
            Process();
            return;
        }

        uint32_t localOffset = 0;
        while (localOffset < elemCount) {
            // This path receives only whole groups from Init(), so every tile can
            // go straight through ProcessExpandedTile without boundary cleanup.
            const uint32_t remainingGroups = (elemCount - localOffset) / suffixCount;
            uint32_t currentGroups = remainingGroups;
            if (currentGroups > groupsPerTile) {
                currentGroups = groupsPerTile;
            }
            ProcessExpandedTile(localOffset, currentGroups);
            localOffset += currentGroups * suffixCount;
        }
    }

    __aicore__ inline void ProcessScalarTile(uint32_t offset, uint32_t count,
                                             T scaleValue, T biasValue) {
        // Scalar tile copies only input/output through UB. scaleValue and
        // biasValue are already loaded from GM and used by vector Muls/Adds.
        LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
        inputQueue.EnQue<T>(inputLocal);
        LocalTensor<T> readyInput = inputQueue.DeQue<T>();

        LocalTensor<T> outputLocal = outputQueue.AllocTensor<T>();
        if constexpr (std::is_same<T, float>::value) {
            Muls(outputLocal, readyInput, scaleValue, count);
            if (hasBias) {
                Adds(outputLocal, outputLocal, biasValue, count);
            }
        } else if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
            // Reduced precision scalar broadcast uses FP32 temporaries for the
            // multiply and optional add, with a round after each operation.
            LocalTensor<float> inputFloat = tmpBuf.Get<float>();
            LocalTensor<float> outputFloat = inputFloat[floatTileElems];
            Cast(inputFloat, readyInput, RoundMode::CAST_NONE, count);
            Muls(outputFloat, inputFloat, ScalarToFloat(scaleValue), count);
            Cast(outputLocal, outputFloat, RoundMode::CAST_RINT, count);
            if (hasBias) {
                Cast(inputFloat, outputLocal, RoundMode::CAST_NONE, count);
                Adds(outputFloat, inputFloat, ScalarToFloat(biasValue), count);
                Cast(outputLocal, outputFloat, RoundMode::CAST_RINT, count);
            }
        }

        inputQueue.FreeTensor<T>(readyInput);
        outputQueue.EnQue<T>(outputLocal);
        LocalTensor<T> readyOutput = outputQueue.DeQue<T>();
        DataCopyPad(outputGm[offset], readyOutput, copyParams);
        outputQueue.FreeTensor<T>(readyOutput);
    }

    __aicore__ inline void ProcessVectorTile(uint32_t dataOffset, uint32_t scaleOffset,
                                             uint32_t count) {
        // Vector tile copies input, scale, and optional bias as same-length
        // vectors, then ComputeElementwiseTile performs elementwise affine math.
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};

        LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
        DataCopyPad(inputLocal, inputGm[dataOffset], copyParams, padParams);
        inputQueue.EnQue<T>(inputLocal);

        LocalTensor<T> scaleLocal = scaleQueue.AllocTensor<T>();
        DataCopyPad(scaleLocal, scaleGm[scaleOffset], copyParams, padParams);
        scaleQueue.EnQue<T>(scaleLocal);

        LocalTensor<T> biasLocal;
        if (hasBias) {
            biasLocal = biasQueue.AllocTensor<T>();
            DataCopyPad(biasLocal, biasGm[scaleOffset], copyParams, padParams);
            biasQueue.EnQue<T>(biasLocal);
        }

        LocalTensor<T> readyInput = inputQueue.DeQue<T>();
        LocalTensor<T> readyScale = scaleQueue.DeQue<T>();
        LocalTensor<T> readyBias;
        if (hasBias) {
            readyBias = biasQueue.DeQue<T>();
        }

        LocalTensor<T> outputLocal = outputQueue.AllocTensor<T>();
        ComputeElementwiseTile(outputLocal, readyInput, readyScale, readyBias, count);

        inputQueue.FreeTensor<T>(readyInput);
        scaleQueue.FreeTensor<T>(readyScale);
        if (hasBias) {
            biasQueue.FreeTensor<T>(readyBias);
        }
        outputQueue.EnQue<T>(outputLocal);
        LocalTensor<T> readyOutput = outputQueue.DeQue<T>();
        DataCopyPad(outputGm[dataOffset], readyOutput, copyParams);
        outputQueue.FreeTensor<T>(readyOutput);
    }

    __aicore__ inline void ProcessExpandedTile(uint32_t offset, uint32_t groupCount) {
        // Expanded tile processes groupCount suffix groups. Input is copied as
        // groupCount blocks of suffixCount elements, while scale/bias are
        // duplicated into padded per-group vectors of expandedGroupStride.
        const uint32_t paddedCount = groupCount * expandedGroupStride;
        DataCopyExtParams copyInParams{static_cast<uint16_t>(groupCount),
                                       static_cast<uint32_t>(suffixCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(groupCount),
                                        static_cast<uint32_t>(suffixCount * sizeof(T)), 0, 0, 0};

        LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
        DataCopyPad(inputLocal, inputGm[offset], copyInParams, padParams);
        inputQueue.EnQue<T>(inputLocal);

        const uint32_t startGroupIdx = ((startOffset + offset) / suffixCount) % scaleCount;
        LocalTensor<T> scaleLocal = scaleQueue.AllocTensor<T>();
        LocalTensor<T> biasLocal;
        if (hasBias) {
            biasLocal = biasQueue.AllocTensor<T>();
        }
        if constexpr (std::is_same<T, half>::value) {
            // Duplicate each scalar parameter across its padded group region.
            // The duplicated padding is computed too but not written back because
            // copyOutParams writes only suffixCount elements per group.
            for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
                uint32_t scaleIdx = startGroupIdx + groupIdx;
                if (scaleIdx >= scaleCount) {
                    scaleIdx %= scaleCount;
                }
                const uint32_t groupOffset = groupIdx * expandedGroupStride;
                Duplicate(scaleLocal[groupOffset], scaleGm.GetValue(scaleIdx), expandedGroupStride);
                if (hasBias) {
                    Duplicate(biasLocal[groupOffset], biasGm.GetValue(scaleIdx), expandedGroupStride);
                }
            }
        } else {
            // Same duplication logic for BF16/FP32. Kept as a constexpr branch so
            // dtype-specific tuning can be added without changing call sites.
            for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
                uint32_t scaleIdx = startGroupIdx + groupIdx;
                if (scaleIdx >= scaleCount) {
                    scaleIdx %= scaleCount;
                }
                const uint32_t groupOffset = groupIdx * expandedGroupStride;
                Duplicate(scaleLocal[groupOffset], scaleGm.GetValue(scaleIdx), expandedGroupStride);
                if (hasBias) {
                    Duplicate(biasLocal[groupOffset], biasGm.GetValue(scaleIdx), expandedGroupStride);
                }
            }
        }
        scaleQueue.EnQue<T>(scaleLocal);
        if (hasBias) {
            biasQueue.EnQue<T>(biasLocal);
        }

        LocalTensor<T> readyInput = inputQueue.DeQue<T>();
        LocalTensor<T> readyScale = scaleQueue.DeQue<T>();
        LocalTensor<T> readyBias;
        if (hasBias) {
            readyBias = biasQueue.DeQue<T>();
        }

        LocalTensor<T> outputLocal = outputQueue.AllocTensor<T>();
        ComputeElementwiseTile(outputLocal, readyInput, readyScale, readyBias, paddedCount);

        inputQueue.FreeTensor<T>(readyInput);
        scaleQueue.FreeTensor<T>(readyScale);
        if (hasBias) {
            biasQueue.FreeTensor<T>(readyBias);
        }
        outputQueue.EnQue<T>(outputLocal);
        LocalTensor<T> readyOutput = outputQueue.DeQue<T>();
        DataCopyPad(outputGm[offset], readyOutput, copyOutParams);
        outputQueue.FreeTensor<T>(readyOutput);
    }

    // Global memory tensors are anchored at this core's startOffset for input
    // and output. scale/bias remain anchored at zero because they are indexed by
    // broadcast group.
    TPipe *pipe = nullptr;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> outputGm;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECIN, 1> scaleQueue;
    TQue<QuePosition::VECIN, 1> biasQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    // startOffset/elemCount describe this core's flattened input range.
    uint32_t startOffset = 0;
    uint32_t elemCount = 0;
    // scaleCount is the flattened scale tensor size; suffixCount is the number
    // of contiguous input elements covered by one scale/bias value.
    uint32_t scaleCount = 0;
    uint32_t suffixCount = 0;
    // expandedGroupStride is suffixCount rounded up for aligned expanded
    // broadcast. tileElems is the UB tile capacity selected on host.
    uint32_t expandedGroupStride = 0;
    uint32_t tileElems = 0;
    uint32_t floatTileElems = 0;
    bool hasBias = false;
    bool vectorScale = false;
    bool expandedBroadcast = false;
    bool relaxedExpandedFp16 = false;
    bool groupAlignedExpandedSplit = false;
};

extern "C" __global__ __aicore__ void scale(GM_ADDR input, GM_ADDR scale,
                                            GM_ADDR bias, GM_ADDR output,
                                            GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(8)) {
        TPipe pipe;
        KernelScale_broadcast_3_midaxis<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, tiling_data.smallSize,
                tiling_data.incSize, tiling_data.formerNum,
                tiling_data.totalSize, tiling_data.mmInputDims,
                tiling_data.mmOtherDims, tiling_data.mmOutputDims,
                tiling_data.nOutputDims, tiling_data.hasBias != 0U, &pipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(1)) {
        // The host tiling fully determines work partitioning and execution mode.
        // The entry only wires tiling values into the templated kernel instance.
        TPipe pipe;
        KernelScale<DTYPE_INPUT> op;
        op.Init(input, scale, bias, output, tiling_data.scaleElemCount,
                tiling_data.suffixElemCount, tiling_data.baseElems,
                tiling_data.formerElems, tiling_data.baseGroups,
                tiling_data.formerGroups, tiling_data.tileSize,
                tiling_data.expandedGroupStride,
                tiling_data.hasBias, tiling_data.useExpandedBroadcast,
                tiling_data.useRelaxedExpandedFp16,
                tiling_data.useGroupAlignedExpandedSplit, &pipe);
        op.Process();
        return;
    }
}
