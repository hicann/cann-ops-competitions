#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline __gm__ void* ListTensorGetNth(__gm__ void* data, uint32_t n) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t *>(data);
    uint64_t dataPtrOffset = *dataAddr;
    __gm__ uint64_t* dataPtr_ = dataAddr + (dataPtrOffset >> 3);
    return reinterpret_cast<__gm__ void *>(dataPtr_[n]);
}

// 选择 Gather 的工作类型: <4字节的类型(int8/uint8/int16/fp16/bf16) 用 float 做 gather
// (硬件对 1/2 字节 gather 的逐元素寻址不可靠, float(4字节) 已验证正确), 4字节类型直接用本身.
template<bool C, typename A, typename B> struct CondT { using type = A; };
template<typename A, typename B> struct CondT<false, A, B> { using type = B; };

template<typename T>
class KernelUnpack {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, const UnpackTilingData& tiling) {
        this->axis      = tiling.axis;
        this->num       = tiling.num;
        this->ndim      = tiling.ndim;
        this->dtypeSize = tiling.dtypeSize;
        for (int i = 0; i < 4; ++i) this->shape[i] = tiling.shape[i];
        this->tileDataNum = tiling.tileDataNum;
        this->transRows   = tiling.transRows;

        this->outerSize = 1;
        for (int i = 0; i < axis; ++i) outerSize *= shape[i];
        this->innerSize = 1;
        for (int i = axis + 1; i < ndim; ++i) innerSize *= shape[i];
        this->_num = shape[axis];

        xBase   = reinterpret_cast<__gm__ T*>(x);
        outDesc = out;

        // Task distribution for Process() (generic path)
        uint64_t totalTasks = (uint64_t)num * outerSize;
        uint32_t coreIdx    = GetBlockIdx();
        uint32_t coreNum    = GetBlockNum();
        uint64_t tasksPerCore = totalTasks / coreNum;
        uint64_t rem          = totalTasks % coreNum;
        if (coreIdx < rem) {
            myTaskCount = tasksPerCore + 1;
            myStartTask = coreIdx * myTaskCount;
        } else {
            myTaskCount = tasksPerCore;
            myStartTask = rem * (tasksPerCore + 1) + (coreIdx - rem) * tasksPerCore;
        }
    }

    // Buffer setup for Path 0/10/20 (in-place copy via TQueBind).
    __aicore__ inline void SetupDefault(TPipe* pipe) {
        pipe->InitBuffer(queBind, BUFFER_NUM, tileDataNum * sizeof(T));
    }

    // Gather 工作类型: <4字节用 float(已验证可靠), 4字节用本身.
    static constexpr bool CAST = (sizeof(T) < 4);
    using GT = typename CondT<CAST, float, T>::type;

    // Buffer setup for Path 30 (Gather transpose). innerSize==1 assumed.
    __aicore__ inline void SetupTranspose(TPipe* pipe) {
        uint32_t E = transRows * num * innerSize;          // dense chunk elements
        pipe->InitBuffer(srcQue, 1, E * sizeof(T));         // dense src (type T)
        pipe->InitBuffer(dstQue, 2, transRows * innerSize * sizeof(GT)); // gather result (GT)
        pipe->InitBuffer(baseOffBuf, transRows * innerSize * sizeof(uint32_t));
        pipe->InitBuffer(offBuf,     transRows * innerSize * sizeof(uint32_t));
        if constexpr (CAST) {
            pipe->InitBuffer(srcHalfBuf, E * sizeof(half));                 // int8->half (src 桥接)
            pipe->InitBuffer(gsrcBuf, E * sizeof(GT));                      // half->float (gather 源)
            pipe->InitBuffer(tmpHalfBuf, transRows * innerSize * sizeof(half)); // float->half (dst 桥接)
            pipe->InitBuffer(outQue, 2, transRows * innerSize * sizeof(T)); // half->T cast back
        }
        // baseOff[i] = i * (num*inner) * sizeof(GT)  —— gather 以 GT 元素字节为单位寻址.
        // 每个输出 k 的偏移 = baseOff + k*inner*sizeof(GT), 运行时用 Adds 加 (srcBaseAddr 恒 0).
        LocalTensor<int32_t> baseOff = baseOffBuf.Get<int32_t>();
        Arange<int32_t>(baseOff, (int32_t)0,
            (int32_t)(num * innerSize * (uint32_t)sizeof(GT)), (int32_t)(transRows * innerSize));
    }

    // ── Path 30: Gather transpose for innerSize==1 ───────────────────────────
    // Per chunk of R rows: 1 dense contiguous read [R*num] -> for each output k,
    // Gather its column (R elems) into a contiguous tmp -> 1 big contiguous write.
    // Block count drops from outer*num to (outer/R)*num.
    __aicore__ inline void ProcessTranspose() {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t coreNum = GetBlockNum();
        uint64_t rowsPerCore = outerSize / coreNum;
        uint64_t rem         = outerSize % coreNum;
        uint64_t myRows, myRowStart;
        if (coreIdx < (uint32_t)rem) {
            myRows     = rowsPerCore + 1;
            myRowStart = (uint64_t)coreIdx * myRows;
        } else {
            myRows     = rowsPerCore;
            myRowStart = rem * (rowsPerCore + 1) + (uint64_t)(coreIdx - rem) * rowsPerCore;
        }
        if (myRows == 0) return;

        DataCopyPadExtParams<T> padParams = {false, 0, 0, (T)0};
        LocalTensor<int32_t> baseOff = baseOffBuf.Get<int32_t>();

        uint64_t rowPos   = myRowStart;
        uint64_t rowsLeft = myRows;
        uint32_t R = transRows;

        while (rowsLeft > 0) {
            uint32_t Rc = (rowsLeft < (uint64_t)R) ? (uint32_t)rowsLeft : R;

            // 1. ONE dense contiguous read: Rc*num*inner elements (type T).
            LocalTensor<T> src = srcQue.AllocTensor<T>();
            GlobalTensor<T> srcG;
            srcG.SetGlobalBuffer(xBase + rowPos * (uint64_t)num * innerSize,
                                 (uint64_t)Rc * num * innerSize);
            DataCopyExtParams cpIn = {1, (uint32_t)((uint64_t)Rc * num * innerSize * sizeof(T)), 0, 0, 0};
            DataCopyPad(src, srcG, cpIn, padParams);
            srcQue.EnQue(src);
            src = srcQue.DeQue<T>();

            // gather 源: 1字节先按位 reinterpret 成 int8, 再 int8->half->float
            // (本硬件无 int8<->float 重载, 只有 int8<->half / half<->float; 按位无损).
            // 4字节(fp32/int32)直接用 src.
            LocalTensor<GT> gsrc;
            if constexpr (CAST) {
                uint32_t cnt = (uint32_t)((uint64_t)Rc * num * innerSize);
                LocalTensor<half> srcH = srcHalfBuf.Get<half>();
                Cast(srcH, src.template ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, cnt);
                gsrc = gsrcBuf.Get<GT>();
                Cast(gsrc, srcH, RoundMode::CAST_NONE, cnt);   // half -> float
            } else {
                gsrc = src;
            }
            gsrc.SetSize((uint32_t)((uint64_t)Rc * num * innerSize));

            // 2. For each output k: Gather its column, write one contiguous block.
            for (uint32_t k = 0; k < num; ++k) {
                // off[i] = baseOff[i] + k*inner*sizeof(GT)  (基址加进偏移, srcBaseAddr 恒 0)
                LocalTensor<int32_t> offI = offBuf.Get<int32_t>();
                Adds<int32_t>(offI, baseOff, (int32_t)(k * innerSize * (uint32_t)sizeof(GT)),
                              (int32_t)(Rc * innerSize));
                LocalTensor<uint32_t> off = offBuf.Get<uint32_t>();

                LocalTensor<GT> gdst = dstQue.AllocTensor<GT>();
                AscendC::Gather(gdst, gsrc, off, (uint32_t)0, (uint32_t)(Rc * innerSize));
                dstQue.EnQue(gdst);
                gdst = dstQue.DeQue<GT>();

                __gm__ T* dstSlice = reinterpret_cast<__gm__ T*>(
                    ListTensorGetNth(reinterpret_cast<__gm__ void*>(outDesc), k));
                GlobalTensor<T> dstGm;
                dstGm.SetGlobalBuffer(dstSlice + rowPos * innerSize, (uint64_t)Rc * innerSize);
                DataCopyExtParams cpOut = {1, (uint32_t)((uint64_t)Rc * innerSize * sizeof(T)), 0, 0, 0};

                if constexpr (CAST) {
                    // float -> half -> int8 -> (按位) T. float->int8 文档无, 故经 half 中转;
                    // 值是 [-128,127] 的精确整数, half 可精确表示, 往返无损.
                    LocalTensor<half> th = tmpHalfBuf.Get<half>();
                    Cast(th, gdst, RoundMode::CAST_NONE, (uint32_t)(Rc * innerSize));
                    LocalTensor<T> cOut = outQue.AllocTensor<T>();
                    Cast(cOut.template ReinterpretCast<int8_t>(), th, RoundMode::CAST_ROUND,
                         (uint32_t)(Rc * innerSize));
                    outQue.EnQue(cOut);
                    cOut = outQue.DeQue<T>();
                    DataCopyPad(dstGm, cOut, cpOut);
                    outQue.FreeTensor(cOut);
                } else {
                    DataCopyPad(dstGm, gdst, cpOut);
                }
                dstQue.FreeTensor(gdst);
            }
            srcQue.FreeTensor(src);

            rowPos   += Rc;
            rowsLeft -= Rc;
        }
    }

    // 鈹€鈹€ Path 0: general strided-gather, handles any innerSize 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    __aicore__ inline void Process() {
        if (myTaskCount == 0) return;

        uint32_t inner_bytes     = innerSize * sizeof(T);
        uint32_t blockLen_bytes  = inner_bytes;
        uint16_t blockLen_blocks = (uint16_t)((inner_bytes + 31) / 32);

        uint32_t gm_src_pitch_bytes = _num * inner_bytes;
        uint32_t gm_src_stride_gap  = gm_src_pitch_bytes - blockLen_bytes;
        uint16_t ub_dst_stride_gap  = 0;
        uint16_t ub_src_stride_gap  = 0;
        uint32_t gm_dst_stride_gap  = 0;

        uint32_t elements_per_row_ub = blockLen_blocks * (32u / sizeof(T));
        uint32_t maxRowsPerTile = (elements_per_row_ub > 0 && elements_per_row_ub <= tileDataNum)
                                  ? tileDataNum / elements_per_row_ub : 1;
        if (maxRowsPerTile > 4095) maxRowsPerTile = 4095;

        bool rowFitsInUB = (elements_per_row_ub <= tileDataNum);

        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        uint64_t currentTask    = myStartTask;
        uint64_t remainingTasks = myTaskCount;

        if (rowFitsInUB) {
            // Batch multiple complete rows per DMA
            while (remainingTasks > 0) {
                uint32_t sliceIdx = (uint32_t)(currentTask / outerSize);
                uint32_t rowStart = (uint32_t)(currentTask % outerSize);

                __gm__ T* dstSlice = reinterpret_cast<__gm__ T*>(
                    ListTensorGetNth(reinterpret_cast<__gm__ void*>(outDesc), sliceIdx));

                uint32_t maxRowsInSlice = outerSize - rowStart;
                uint32_t batchRows = (remainingTasks < maxRowsInSlice)
                                     ? (uint32_t)remainingTasks : maxRowsInSlice;
                if (batchRows > maxRowsPerTile) batchRows = maxRowsPerTile;

                LocalTensor<T> local = queBind.AllocTensor<T>();

                uint32_t srcOff = rowStart * _num * innerSize + sliceIdx * innerSize;
                GlobalTensor<T> srcTile;
                srcTile.SetGlobalBuffer(xBase + srcOff, batchRows * elements_per_row_ub);

                DataCopyExtParams cpIn = {
                    (uint16_t)batchRows, blockLen_bytes,
                    gm_src_stride_gap, ub_dst_stride_gap, 0
                };
                DataCopyPad(local, srcTile, cpIn, padParams);
                queBind.EnQue(local);

                local = queBind.DeQue<T>();
                GlobalTensor<T> dstTile;
                dstTile.SetGlobalBuffer(dstSlice + rowStart * innerSize,
                                        batchRows * elements_per_row_ub);

                DataCopyExtParams cpOut = {
                    (uint16_t)batchRows, blockLen_bytes,
                    ub_src_stride_gap, gm_dst_stride_gap, 0
                };
                DataCopyPad(dstTile, local, cpOut);
                queBind.FreeTensor(local);

                currentTask    += batchRows;
                remainingTasks -= batchRows;
            }
        } else {
            // innerSize exceeds UB: chunk within each row
            uint32_t chunkElems = tileDataNum;
            while (remainingTasks > 0) {
                uint32_t sliceIdx = (uint32_t)(currentTask / outerSize);
                uint32_t rowStart = (uint32_t)(currentTask % outerSize);

                __gm__ T* dstSlice = reinterpret_cast<__gm__ T*>(
                    ListTensorGetNth(reinterpret_cast<__gm__ void*>(outDesc), sliceIdx));

                uint32_t srcRowBase = rowStart * _num * innerSize + sliceIdx * innerSize;
                uint32_t dstRowBase = rowStart * innerSize;

                for (uint32_t off = 0; off < innerSize; off += chunkElems) {
                    uint32_t copyLen = innerSize - off;
                    if (copyLen > chunkElems) copyLen = chunkElems;

                    LocalTensor<T> local = queBind.AllocTensor<T>();
                    GlobalTensor<T> srcTile;
                    srcTile.SetGlobalBuffer(xBase + srcRowBase + off, copyLen);

                    uint32_t bytes = copyLen * sizeof(T);
                    DataCopyExtParams cpIn = {1, bytes, 0, 0, 0};
                    DataCopyPad(local, srcTile, cpIn, padParams);
                    queBind.EnQue(local);

                    local = queBind.DeQue<T>();
                    GlobalTensor<T> dstTile;
                    dstTile.SetGlobalBuffer(dstSlice + dstRowBase + off, copyLen);
                    DataCopyExtParams cpOut = {1, bytes, 0, 0, 0};
                    DataCopyPad(dstTile, local, cpOut);
                    queBind.FreeTensor(local);
                }

                currentTask    += 1;
                remainingTasks -= 1;
            }
        }
    }

    // 鈹€鈹€ Path 10: outerSize==1, distribute by total elements across all cores 鈹€
    __aicore__ inline void ProcessOuter1Fast() {
        uint64_t totalElems = (uint64_t)this->num * this->innerSize;
        uint32_t coreIdx = GetBlockIdx();
        uint32_t coreNum = GetBlockNum();
        uint64_t start   = totalElems * coreIdx / coreNum;
        uint64_t end     = totalElems * (coreIdx + 1) / coreNum;
        if (start >= end) return;

        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        uint64_t pos          = start;
        uint32_t cachedSlice  = 0xFFFFFFFFU;
        __gm__ T* dstSlice    = (__gm__ T*)0;

        while (pos < end) {
            uint32_t sliceIdx = (uint32_t)(pos / this->innerSize);
            uint32_t offset   = (uint32_t)(pos - (uint64_t)sliceIdx * this->innerSize);
            if (sliceIdx != cachedSlice) {
                cachedSlice = sliceIdx;
                dstSlice = reinterpret_cast<__gm__ T*>(
                    ListTensorGetNth(reinterpret_cast<__gm__ void*>(outDesc), sliceIdx));
            }

            uint32_t remainInSlice  = this->innerSize - offset;
            uint64_t remainForCore  = end - pos;
            uint32_t len            = this->tileDataNum;
            if ((uint64_t)len > remainForCore) len = (uint32_t)remainForCore;
            if (len > remainInSlice)            len = remainInSlice;
            if (len == 0) break;

            GlobalTensor<T> srcTile;
            GlobalTensor<T> dstTile;
            srcTile.SetGlobalBuffer(xBase + pos, len);
            dstTile.SetGlobalBuffer(dstSlice + offset, len);

            LocalTensor<T> local = queBind.AllocTensor<T>();
            uint32_t bytes = len * sizeof(T);
            DataCopyExtParams cpIn = {1, bytes, 0, 0, 0};
            DataCopyPad(local, srcTile, cpIn, padParams);
            queBind.EnQue(local);

            local = queBind.DeQue<T>();
            DataCopyExtParams cpOut = {1, bytes, 0, 0, 0};
            DataCopyPad(dstTile, local, cpOut);
            queBind.FreeTensor(local);
            pos += len;
        }
    }

    // 鈹€鈹€ Path 20: small innerSize 鈥?one contiguous read + strided UB鈫扜M writes
    // Instead of: strided GM鈫扷B gather (tiny blockLen, huge gap) per slice
    // We do:      1 big contiguous read per chunk + num writes with UB stride
    // Path 20: padded UB layout
    // KEY INSIGHT from DataCopyPad API doc:
    //   UB->GM srcStride unit = 32B blocks (NOT bytes) -- old code was wrong by x32
    //   LocalTensor base must be 32B-aligned
    //   GM->UB: framework auto-pads each block to 32B boundary in UB (dstStride=0)
    //   GM->UB: blockLen non-aligned is OK; UB->GM: blockLen non-aligned is OK too
    //
    // With dstStride=0, GM->UB places block k at byte k*padded_bytes (32B-aligned).
    // local[k*padded_inner] is at byte k*padded_bytes -> always 32B-aligned.
    // write srcStride = (num-1)*padded_bytes/32  (correct 32B-block units).
    __aicore__ inline void ProcessSmallInner() {
        uint32_t innerBytes   = innerSize * (uint32_t)sizeof(T);
        uint32_t padded_bytes = (innerBytes + 31u) / 32u * 32u;   // round up to 32B
        uint32_t padded_inner = padded_bytes / (uint32_t)sizeof(T); // UB elements per slot
        uint32_t stripe_pad   = _num * padded_inner; // padded UB elements per outer row

        uint32_t chunk_size = (stripe_pad > 0) ? tileDataNum / stripe_pad : 1u;
        if (chunk_size == 0) chunk_size = 1;

        uint32_t coreIdx = GetBlockIdx();
        uint32_t coreNum = GetBlockNum();

        uint64_t rowsPerCore = outerSize / coreNum;
        uint64_t rem         = outerSize % coreNum;
        uint64_t myRows, myRowStart;
        if (coreIdx < (uint32_t)rem) {
            myRows     = rowsPerCore + 1;
            myRowStart = (uint64_t)coreIdx * myRows;
        } else {
            myRows     = rowsPerCore;
            myRowStart = rem * (rowsPerCore + 1) + (uint64_t)(coreIdx - rem) * rowsPerCore;
        }
        if (myRows == 0) return;

        DataCopyPadExtParams<T> padParams = {false, 0, 0, (T)0};

        // UB->GM srcStride: gap between consecutive rows of the same output in UB.
        // Unit = 32B blocks.  Gap bytes = (num-1)*padded_bytes.
        uint32_t write_srcStride =
            (uint32_t)(((uint64_t)(_num - 1) * padded_bytes) / 32u);

        uint64_t rowPos   = myRowStart;
        uint64_t rowsLeft = myRows;

        while (rowsLeft > 0) {
            uint32_t actual    = (rowsLeft < (uint64_t)chunk_size)
                                 ? (uint32_t)rowsLeft : chunk_size;
            uint64_t srcOffset = rowPos * (uint64_t)(_num * innerSize);

            LocalTensor<T> local = queBind.AllocTensor<T>();

            // 1. ONE multi-block GM->UB read, producing padded UB layout.
            //    blockCount = actual*num blocks, each innerBytes long, contiguous in GM.
            //    srcStride=0 (contiguous GM), dstStride=0 (auto-pad to 32B per block).
            //    After this: block (j*num+k) is at UB byte (j*num+k)*padded_bytes.
            GlobalTensor<T> srcTile;
            srcTile.SetGlobalBuffer(xBase + srcOffset, actual * _num * innerSize);
            DataCopyExtParams cpIn = {
                (uint16_t)(actual * _num), // blockCount: all chunks for this batch
                innerBytes,                 // blockLen (non-aligned OK for GM->UB)
                0,                          // srcStride_GM: contiguous
                0,                          // dstStride_UB: auto-pad fills to 32B
                0
            };
            DataCopyPad(local, srcTile, cpIn, padParams);
            queBind.EnQue(local);
            local = queBind.DeQue<T>();

            // 2. For each output k: write 'actual' rows in ONE DataCopyPad call.
            //    local[k*padded_inner] at byte k*padded_bytes -> 32B-aligned.
            //    srcStride = (num-1)*padded_bytes/32 (32B-block units) -> skip
            //    to next row of the same output in the padded UB layout.
            //    dstStride=0 -> contiguous writes to output k.
            for (uint32_t k = 0; k < _num; ++k) {
                __gm__ T* dstSlice = reinterpret_cast<__gm__ T*>(
                    ListTensorGetNth(reinterpret_cast<__gm__ void*>(outDesc), k));

                GlobalTensor<T> dstTile;
                dstTile.SetGlobalBuffer(dstSlice + rowPos * innerSize,
                                        actual * innerSize);

                DataCopyExtParams cpOut = {
                    (uint16_t)actual,   // blockCount: one block per outer row
                    innerBytes,          // blockLen (non-aligned OK for UB->GM)
                    write_srcStride,     // srcStride: 32B-block units (CORRECT unit)
                    0,                   // dstStride_GM: contiguous
                    0
                };
                DataCopyPad(dstTile, local[k * padded_inner], cpOut);
            }

            queBind.FreeTensor(local);

            rowPos   += actual;
            rowsLeft -= actual;
        }
    }

private:
    __gm__ T* xBase;
    GM_ADDR   outDesc;
    int32_t   axis, ndim;
    uint32_t  num, _num, outerSize, innerSize, dtypeSize, tileDataNum;
    int32_t   shape[4];
    uint32_t  transRows;

    uint64_t  myStartTask, myTaskCount;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queBind;

    // Path 30 (Gather transpose) buffers
    TQue<QuePosition::VECIN, 1>  srcQue;      // dense read (type T)
    TQue<QuePosition::VECOUT, 2> dstQue;      // gather result (type GT)
    TBuf<QuePosition::VECCALC>   srcHalfBuf;  // CAST: int8->half 桥接 (src 端)
    TBuf<QuePosition::VECCALC>   gsrcBuf;     // CAST: half->float (gather 源)
    TBuf<QuePosition::VECCALC>   tmpHalfBuf;  // CAST: float->half 中转 (1字节 cast-back)
    TQue<QuePosition::VECOUT, 2> outQue;      // CAST: half->T cast back
    TBuf<QuePosition::VECCALC>   baseOffBuf;  // baseOff[j] = j*num*inner*sizeof(GT)
    TBuf<QuePosition::VECCALC>   offBuf;      // per-output work: baseOff + k*inner*sizeof(GT)
};

extern "C" __global__ __aicore__ void unpack(GM_ADDR x, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelUnpack<DTYPE_INPUT> op;
    op.Init(x, out, tiling_data);

    if (TILING_KEY_IS(30)) {
        KERNEL_TASK_TYPE(30, KERNEL_TYPE_AIV_ONLY);
        op.SetupTranspose(&pipe);
        op.ProcessTranspose();
    } else if (TILING_KEY_IS(10)) {
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_AIV_ONLY);
        op.SetupDefault(&pipe);
        op.ProcessOuter1Fast();
    } else if (TILING_KEY_IS(20)) {
        KERNEL_TASK_TYPE(20, KERNEL_TYPE_AIV_ONLY);
        op.SetupDefault(&pipe);
        op.ProcessSmallInner();
    } else if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        op.SetupDefault(&pipe);
        op.Process();
    }
}
