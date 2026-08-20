// #include "kernel_operator.h"
// #include "kernel_operator_list_tensor_intf.h"


// using namespace AscendC;

// __aicore__ inline int align_to(int a, int b) {
//     return (a + b - 1) / b * b;
// }

// template<typename T>
// class unpack_0 {
// public:
//     __aicore__ inline unpack_0() {}
    
//     __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int chunk)
//     {
//         pipe = pipeIn;
//         this->inputGm.SetGlobalBuffer((__gm__ T *)input);
//         this->listTensorDesc = listTensorDesc;
//         this->chunk = chunk;

//         pipe->InitBuffer(queBindA, 2, 32768);
//         this->chunksize = 32768 / sizeof(T);
//     }
    
//     __aicore__ inline void Process(int offset, int size, int flag = 0)
//     {
//         int loop = (size + chunksize - 1) / chunksize;
//         int tile = size % chunksize;
//         tile = tile == 0 ? chunksize : tile;
//         uint32_t num_id = offset / chunk;
//         auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//         outputGm.SetGlobalBuffer(dataPtr);
//         int base_offset = offset % chunk;

//         if(flag) {
//             for(int i = 0; i != loop - 1; i++) {
//                 if(base_offset + chunksize <= chunk) {
//                     CopyIn(offset + i * chunksize, chunksize);
//                     CopyOut(base_offset, chunksize);
//                     base_offset += chunksize;
//                     if(base_offset == chunk) {
//                         num_id++;
//                         auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//                         outputGm.SetGlobalBuffer(dataPtr);
//                         base_offset = 0;
//                     }
//                 } else {
//                     int len_left = chunksize;
//                     while(len_left) {
//                         int len = min(chunk - base_offset, len_left);
//                         CopyIn(offset + (i + 1) * chunksize - len_left, align_to(len, 32));
//                         if(len % 32) {
//                             CopyOutPad(base_offset, len);
//                         } else {
//                             CopyOut(base_offset, len);
//                         }

//                         len_left -= len;

//                         base_offset += len;
//                         if(base_offset == chunk) {
//                             num_id++;
//                             auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//                             outputGm.SetGlobalBuffer(dataPtr);
//                             base_offset = 0;
//                         }
//                     }
//                 }
//             }
//             if(base_offset + tile <= chunk) {
//                 CopyIn(offset + (loop - 1) * chunksize, align_to(tile, 32));
//                 if(tile % 32) {
//                     CopyOutPad(base_offset, tile);
//                 } else {
//                     CopyOut(base_offset, tile);
//                 }
//                 base_offset += tile;
//                 if(base_offset == chunk) {
//                     num_id++;
//                     auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//                     outputGm.SetGlobalBuffer(dataPtr);
//                     base_offset = 0;
//                 }
//             } else {
//                 int len_left = tile;
//                 while(len_left) {
//                     int len = min(chunk - base_offset, len_left);
//                     CopyIn(offset + (loop - 1) * chunksize + tile - len_left, align_to(len, 32));
//                     if(len % 32) {
//                         CopyOutPad(base_offset, len);
//                     } else {
//                         CopyOut(base_offset, len);
//                     }

//                     len_left -= len;

//                     base_offset += len;
//                     if(base_offset == chunk) {
//                         num_id++;
//                         auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//                         outputGm.SetGlobalBuffer(dataPtr);
//                         base_offset = 0;
//                     }
//                 }
//             }
//         } else {
            
//         }
//     }

// private:
//     __aicore__ inline void CopyIn(int offset, int len)
//     {
//         LocalTensor<T> aLocal = queBindA.AllocTensor<T>();
//         DataCopy(aLocal, inputGm[offset], len);
//         //DumpTensor(inputGm[offset], offset, 16);
//         queBindA.EnQue(aLocal);
//     }

//     __aicore__ inline void CopyOut(int offset, int len)
//     {
//         LocalTensor<T> cLocal = queBindA.DeQue<T>();
//         //DumpTensor(cLocal, offset, 16);
//         DataCopy(outputGm[offset], cLocal, len);
//         queBindA.FreeTensor(cLocal);
//         //DumpTensor(outputGm[offset], offset, 16);
//     }
    
//     __aicore__ inline void CopyOutPad(int offset, int len)
//     {
//         DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
//         LocalTensor<T> cLocal = queBindA.DeQue<T>();
//         //DumpTensor(cLocal, offset, 16);
//         DataCopyPad(outputGm[offset], cLocal, copyParams);
//         queBindA.FreeTensor(cLocal);
//         //DumpTensor(outputGm[offset], offset, 16);
//     }
    
// private:
//     int chunksize, chunk;
    
//     TPipe* pipe;
//     TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindA;
//     ListTensorDesc listTensorDesc;
    
//     GlobalTensor<T> inputGm;
//     GlobalTensor<T> outputGm;
// };

// template<typename T>
// class unpack_1 {
// public:
//     __aicore__ inline unpack_1() {}
    
//     __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int m, int n, int k)
//     {
//         pipe = pipeIn;
//         this->inputGm.SetGlobalBuffer((__gm__ T *)input);
//         this->listTensorDesc = listTensorDesc;
        
//         pipe->InitBuffer(queBindA, 2, 32768);
//         this->chunksize = 32768 / sizeof(T);
//         this->m = m;
//         this->n = n;
//         this->k = k;
//         this->chunk = m*k;
//     }
    
//     __aicore__ inline void Process(int offset, int size, int flag = 0)
//     {
//         int nk = n*k;
//         if(k > chunksize) {
//             int loop = (k + chunksize - 1) / chunksize;
//             int tile = k % chunksize;
//             tile = tile == 0 ? chunksize : tile;
//             int start = flag ? 0 : size - 1;
//             int end = flag ? size : -1;
//             int step = flag ? 1 : -1;
//             for(int i = start; i != end; i+=step) {
//                 int offsetA = int((offset + i) / m) * k + int((offset + i) % m) * nk;
//                 uint32_t num_id = (offset + i) / m;
//                 auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//                 outputGm.SetGlobalBuffer(dataPtr);
//                 int base_offset = int((offset + i) % m) * k;
//                 for(int j = 0; j != loop - 1; j++) {
//                     CopyIn(offsetA + j * chunksize, chunksize);
//                     CopyOut(base_offset + j * chunksize, chunksize);
//                 }
//                 CopyIn(offsetA + (loop - 1) * chunksize, align_to(tile, 32));
//                 if(tile % 32) {
//                     CopyOutPad(base_offset + (loop - 1) * chunksize, tile);
//                 } else {
//                     CopyOut(base_offset + (loop - 1) * chunksize, tile);
//                 }
//             }
//         } else if(k * 2 > chunksize || k % 32) {
//             int start = flag ? 0 : size - 1;
//             int end = flag ? size : -1;
//             int step = flag ? 1 : -1;
//             for(int i = start; i != end; i+=step) {
//                 int offsetA = int((offset + i) / m) * k + int((offset + i) % m) * nk;
//                 uint32_t num_id = (offset + i) / m;
//                 auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//                 outputGm.SetGlobalBuffer(dataPtr);
//                 int base_offset = int((offset + i) % m) * k;
//                 CopyIn(offsetA, align_to(k, 32));
//                 if(k % 32) {
//                     CopyOutPad(base_offset, k);
//                 } else {
//                     CopyOut(base_offset, k);
//                 }
//             }
//         } else {
//             // ================= TODO 完成部分 =================
//             // 场景：k 较小，单次搬运效率低。
//             // 策略：一次搬入多行（batch 行），输入带跨度 nk，UB 内部连续排列。
            
//             // 计算 UB 能容纳的最大行数 (确保 batch * k 不超过 chunksize)
//             int max_batch = chunksize / k;
//             if (max_batch > 65535) max_batch = 65535; // 硬件指令 repeatTimes 限制
//             if (max_batch == 0) max_batch = 1;

//             int i = 0;
//             while (i < size) {
//                 // 计算当前任务在输出侧的归属
//                 int global_idx = offset + i;
//                 uint32_t num_id = global_idx / m;    // 属于第几个输出张量
//                 int m_idx = global_idx % m;          // 属于该张量的第几行
                
//                 // 计算本批次可以搬运的行数 (不能超过 max_batch，且不能跨越当前输出张量的 m 边界)
//                 int batch = min(max_batch, size - i);
//                 batch = min(batch, m - m_idx);
                
//                 // 1. 切换输出张量指针
//                 auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
//                 outputGm.SetGlobalBuffer(dataPtr);
                
//                 // 2. 计算地址偏移
//                 // Input 偏移：i_unstack * k + m_idx * (n * k)
//                 int offsetA = num_id * k + m_idx * nk;
//                 // Output 偏移：m_idx * k
//                 int base_offset = m_idx * k;
                
//                 // 3. 核心：调用 2D 搬运函数，将间隔的 k 向量拼成连续块
//                 // 这里的 total_len 是搬入 UB 后的总长度
//                 int total_len = batch * k;
//                 CopyInBatch(offsetA, batch, k, nk);
                
//                 // 4. 复用你原来的 CopyOut / CopyOutPad
//                 // 因为 UB 内部现在是连续的，所以直接搬出 total_len 长度即可
//                 if (total_len % 32 == 0) {
//                     CopyOut(base_offset, total_len);
//                 } else {
//                     CopyOutPad(base_offset, total_len);
//                 }
                
//                 i += batch;
//             }
//         }
//     }

// private:
//     __aicore__ inline void CopyIn(int offset, int len)
//     {
//         LocalTensor<T> aLocal = queBindA.AllocTensor<T>();
//         DataCopy(aLocal, inputGm[offset], len);
//         //DumpTensor(inputGm[offset], offset, 16);
//         queBindA.EnQue(aLocal);
//     }

//     __aicore__ inline void CopyOut(int offset, int len)
//     {
//         LocalTensor<T> cLocal = queBindA.DeQue<T>();
//         //DumpTensor(cLocal, offset, 16);
//         DataCopy(outputGm[offset], cLocal, len);
//         queBindA.FreeTensor(cLocal);
//         //DumpTensor(outputGm[offset], offset, 16);
//     }
    
//     __aicore__ inline void CopyOutPad(int offset, int len)
//     {
//         DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
//         LocalTensor<T> cLocal = queBindA.DeQue<T>();
//         //DumpTensor(cLocal, offset, 16);
//         DataCopyPad(outputGm[offset], cLocal, copyParams);
//         queBindA.FreeTensor(cLocal);
//         //DumpTensor(outputGm[offset], offset, 16);
//     }

//     __aicore__ inline void CopyInBatch(int offset, int batch, int k_len, int nk_stride)
//     {
//         LocalTensor<T> aLocal = queBindA.AllocTensor<T>();
        
//         // 配置 2D 搬运参数
//         // repeatTimes: 搬运多少行 (batch)
//         // blockLen: 每行搬运的字节数 (k * sizeof(T))
//         // srcStride: 输入侧每行结束到下一行开始的跳跃字节数 ((n-1) * k * sizeof(T))
//         // dstStride: 输出侧（UB）每行结束到下一行开始的跳跃字节数 (0，表示在 UB 里紧凑排列)
        
//         uint32_t block_bytes = k_len * sizeof(T);
//         uint32_t stride_bytes = (nk_stride - k_len) * sizeof(T);
        
//         DataCopyExtParams copyParams{
//             static_cast<uint16_t>(batch), 
//             block_bytes, 
//             stride_bytes, 
//             0, // dstStride=0 关键点：实现 UB 内的连续挤压
//             0
//         };
//         DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        
//         // 使用 DataCopyPad 支持非 32B 对齐的跨度搬运
//         DataCopyPad(aLocal, inputGm[offset], copyParams, padParams);
        
//         queBindA.EnQue(aLocal);
//     }
    
// private:
//     int chunksize, m, n, k, chunk;
    
//     TPipe* pipe;
//     TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindA;
//     ListTensorDesc listTensorDesc;
    
//     GlobalTensor<T> inputGm;
//     GlobalTensor<T> outputGm;
// };

// template<typename T>
// class unpack_2 {
// public:
//     __aicore__ inline unpack_2() {}
    
//     __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int m, int n)
//     {
//         pipe = pipeIn;
//         this->inputGm.SetGlobalBuffer((__gm__ T *)input);
//         this->listTensorDesc = listTensorDesc;
//         this->m = m;
//         this->n = n;
        
//         // 核心设计：固定每次读取 16 行（完美契合 TransDataTo5HD 的 16 个源指针要求）
//         // 列方向 TILE_N 设为 256（保证一次转置的块大小适中，且不会撑爆 UB）
//         this->TILE_M = 16;
//         this->TILE_N = 512;

//         pipe->InitBuffer(inQueueSrc, 2, TILE_M * TILE_N * sizeof(T));
//         pipe->InitBuffer(outQueueDst, 2, TILE_M * TILE_N * sizeof(T));
//     }
    
//     __aicore__ inline void Process(int offset, int size, int flag = 0)
//     {
//         int start_row = offset;
//         int end_row = offset + size;

//         // 外层循环：每次处理 M 维度上的 16 行
//         for (int i = start_row; i < end_row; i += TILE_M) {
//             int valid_rows = min(TILE_M, end_row - i);

//             // 内层循环：按 TILE_N 切分 N 维度
//             for (int j = 0; j < n; j += TILE_N) {
//                 int valid_cols = min(TILE_N, n - j);

//                 CopyIn(i, j, valid_rows, valid_cols);
//                 Compute(valid_rows, valid_cols);
//                 CopyOut(i, j, valid_rows, valid_cols);
//             }
//         }
//     }

// private:
//     __aicore__ inline void CopyIn(int row_start, int col_start, int valid_rows, int valid_cols)
//     {
//         LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
//         int loop = valid_cols / block_size;
//         DataCopyExtParams copyParams{uint16_t(valid_rows), uint32_t(valid_cols * sizeof(T)), uint32_t((n - valid_cols) * sizeof(T)), uint32_t(16 * sizeof(T) - align_to(valid_cols * sizeof(T), 32) / 32), 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
//         DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        
//         // 从 Global Memory 中按行读取 2D 数据块到 UB
//         DataCopyPad(srcLocal, inputGm[row_start*n+col_start], copyParams, padParams); 
//         // DumpTensor(srcLocal, 0, 16);
//         inQueueSrc.EnQue(srcLocal);
//     }

//     __aicore__ inline void Compute(int valid_rows, int valid_cols)
//     {
//         LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
//         LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

//         // ========== 核心转置魔法 ==========
//         TransDataTo5HDParams transDataParams;
//         transDataParams.dstHighHalf = false;
//         transDataParams.srcHighHalf = false;

//         int pad_cols = align_to(valid_cols, block_size); // NPU 要求 repeat 必须是 C0 的整数倍
        
//         transDataParams.repeatTimes = pad_cols / block_size;
//         transDataParams.dstRepStride = pad_cols <= block_size ? 0 : 16;
//         transDataParams.srcRepStride = pad_cols <= block_size ? 0 : 1;

//         uint64_t dstLocalList[16];
//         uint64_t srcLocalList[16];

//         if constexpr (sizeof(T) == 2) { 
//             // 16-bit (half, bfloat16) 映射逻辑：1 datablock = 16 elements
//             for (int i = 0; i < 16; i++) {
//                 dstLocalList[i] = (uint64_t)(dstLocal[i * 16].GetPhyAddr());
//                 srcLocalList[i] = (uint64_t)(srcLocal[i * TILE_N].GetPhyAddr());
//             }
//             AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
//         } 
//         else if constexpr (sizeof(T) == 4) { 
//             // 32-bit (float, int32) 映射逻辑：1 datablock = 8 elements
//             for (int i = 0; i < 16; i++) {
//                 // 32-bit 时，每两个指针拼接成输出的一行
//                 dstLocalList[i] = (uint64_t)(dstLocal[(i / 2) * 16 + (i % 2) * 8].GetPhyAddr());
//                 srcLocalList[i] = (uint64_t)(srcLocal[i * TILE_N].GetPhyAddr());
//             }
//             AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
//         } 
//         else {
//             // 8-bit (int8, uint8, bool) 的防崩溃降级方案
//             // 由于 8-bit 的 TransData 需要两轮（HighHalf/LowHalf）非常复杂，
//             // 为了保证绝对的稳定性和泛化性，这里使用 UB 内的高速标量转置
//             for (int r = 0; r < valid_rows; r++) {
//                 for (int c = 0; c < valid_cols; c++) {
//                     dstLocal.SetValue(c * 16 + r, srcLocal.GetValue(r * TILE_N + c));
//                 }
//             }
//         }

//         outQueueDst.EnQue(dstLocal);
//         inQueueSrc.FreeTensor(srcLocal);
//     }

//     __aicore__ inline void CopyOut(int row_start, int col_start, int valid_rows, int valid_cols)
//     {
//         LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
//         // DumpTensor(dstLocal, 1, 16);
//         // 转置后，dstLocal 的形状变成了 [TILE_N, 16]
//         // 也就是它的第 c 行，包含了我们要写入 Output[col_start + c] 的 valid_rows 个数据
//         for (int c = 0; c < valid_cols; c++) {
//             uint32_t out_idx = col_start + c;
//             GlobalTensor<T> out_c;
//             out_c.SetGlobalBuffer(listTensorDesc.GetDataPtr<T>(out_idx));

//             // 将转置好的这一小段连续内存，写入对应的输出 Tensor 的正确行位置
//             if(valid_rows % 16) {
//                 DataCopyExtParams copyParams{1, static_cast<uint32_t>(valid_rows * sizeof(T)), 0, 0, 0};
//                 DataCopyPad(out_c[row_start], dstLocal[c * 16], copyParams);
//             } else {
//                 DataCopy(out_c[row_start], dstLocal[c * 16], valid_rows);
//             }
//         }

//         outQueueDst.FreeTensor(dstLocal);
//     }
    
// private:
//     int m, n;
//     int TILE_M, TILE_N;
//     int block_size = 32 / sizeof(T);
//     TPipe* pipe;
//     TQue<QuePosition::VECIN, 1> inQueueSrc;
//     TQue<QuePosition::VECOUT, 1> outQueueDst;
//     ListTensorDesc listTensorDesc;
//     GlobalTensor<T> inputGm;
// };

// class unpack_21 {
// public:
//     __aicore__ inline unpack_21() {}
    
//     __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int m, int n)
//     {
//         pipe = pipeIn;
//         this->inputGm.SetGlobalBuffer((__gm__ int32_t *)input);
//         this->listTensorDesc = listTensorDesc;
//         this->m = m;
//         this->n = n;
        
//         // 核心设计：固定每次读取 16 行（完美契合 TransDataTo5HD 的 16 个源指针要求）
//         // 列方向 TILE_N 设为 256（保证一次转置的块大小适中，且不会撑爆 UB）
//         this->TILE_M = 400;
//         this->TILE_N = 40;

//         pipe->InitBuffer(inQueueSrc, 1, TILE_M * TILE_N * sizeof(int32_t));
//         pipe->InitBuffer(outQueueDst, 1, TILE_M * TILE_N * sizeof(int32_t));
//         pipe->InitBuffer(workBuf, TILE_M * TILE_N * sizeof(uint32_t));
//     }
    
//     __aicore__ inline void Process(int offset, int size, int flag = 0)
//     {
//         int start_row = offset;
//         int end_row = offset + size;

//         tempTensor = workBuf.Get<uint32_t>();
//         LocalTensor<int32_t> interpreTensor = tempTensor.template ReinterpretCast<int32_t>();
//         for(int i = 0; i != TILE_M; i++) {
//             CreateVecIndex(interpreTensor[i*TILE_N], i*n, TILE_N);
//         }
//         Muls(interpreTensor, interpreTensor, 4, TILE_M * TILE_N);

//         // 外层循环：每次处理 M 维度上的 16 行
//         for (int i = start_row; i < end_row; i += TILE_M) {
//             int valid_rows = min(int(TILE_M), end_row - i);

//             CopyIn(i, valid_rows);
//             Compute();
//             CopyOut(i, valid_rows);
//         }
//     }

// private:
//     __aicore__ inline void CopyIn(int row_start, int valid_rows)
//     {
//         LocalTensor<int32_t> srcLocal = inQueueSrc.AllocTensor<int32_t>();
        
//         // 从 Global Memory 中按行读取 2D 数据块到 UB
//         DataCopy(srcLocal, inputGm[row_start*n], align_to(valid_rows*n, 32)); 
//         // DumpTensor(srcLocal, 0, 16);
//         inQueueSrc.EnQue(srcLocal);
//     }

//     __aicore__ inline void Compute()
//     {
//         LocalTensor<int32_t> srcLocal = inQueueSrc.DeQue<int32_t>();
//         LocalTensor<int32_t> dstLocal = outQueueDst.AllocTensor<int32_t>();

//         Gather(dstLocal, srcLocal, tempTensor, 0, TILE_M * TILE_N);
//         // ========== 核心转置魔法 ==========
//         TransDataTo5HDParams transDataParams;
//         transDataParams.dstHighHalf = false;
//         transDataParams.srcHighHalf = false; 
        
//         transDataParams.repeatTimes = TILE_M / 16;
//         transDataParams.dstRepStride = 2;
//         transDataParams.srcRepStride = 2 * TILE_N;

//         uint64_t dstLocalList[16];
//         uint64_t srcLocalList[16];

//         for(int i = 0; i != (TILE_N / 8); i++) {
//             for (int j = 0; j < 16; j++) {
//                 // 32-bit 时，每两个指针拼接成输出的一行
//                 dstLocalList[j] = (uint64_t)(srcLocal[(j / 2) * TILE_M + (j % 2) * 8 + i * 8 * TILE_M].GetPhyAddr());
//                 srcLocalList[j] = (uint64_t)(dstLocal[i * 8 + j * TILE_N].GetPhyAddr());
//             }
//             TransDataTo5HD<int32_t>(dstLocalList, srcLocalList, transDataParams);
//         }
//         Adds(dstLocal, srcLocal, 0, TILE_M * TILE_N);
//         outQueueDst.EnQue(dstLocal);
//         inQueueSrc.FreeTensor(srcLocal);
//     }

//     __aicore__ inline void CopyOut(int row_start, int valid_rows)
//     {
//         LocalTensor<int32_t> dstLocal = outQueueDst.DeQue<int32_t>();

//         for (int c = 0; c < n; c++) {
//             uint32_t out_idx = c;
//             GlobalTensor<int32_t> out_c;
//             out_c.SetGlobalBuffer(listTensorDesc.GetDataPtr<int32_t>(out_idx));
//             // if(GetBlockIdx() == 39) DumpTensor(out_c[9*16384*((c+1)%2)], c, 8);
//             // 将转置好的这一小段连续内存，写入对应的输出 Tensor 的正确行位置
//             if(valid_rows % 8) {
//                 DataCopyExtParams copyParams{1, static_cast<uint32_t>(valid_rows * sizeof(int32_t)), 0, 0, 0};
//                 DataCopyPad(out_c[row_start], dstLocal[c * TILE_M], copyParams);
//             } else {
//                 DataCopy(out_c[row_start], dstLocal[c*TILE_M], valid_rows);
//             }
//         }

//         outQueueDst.FreeTensor(dstLocal);
//     }
    
// private:
//     int m, n;
//     uint32_t TILE_M, TILE_N;
//     int block_size = 8;
//     TPipe* pipe;
//     TQue<QuePosition::VECIN, 1> inQueueSrc;
//     TQue<QuePosition::VECOUT, 1> outQueueDst;
//     TBuf<TPosition::VECCALC> workBuf;
//     ListTensorDesc listTensorDesc;
//     GlobalTensor<int32_t> inputGm;
//     LocalTensor<uint32_t> tempTensor;
// };

// extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
//     GET_TILING_DATA(tiling_data, tiling);
    
//     const uint32_t m = tiling_data.m;
//     const uint32_t n = tiling_data.n;
//     const uint32_t k = tiling_data.k;
//     const uint32_t mode = tiling_data.mode;
//     const uint32_t size = m*n*k;

//     int blockIdx = GetBlockIdx();
//     int blockDim = GetBlockNum();
//     int start, len;
//     if(mode == 1) {
//         int cacheline = 128;
//         int chunks = (size + cacheline - 1) / cacheline;
//         int base_chunks = chunks / blockDim;
//         int tile_chunks = chunks % blockDim;
//         start = (base_chunks * blockIdx + min(blockIdx, tile_chunks)) * cacheline;
//         len = min(int(size) - start, (base_chunks + (blockIdx < tile_chunks)) * cacheline); 
//     } else if(mode == 3) {
//         int chunks = m*n;
//         int base_chunks = chunks / blockDim;
//         int tile_chunks = chunks % blockDim;
//         start = base_chunks * blockIdx + min(blockIdx, tile_chunks);
//         len = base_chunks + (blockIdx < tile_chunks ? 1 : 0);
//     } else if(mode == 2) {
//         // mode=2 时，按 M（行数）均匀分摊给各个 NPU 核心
//         int chunks = m;
//         int base_chunks = chunks / blockDim;
//         int tile_chunks = chunks % blockDim;
//         start = base_chunks * blockIdx + min(blockIdx, tile_chunks);
//         len = base_chunks + (blockIdx < tile_chunks ? 1 : 0);
//     }
//     TPipe pipe;

//     ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void *>(output));

//     if (TILING_KEY_IS(1)){
//         if(mode == 1) {
//             if(len <= 0) return;
//             unpack_0<float> op;
//             op.Init(&pipe, input, listTensorDesc, m*k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             op.Process(start, len, workGm(0)%2);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else if(mode == 3) {
//             if(len <= 0) return;
//             unpack_1<float> op;
//             op.Init(&pipe, input, listTensorDesc, m, n, k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             op.Process(start, len, workGm(0)%2);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else {
//             if(len <= 0) return;
//             unpack_2<DTYPE_INPUT> op;
//             op.Init(&pipe, input, listTensorDesc, m, n);
//             op.Process(start, len, 0);
//         }
//     }
//     else if (TILING_KEY_IS(2)){
//         if(mode == 1) {
//             if(len <= 0) return;
//             unpack_0<half> op;
//             op.Init(&pipe, input, listTensorDesc, m*k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             // op.Process(start, len, 1);
//             op.Process(start, len, 1);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else if(mode == 3) {
//             if(len <= 0) return;
//             unpack_1<half> op;
//             op.Init(&pipe, input, listTensorDesc, m, n, k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             op.Process(start, len, workGm(0)%2);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else {
//             if(len <= 0) return;
//             unpack_2<half> op;
//             op.Init(&pipe, input, listTensorDesc, m, n);
//             op.Process(start, len, 0);
//         }
//     }
//     else if (TILING_KEY_IS(3)){
//         if(mode == 1) {
//             if(len <= 0) return;
//             unpack_0<uint8_t> op;
//             op.Init(&pipe, input, listTensorDesc, m*k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             op.Process(start, len, workGm(0)%2);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else if(mode == 3) {
//             if(len <= 0) return;
//             unpack_1<int8_t> op;
//             op.Init(&pipe, input, listTensorDesc, m, n, k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             op.Process(start, len, workGm(0)%2);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else {
//             if(len <= 0) return;
//             unpack_2<int8_t> op;
//             op.Init(&pipe, input, listTensorDesc, m, n);
//             op.Process(start, len, 0);
//         }
//     }
//     else if (TILING_KEY_IS(4)){
//         if(mode == 1) {
//             if(len <= 0) return;
//             unpack_0<bool> op;
//             op.Init(&pipe, input, listTensorDesc, m*k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             op.Process(start, len, workGm(0)%2);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else if(mode == 3) {
//             if(len <= 0) return;
//             unpack_1<bool> op;
//             op.Init(&pipe, input, listTensorDesc, m, n, k);

//             GlobalTensor<int32_t> workGm;
//             workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
//             op.Process(start, len, workGm(0)%2);
//             if(blockIdx == 0) workGm(0) += 1;
//         } else {
//             if(len <= 0) return;
//             unpack_2<bool> op;
//             op.Init(&pipe, input, listTensorDesc, m, n);
//             op.Process(start, len, 0);
//         }
//     }
//     else if (TILING_KEY_IS(5)){
//         if(len <= 0) return;
//         unpack_21 op;
//         op.Init(&pipe, input, listTensorDesc, m, n);
//         op.Process(start, len, 0);
//     }
// }

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"


using namespace AscendC;

__aicore__ inline int align_to(int a, int b) {
    return (a + b - 1) / b * b;
}

template<typename T>
class unpack_0 {
public:
    __aicore__ inline unpack_0() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int chunk)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ T *)input);
        this->listTensorDesc = listTensorDesc;
        this->chunk = chunk;

        pipe->InitBuffer(queBindA, 2, 32768);
        this->chunksize = 32768 / sizeof(T);
    }
    
    __aicore__ inline void Process(int offset, int size, int flag = 0)
    {
        int loop = (size + chunksize - 1) / chunksize;
        int tile = size % chunksize;
        tile = tile == 0 ? chunksize : tile;
        uint32_t num_id = offset / chunk;
        auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
        outputGm.SetGlobalBuffer(dataPtr);
        int base_offset = offset % chunk;

        if(flag) {
            for(int i = 0; i != loop - 1; i++) {
                if(base_offset + chunksize <= chunk) {
                    CopyIn(offset + i * chunksize, chunksize);
                    CopyOut(base_offset, chunksize);
                    base_offset += chunksize;
                    if(base_offset == chunk) {
                        num_id++;
                        auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
                        outputGm.SetGlobalBuffer(dataPtr);
                        base_offset = 0;
                    }
                } else {
                    int len_left = chunksize;
                    while(len_left) {
                        int len = min(chunk - base_offset, len_left);
                        CopyIn(offset + (i + 1) * chunksize - len_left, align_to(len, 32));
                        if(len % 32) {
                            CopyOutPad(base_offset, len);
                        } else {
                            CopyOut(base_offset, len);
                        }

                        len_left -= len;

                        base_offset += len;
                        if(base_offset == chunk) {
                            num_id++;
                            auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
                            outputGm.SetGlobalBuffer(dataPtr);
                            base_offset = 0;
                        }
                    }
                }
            }
            if(base_offset + tile <= chunk) {
                CopyIn(offset + (loop - 1) * chunksize, align_to(tile, 32));
                if(tile % 32) {
                    CopyOutPad(base_offset, tile);
                } else {
                    CopyOut(base_offset, tile);
                }
                base_offset += tile;
                if(base_offset == chunk) {
                    num_id++;
                    auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
                    outputGm.SetGlobalBuffer(dataPtr);
                    base_offset = 0;
                }
            } else {
                int len_left = tile;
                while(len_left) {
                    int len = min(chunk - base_offset, len_left);
                    CopyIn(offset + (loop - 1) * chunksize + tile - len_left, align_to(len, 32));
                    if(len % 32) {
                        CopyOutPad(base_offset, len);
                    } else {
                        CopyOut(base_offset, len);
                    }

                    len_left -= len;

                    base_offset += len;
                    if(base_offset == chunk) {
                        num_id++;
                        auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
                        outputGm.SetGlobalBuffer(dataPtr);
                        base_offset = 0;
                    }
                }
            }
        } else {
            
        }
    }

private:
    __aicore__ inline void CopyIn(int offset, int len)
    {
        LocalTensor<T> aLocal = queBindA.AllocTensor<T>();
        DataCopy(aLocal, inputGm[offset], len);
        //DumpTensor(inputGm[offset], offset, 16);
        queBindA.EnQue(aLocal);
    }

    __aicore__ inline void CopyOut(int offset, int len)
    {
        LocalTensor<T> cLocal = queBindA.DeQue<T>();
        //DumpTensor(cLocal, offset, 16);
        DataCopy(outputGm[offset], cLocal, len);
        queBindA.FreeTensor(cLocal);
        //DumpTensor(outputGm[offset], offset, 16);
    }
    
    __aicore__ inline void CopyOutPad(int offset, int len)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        LocalTensor<T> cLocal = queBindA.DeQue<T>();
        //DumpTensor(cLocal, offset, 16);
        DataCopyPad(outputGm[offset], cLocal, copyParams);
        queBindA.FreeTensor(cLocal);
        //DumpTensor(outputGm[offset], offset, 16);
    }
    
private:
    int chunksize, chunk;
    
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindA;
    ListTensorDesc listTensorDesc;
    
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
};

template<typename T>
class unpack_1 {
public:
    __aicore__ inline unpack_1() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int m, int n, int k)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ T *)input);
        this->listTensorDesc = listTensorDesc;
        
        pipe->InitBuffer(queBindA, 2, 32768);
        this->chunksize = 32768 / sizeof(T);
        this->m = m;
        this->n = n;
        this->k = k;
        this->chunk = m*k;
    }
    
    __aicore__ inline void Process(int offset, int size, int flag = 0)
    {
        int nk = n*k;
        if(k > chunksize) {
            int loop = (k + chunksize - 1) / chunksize;
            int tile = k % chunksize;
            tile = tile == 0 ? chunksize : tile;
            int start = flag ? 0 : size - 1;
            int end = flag ? size : -1;
            int step = flag ? 1 : -1;
            for(int i = start; i != end; i+=step) {
                int offsetA = int((offset + i) / m) * k + int((offset + i) % m) * nk;
                uint32_t num_id = (offset + i) / m;
                auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
                outputGm.SetGlobalBuffer(dataPtr);
                int base_offset = int((offset + i) % m) * k;
                for(int j = 0; j != loop - 1; j++) {
                    CopyIn(offsetA + j * chunksize, chunksize);
                    CopyOut(base_offset + j * chunksize, chunksize);
                }
                CopyIn(offsetA + (loop - 1) * chunksize, align_to(tile, 32));
                if(tile % 32) {
                    CopyOutPad(base_offset + (loop - 1) * chunksize, tile);
                } else {
                    CopyOut(base_offset + (loop - 1) * chunksize, tile);
                }
            }
        } else if(k * 2 > chunksize || k % 32) {
            int start = flag ? 0 : size - 1;
            int end = flag ? size : -1;
            int step = flag ? 1 : -1;
            for(int i = start; i != end; i+=step) {
                int offsetA = int((offset + i) / m) * k + int((offset + i) % m) * nk;
                uint32_t num_id = (offset + i) / m;
                auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
                outputGm.SetGlobalBuffer(dataPtr);
                int base_offset = int((offset + i) % m) * k;
                CopyIn(offsetA, align_to(k, 32));
                if(k % 32) {
                    CopyOutPad(base_offset, k);
                } else {
                    CopyOut(base_offset, k);
                }
            }
        } else {
            // ================= TODO 完成部分 =================
            // 场景：k 较小，单次搬运效率低。
            // 策略：一次搬入多行（batch 行），输入带跨度 nk，UB 内部连续排列。
            
            // 计算 UB 能容纳的最大行数 (确保 batch * k 不超过 chunksize)
            int max_batch = chunksize / k;
            if (max_batch > 65535) max_batch = 65535; // 硬件指令 repeatTimes 限制
            if (max_batch == 0) max_batch = 1;

            int i = 0;
            while (i < size) {
                // 计算当前任务在输出侧的归属
                int global_idx = offset + i;
                uint32_t num_id = global_idx / m;    // 属于第几个输出张量
                int m_idx = global_idx % m;          // 属于该张量的第几行
                
                // 计算本批次可以搬运的行数 (不能超过 max_batch，且不能跨越当前输出张量的 m 边界)
                int batch = min(max_batch, size - i);
                batch = min(batch, m - m_idx);
                
                // 1. 切换输出张量指针
                auto dataPtr = listTensorDesc.GetDataPtr<T>(num_id);        
                outputGm.SetGlobalBuffer(dataPtr);
                
                // 2. 计算地址偏移
                // Input 偏移：i_unstack * k + m_idx * (n * k)
                int offsetA = num_id * k + m_idx * nk;
                // Output 偏移：m_idx * k
                int base_offset = m_idx * k;
                
                // 3. 核心：调用 2D 搬运函数，将间隔的 k 向量拼成连续块
                // 这里的 total_len 是搬入 UB 后的总长度
                int total_len = batch * k;
                CopyInBatch(offsetA, batch, k, nk);
                
                // 4. 复用你原来的 CopyOut / CopyOutPad
                // 因为 UB 内部现在是连续的，所以直接搬出 total_len 长度即可
                if (total_len % 32 == 0) {
                    CopyOut(base_offset, total_len);
                } else {
                    CopyOutPad(base_offset, total_len);
                }
                
                i += batch;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int offset, int len)
    {
        LocalTensor<T> aLocal = queBindA.AllocTensor<T>();
        DataCopy(aLocal, inputGm[offset], len);
        //DumpTensor(inputGm[offset], offset, 16);
        queBindA.EnQue(aLocal);
    }

    __aicore__ inline void CopyOut(int offset, int len)
    {
        LocalTensor<T> cLocal = queBindA.DeQue<T>();
        //DumpTensor(cLocal, offset, 16);
        DataCopy(outputGm[offset], cLocal, len);
        queBindA.FreeTensor(cLocal);
        //DumpTensor(outputGm[offset], offset, 16);
    }
    
    __aicore__ inline void CopyOutPad(int offset, int len)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        LocalTensor<T> cLocal = queBindA.DeQue<T>();
        //DumpTensor(cLocal, offset, 16);
        DataCopyPad(outputGm[offset], cLocal, copyParams);
        queBindA.FreeTensor(cLocal);
        //DumpTensor(outputGm[offset], offset, 16);
    }

    __aicore__ inline void CopyInBatch(int offset, int batch, int k_len, int nk_stride)
    {
        LocalTensor<T> aLocal = queBindA.AllocTensor<T>();
        
        // 配置 2D 搬运参数
        // repeatTimes: 搬运多少行 (batch)
        // blockLen: 每行搬运的字节数 (k * sizeof(T))
        // srcStride: 输入侧每行结束到下一行开始的跳跃字节数 ((n-1) * k * sizeof(T))
        // dstStride: 输出侧（UB）每行结束到下一行开始的跳跃字节数 (0，表示在 UB 里紧凑排列)
        
        uint32_t block_bytes = k_len * sizeof(T);
        uint32_t stride_bytes = (nk_stride - k_len) * sizeof(T);
        
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(batch), 
            block_bytes, 
            stride_bytes, 
            0, // dstStride=0 关键点：实现 UB 内的连续挤压
            0
        };
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        
        // 使用 DataCopyPad 支持非 32B 对齐的跨度搬运
        DataCopyPad(aLocal, inputGm[offset], copyParams, padParams);
        
        queBindA.EnQue(aLocal);
    }
    
private:
    int chunksize, m, n, k, chunk;
    
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBindA;
    ListTensorDesc listTensorDesc;
    
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
};

template<typename T>
class unpack_2 {
public:
    __aicore__ inline unpack_2() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int m, int n)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ T *)input);
        this->listTensorDesc = listTensorDesc;
        this->m = m;
        this->n = n;
        
        // 核心设计：固定每次读取 16 行（完美契合 TransDataTo5HD 的 16 个源指针要求）
        // 列方向 TILE_N 设为 256（保证一次转置的块大小适中，且不会撑爆 UB）
        this->TILE_M = 16;
        this->TILE_N = 512;

        pipe->InitBuffer(inQueueSrc, 2, TILE_M * TILE_N * sizeof(T));
        pipe->InitBuffer(outQueueDst, 2, TILE_M * TILE_N * sizeof(T));
    }
    
    __aicore__ inline void Process(int offset, int size, int flag = 0)
    {
        int start_row = offset;
        int end_row = offset + size;

        // 外层循环：每次处理 M 维度上的 16 行
        for (int i = start_row; i < end_row; i += TILE_M) {
            int valid_rows = min(TILE_M, end_row - i);

            // 内层循环：按 TILE_N 切分 N 维度
            for (int j = 0; j < n; j += TILE_N) {
                int valid_cols = min(TILE_N, n - j);

                CopyIn(i, j, valid_rows, valid_cols);
                Compute(valid_rows, valid_cols);
                CopyOut(i, j, valid_rows, valid_cols);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int row_start, int col_start, int valid_rows, int valid_cols)
    {
        LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        int loop = valid_cols / block_size;
        DataCopyExtParams copyParams{uint16_t(valid_rows), uint32_t(valid_cols * sizeof(T)), uint32_t((n - valid_cols) * sizeof(T)), uint32_t(16 * sizeof(T) - align_to(valid_cols * sizeof(T), 32) / 32), 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        
        // 从 Global Memory 中按行读取 2D 数据块到 UB
        DataCopyPad(srcLocal, inputGm[row_start*n+col_start], copyParams, padParams); 
        // DumpTensor(srcLocal, 0, 16);
        inQueueSrc.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(int valid_rows, int valid_cols)
    {
        LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

        // ========== 核心转置魔法 ==========
        TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;

        int pad_cols = align_to(valid_cols, block_size); // NPU 要求 repeat 必须是 C0 的整数倍
        
        transDataParams.repeatTimes = pad_cols / block_size;
        transDataParams.dstRepStride = pad_cols <= block_size ? 0 : 16;
        transDataParams.srcRepStride = pad_cols <= block_size ? 0 : 1;

        uint64_t dstLocalList[16];
        uint64_t srcLocalList[16];

        if constexpr (sizeof(T) == 2) { 
            // 16-bit (half, bfloat16) 映射逻辑：1 datablock = 16 elements
            for (int i = 0; i < 16; i++) {
                dstLocalList[i] = (uint64_t)(dstLocal[i * 16].GetPhyAddr());
                srcLocalList[i] = (uint64_t)(srcLocal[i * TILE_N].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        } 
        else if constexpr (sizeof(T) == 4) { 
            // 32-bit (float, int32) 映射逻辑：1 datablock = 8 elements
            for (int i = 0; i < 16; i++) {
                // 32-bit 时，每两个指针拼接成输出的一行
                dstLocalList[i] = (uint64_t)(dstLocal[(i / 2) * 16 + (i % 2) * 8].GetPhyAddr());
                srcLocalList[i] = (uint64_t)(srcLocal[i * TILE_N].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        } 
        else {
            // 8-bit (int8, uint8, bool) 的防崩溃降级方案
            // 由于 8-bit 的 TransData 需要两轮（HighHalf/LowHalf）非常复杂，
            // 为了保证绝对的稳定性和泛化性，这里使用 UB 内的高速标量转置
            for (int r = 0; r < valid_rows; r++) {
                for (int c = 0; c < valid_cols; c++) {
                    dstLocal.SetValue(c * 16 + r, srcLocal.GetValue(r * TILE_N + c));
                }
            }
        }

        outQueueDst.EnQue(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(int row_start, int col_start, int valid_rows, int valid_cols)
    {
        LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        // DumpTensor(dstLocal, 1, 16);
        // 转置后，dstLocal 的形状变成了 [TILE_N, 16]
        // 也就是它的第 c 行，包含了我们要写入 Output[col_start + c] 的 valid_rows 个数据
        for (int c = 0; c < valid_cols; c++) {
            uint32_t out_idx = col_start + c;
            GlobalTensor<T> out_c;
            out_c.SetGlobalBuffer(listTensorDesc.GetDataPtr<T>(out_idx));

            // 将转置好的这一小段连续内存，写入对应的输出 Tensor 的正确行位置
            if(valid_rows % 16) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(valid_rows * sizeof(T)), 0, 0, 0};
                DataCopyPad(out_c[row_start], dstLocal[c * 16], copyParams);
            } else {
                DataCopy(out_c[row_start], dstLocal[c * 16], valid_rows);
            }
        }

        outQueueDst.FreeTensor(dstLocal);
    }
    
private:
    int m, n;
    int TILE_M, TILE_N;
    int block_size = 32 / sizeof(T);
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueueSrc;
    TQue<QuePosition::VECOUT, 1> outQueueDst;
    ListTensorDesc listTensorDesc;
    GlobalTensor<T> inputGm;
};

class unpack_21 {
public:
    __aicore__ inline unpack_21() {}
    
    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR input, ListTensorDesc listTensorDesc, int m, int n)
    {
        pipe = pipeIn;
        this->inputGm.SetGlobalBuffer((__gm__ int32_t *)input);
        this->listTensorDesc = listTensorDesc;
        this->m = m;
        this->n = n;
        
        // 核心设计：固定每次读取 16 行（完美契合 TransDataTo5HD 的 16 个源指针要求）
        // 列方向 TILE_N 设为 256（保证一次转置的块大小适中，且不会撑爆 UB）
        this->TILE_M = 400;
        this->TILE_N = 40;

        pipe->InitBuffer(inQueueSrc, 1, TILE_M * TILE_N * sizeof(int32_t));
        pipe->InitBuffer(outQueueDst, 1, TILE_M * TILE_N * sizeof(int32_t));
        pipe->InitBuffer(workBuf, TILE_M * TILE_N * sizeof(uint32_t));
    }
    
    __aicore__ inline void Process(int offset, int size, int flag = 0)
    {
        int start_row = offset;
        int end_row = offset + size;

        tempTensor = workBuf.Get<uint32_t>();
        LocalTensor<int32_t> interpreTensor = tempTensor.template ReinterpretCast<int32_t>();
        for(int i = 0; i != TILE_M; i++) {
            CreateVecIndex(interpreTensor[i*TILE_N], i*n, TILE_N);
        }
        Muls(interpreTensor, interpreTensor, 4, TILE_M * TILE_N);

        // 外层循环：每次处理 M 维度上的 16 行
        for (int i = start_row; i < end_row; i += TILE_M) {
            int valid_rows = min(int(TILE_M), end_row - i);

            CopyIn(i, valid_rows);
            Compute();
            CopyOut(i, valid_rows);
        }
    }

private:
    __aicore__ inline void CopyIn(int row_start, int valid_rows)
    {
        LocalTensor<int32_t> srcLocal = inQueueSrc.AllocTensor<int32_t>();
        
        // 从 Global Memory 中按行读取 2D 数据块到 UB
        DataCopy(srcLocal, inputGm[row_start*n], align_to(valid_rows*n, 32)); 
        // DumpTensor(srcLocal, 0, 16);
        inQueueSrc.EnQue(srcLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> srcLocal = inQueueSrc.DeQue<int32_t>();
        LocalTensor<int32_t> dstLocal = outQueueDst.AllocTensor<int32_t>();

        Gather(dstLocal, srcLocal, tempTensor, 0, TILE_M * TILE_N);
        // ========== 核心转置魔法 ==========
        TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false; 
        
        transDataParams.repeatTimes = TILE_M / 16;
        transDataParams.dstRepStride = 2;
        transDataParams.srcRepStride = 2 * TILE_N;

        uint64_t dstLocalList[16];
        uint64_t srcLocalList[16];

        for(int i = 0; i != (TILE_N / 8); i++) {
            for (int j = 0; j < 16; j++) {
                // 32-bit 时，每两个指针拼接成输出的一行
                dstLocalList[j] = (uint64_t)(srcLocal[(j / 2) * TILE_M + (j % 2) * 8 + i * 8 * TILE_M].GetPhyAddr());
                srcLocalList[j] = (uint64_t)(dstLocal[i * 8 + j * TILE_N].GetPhyAddr());
            }
            TransDataTo5HD<int32_t>(dstLocalList, srcLocalList, transDataParams);
        }
        Adds(dstLocal, srcLocal, 0, TILE_M * TILE_N);
        outQueueDst.EnQue(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(int row_start, int valid_rows)
    {
        LocalTensor<int32_t> dstLocal = outQueueDst.DeQue<int32_t>();

        for (int c = 0; c < n; c++) {
            uint32_t out_idx = c;
            GlobalTensor<int32_t> out_c;
            out_c.SetGlobalBuffer(listTensorDesc.GetDataPtr<int32_t>(out_idx));
            // if(GetBlockIdx() == 39) DumpTensor(out_c[9*16384*((c+1)%2)], c, 8);
            // 将转置好的这一小段连续内存，写入对应的输出 Tensor 的正确行位置
            if(valid_rows % 8) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(valid_rows * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(out_c[row_start], dstLocal[c * TILE_M], copyParams);
            } else {
                DataCopy(out_c[row_start], dstLocal[c*TILE_M], valid_rows);
            }
        }

        outQueueDst.FreeTensor(dstLocal);
    }
    
private:
    int m, n;
    uint32_t TILE_M, TILE_N;
    int block_size = 8;
    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueueSrc;
    TQue<QuePosition::VECOUT, 1> outQueueDst;
    TBuf<TPosition::VECCALC> workBuf;
    ListTensorDesc listTensorDesc;
    GlobalTensor<int32_t> inputGm;
    LocalTensor<uint32_t> tempTensor;
};

extern "C" __global__ __aicore__ void unpack(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    const uint32_t m = tiling_data.m;
    const uint32_t n = tiling_data.n;
    const uint32_t k = tiling_data.k;
    const uint32_t mode = tiling_data.mode;
    const uint32_t size = m*n*k;

    int blockIdx = GetBlockIdx();
    int blockDim = GetBlockNum();
    int start, len;
    if(mode == 1) {
        int cacheline = 128;
        int chunks = (size + cacheline - 1) / cacheline;
        int base_chunks = chunks / blockDim;
        int tile_chunks = chunks % blockDim;
        start = (base_chunks * blockIdx + min(blockIdx, tile_chunks)) * cacheline;
        len = min(int(size) - start, (base_chunks + (blockIdx < tile_chunks)) * cacheline); 
    } else if(mode == 3) {
        int chunks = m*n;
        int base_chunks = chunks / blockDim;
        int tile_chunks = chunks % blockDim;
        start = base_chunks * blockIdx + min(blockIdx, tile_chunks);
        len = base_chunks + (blockIdx < tile_chunks ? 1 : 0);
    } else if(mode == 2) {
        // mode=2 时，按 M（行数）均匀分摊给各个 NPU 核心
        int chunks = m;
        int base_chunks = chunks / blockDim;
        int tile_chunks = chunks % blockDim;
        start = base_chunks * blockIdx + min(blockIdx, tile_chunks);
        len = base_chunks + (blockIdx < tile_chunks ? 1 : 0);
    }
    TPipe pipe;

    ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void *>(output));

    if (TILING_KEY_IS(1)){
        if(mode == 1) {
            if(len <= 0) return;
            unpack_0<float> op;
            op.Init(&pipe, input, listTensorDesc, m*k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            op.Process(start, len, workGm(0)%2);
            if(blockIdx == 0) workGm(0) += 1;
        } else if(mode == 3) {
            if(len <= 0) return;
            unpack_1<float> op;
            op.Init(&pipe, input, listTensorDesc, m, n, k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            op.Process(start, len, workGm(0)%2);
            if(blockIdx == 0) workGm(0) += 1;
        } else {
            if(len <= 0) return;
            unpack_2<float> op;
            op.Init(&pipe, input, listTensorDesc, m, n);
            op.Process(start, len, 0);
        }
    }
    else if (TILING_KEY_IS(2)){
        if(mode == 1) {
            if(len <= 0) return;
            unpack_0<half> op;
            op.Init(&pipe, input, listTensorDesc, m*k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            // op.Process(start, len, 1);
            op.Process(start, len, 1);
            if(blockIdx == 0) workGm(0) += 1;
        } else if(mode == 3) {
            if(len <= 0) return;
            unpack_1<half> op;
            op.Init(&pipe, input, listTensorDesc, m, n, k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            op.Process(start, len, workGm(0)%2);
            if(blockIdx == 0) workGm(0) += 1;
        } else {
            if(len <= 0) return;
            unpack_2<half> op;
            op.Init(&pipe, input, listTensorDesc, m, n);
            op.Process(start, len, 0);
        }
    }
    else if (TILING_KEY_IS(3)){
        if(mode == 1) {
            if(len <= 0) return;
            unpack_0<uint8_t> op;
            op.Init(&pipe, input, listTensorDesc, m*k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            op.Process(start, len, workGm(0)%2);
            if(blockIdx == 0) workGm(0) += 1;
        } else if(mode == 3) {
            if(len <= 0) return;
            unpack_1<int8_t> op;
            op.Init(&pipe, input, listTensorDesc, m, n, k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            op.Process(start, len, workGm(0)%2);
            if(blockIdx == 0) workGm(0) += 1;
        } else {
            if(len <= 0) return;
            unpack_2<int8_t> op;
            op.Init(&pipe, input, listTensorDesc, m, n);
            op.Process(start, len, 0);
        }
    }
    else if (TILING_KEY_IS(4)){
        if(mode == 1) {
            if(len <= 0) return;
            unpack_0<bool> op;
            op.Init(&pipe, input, listTensorDesc, m*k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            op.Process(start, len, workGm(0)%2);
            if(blockIdx == 0) workGm(0) += 1;
        } else if(mode == 3) {
            if(len <= 0) return;
            unpack_1<bool> op;
            op.Init(&pipe, input, listTensorDesc, m, n, k);

            GlobalTensor<int32_t> workGm;
            workGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
            op.Process(start, len, workGm(0)%2);
            if(blockIdx == 0) workGm(0) += 1;
        } else {
            if(len <= 0) return;
            unpack_2<bool> op;
            op.Init(&pipe, input, listTensorDesc, m, n);
            op.Process(start, len, 0);
        }
    } else if (TILING_KEY_IS(5)){
        if(len <= 0) return;
        unpack_21 op;
        op.Init(&pipe, input, listTensorDesc, m, n);
        op.Process(start, len, 0);
    }
}