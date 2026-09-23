/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO ANY DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

#include "common.h"
#include "custom.h"

namespace ops_hccl {

// Scatter CCU Kernel: root publishes its input capability once per IO Die;
// receivers pull their distinct slice concurrently and acknowledge completion.
// 注册期参数见 ScatterCcuKernelArg（custom.h）；每次调用的动态参数见 exec_op.cc 的 taskArgs 布局
CcuResult CcuScatterKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayRead(CcuKernelArg arg);
CcuResult CcuScatterDirectPush(CcuKernelArg arg);
CcuResult CcuScatterResidentReceiver(CcuKernelArg arg);
CcuResult CcuScatterCoarseRelay(CcuKernelArg arg);
CcuResult CcuScatterHybridRoot(CcuKernelArg arg);
CcuResult CcuScatterHybridReceiver(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
