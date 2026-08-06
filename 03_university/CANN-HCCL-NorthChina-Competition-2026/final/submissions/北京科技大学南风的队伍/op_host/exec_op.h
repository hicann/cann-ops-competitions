/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// 北京科技大学南风的队伍：决赛 Host 侧执行接口声明。

#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include <hccl/hcomm_primitives.h>
#include "common.h"

namespace ops_hccl {
// 反序列化缓存资源，并分派到选定的执行路径。
HcclResult ExecOp(const OpParam &param);
}

#endif
