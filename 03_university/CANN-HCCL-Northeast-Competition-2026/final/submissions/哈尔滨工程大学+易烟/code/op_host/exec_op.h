/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include "common.h"

namespace ops_hccl {
// Small and 4x1 use direct receiver Reads. One actual-die group can fuse a safe
// nonaliasing root copy while all groups retain stream start/join ordering.
// Other large topologies use explicit frame ranges held in per-channel XNs.
// Each 2x8/4x3 and 8+4/root4 helper supplies one distinct target's prefix.
// Targets accept authorized suppliers on their root-channel die; root fills gaps.
// V16: 2x8 receivers specialize the unique prefix supplier after root authorization.
// V17: a helper with root/target channels on different actual dies launches its
// own root-die Pull on the worker while main sends READY, then drains target-die
// offers. Join precedes the single ACK; other helpers retain the original order.
// Confirmed 4x1 uses compact direct receivers for Small and all Large frames;
// its compact root-copy remains exclusive to the safe Small specialization.
// Confirmed 4x3 retains the compact Small construction. Eligible 2x8 helpers
// replace capture/offers/prefetch with two jointly registered HP8 prepare
// kernels, even for zero supply. The root-die group alone consumes META.
// V20 queues a nonzero helper's root-die Pull immediately after preparation;
// PREP_DONE is consumed before START2_ACK, and FINISHED follows that ACK.
// The two plans, actual dies, frames and arguments are checked before enqueue.
// Eligible 8+4/root8 roots use two RPH9 publish/copy/wait kernels and a join;
// group 0 alone may copy, while no-copy/alias cases keep the RPH9 contract.
// 8+4/root8 fuses root capture, zero-offer publication and Pull on both actual dies;
// Host must not consume root META or publish offers before that fused entry.
// Every original META/READY and final release is still consumed.
// Root8 zero offers send BYTES=0 and the original complete META mask;
// all N12 Large receivers capture BYTES before reading remaining fields.
// Prefix offers precede other-offer waits; READY follows completed preparation.
// Archived 8+4 bands retain post-prefetch META without READY.
// Large root copy overlaps remote Reads only when its output is disjoint from
// the entire input range. Alias cases retain the original completion order.
// Root input and helper scratch remain valid through all-rank DONE and RELEASE.
// Completion is enqueued on param.cpuThread, which belongs to the current stream.
HcclResult ExecOp(const OpParam &param);
} // namespace ops_hccl
#endif // OPS_HCCL_CCU_EXEC_OP_H
