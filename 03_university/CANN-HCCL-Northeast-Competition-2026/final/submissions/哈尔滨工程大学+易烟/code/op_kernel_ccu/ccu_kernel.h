/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

namespace ops_hccl {

// Each group registration contains channels from one local IO die only.
CcuResult CcuSmallPullRootKernel(CcuKernelArg arg);
// Compact Direct variants: publish input/token, optionally overlap a safe copy.
CcuResult CcuDirectRootKernel(CcuKernelArg arg);
CcuResult CcuDirectRootCopyKernel(CcuKernelArg arg);
CcuResult CcuSmallPullPeerKernel(CcuKernelArg arg);
// V17 N4 variants retain META/DONE and the original argument counts.
// Peer receives every positive Direct frame (Small and Large); root copy is Small only.
CcuResult CcuFourByOnePullPeerKernel(CcuKernelArg arg);
CcuResult CcuFourByOneDirectRootCopyKernel(CcuKernelArg arg);
// V18: the same compact instructions under a confirmed 4x3 Small registration.
CcuResult CcuFourByThreePullPeerKernel(CcuKernelArg arg);
CcuResult CcuFourByThreeDirectRootCopyKernel(CcuKernelArg arg);
CcuResult CcuRootPublishKernel(CcuKernelArg arg);
// V18: one actual-die group publishes ROOT4, optionally copies, then waits DONE.
// Both groups use RPH9; Host joins them before the unchanged root release.
CcuResult CcuEightPlusFourRootPhaseKernel(CcuKernelArg arg);
CcuResult CcuRootWaitKernel(CcuKernelArg arg);
CcuResult CcuRootReleaseKernel(CcuKernelArg arg);
// Waits for root metadata in channel XNs; no task arguments or HBM control page.
CcuResult CcuRootCaptureKernel(CcuKernelArg arg);
CcuResult CcuPrefetchKernel(CcuKernelArg arg); // One nonzero supplier piece per launch.
// Publish one complete descriptor; early layouts notify readiness separately.
CcuResult CcuOfferSendKernel(CcuKernelArg arg);
// Publish every nonroot descriptor on one die; at most one nonzero prefix.
CcuResult CcuOfferGroupKernel(CcuKernelArg arg);
// V18 HP8: prepare a 2x8 helper's prefix while publishing full descriptors.
// Only the root-channel group consumes root META and issues the prefetch.
CcuResult CcuTwoByEightPrepareKernel(CcuKernelArg arg);
// Zero arguments: finish preparation for one nonzero early descriptor.
CcuResult CcuOfferReadyKernel(CcuKernelArg arg);
// Consume META and promised READY; only the root-channel group accepts payloads.
// V20: every N12 offer capture reads BYTES first and locally normalizes zero offers.
CcuResult CcuPullGroupKernel(CcuKernelArg arg);
// Retained V14A builder: early Direct after Host Capture. V16 registers the
// fused root8 entry below instead; both authorize Read using the root header.
CcuResult CcuEightPlusFourPullGroupKernel(CcuKernelArg arg);
// V16: root-channel die only; PrefixRegArg carries the expected eight-rank mask.
// Root META is consumed by Host's capture or V18 preparation before this receiver.
CcuResult CcuTwoByEightPrefixPullGroupKernel(CcuKernelArg arg);
// V20: both dies for 8+4/root8; consumes root META on its owning die and
// publishes BYTES=0 with full META before draining peers. Host skips the old setup.
CcuResult CcuEightPlusFourFusedPullGroupKernel(CcuKernelArg arg);
// After Host joins both dies: acknowledge root, then wait for collective release.
CcuResult CcuRootAckKernel(CcuKernelArg arg);
// Host supplies one bounded chunk and its overlap-safe copy order.
CcuResult CcuCopyKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
