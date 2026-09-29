#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_primitives.hpp>
#include <ccu/ccu_types.h>

#include "custom.h"
#include "log.h"

/* 检查 CCU 接口返回值：失败时记录日志并返回该 CcuResult。
 *
 * 说明：工程自带的 include/log.h（不可修改）只提供了面向 Host 侧的 CHK_RET_CCU
 * （失败时返回 HcclResult），而 CCU Kernel 函数的返回值类型是 CcuResult，
 * 因此在这里补充定义 CCU Kernel 侧使用的 CCU_CHK_RET。 */
#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call)                                                  \
    do {                                                                   \
        CcuResult ccuRet = (call);                                         \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) {                             \
            HCCL_ERROR("[%s] call trace: ccuRet -> %d", __func__, ccuRet); \
            return ccuRet;                                                 \
        }                                                                  \
    } while (0)
#endif

namespace ops_hccl {

// CCU Kernel 注册期参数：Host 侧注册 Kernel 时确定，各 rank 取值可能不同
struct CcuKernelArgScatter : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t root = 0;
    uint32_t peerRank[MAX_RANK_SIZE] = {0}; // 本 kernel 负责的远端 rank，按 rank 升序
    uint32_t doSelfCopy = 0;                // 是否由本 kernel 承担 root 自拷贝（仅一个 kernel 置 1）

    // 本 kernel 落在哪个 die：0 = Clos 组（跨服对端），1 = mesh 组（本服对端）。
    // 注册期常量，分支在注册时就定死，不进微码。
    uint32_t axisId = 0;

    // ---- 中继计划（全 rank 同一套规则算出，见 scatter.cc）----
    // 开启后，对每一片被中继的跨服数据：root 的 mesh kernel 把它额外写进对应助手的

    uint32_t relayEnabled = 0;
    uint32_t relayCount = 0;
    uint32_t relayDest[RELAY_DEST_MAX] = {0};   // 第 i 片的目的端
    uint32_t relayHelper[RELAY_DEST_MAX] = {0}; // 第 i 片的助手
    uint32_t relayIsHelper = 0;                 // 本 rank 是否承担某一（几）片的中继
    // 中继量均摊：【每个】被中继的对端都按同一个比例 r = relayFracNum/relayFracDen 切开 ——
    // [0, prefixLen) 由 root 直发 Clos，[prefixLen, chunkBytes) 走 mesh → 助手 → Clos。
    // 这样每条 mesh 链路的负载才是均匀的 S(1+r)。都是注册期常量，不进微码。
    uint32_t relayFracNum = 0;
    uint32_t relayFracDen = 1;

    // ---- 拉取式数据路径（小包专用）----
    // 置 1 时数据面改成「各 rank 自己去 root 的 sendBuf 里读自己那片」：
    uint32_t pullMode = 0;
};

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);

// TODO: 可编写多个 CCU Kernel 函数，以最大化性能
// CcuResult CcuKernel2(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
