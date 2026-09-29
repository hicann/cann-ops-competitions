#include <vector>

#include <hcomm/hcomm_primitives.h>
#include <ccu/ccu_launch.h>

#include "ccu_kernel.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

namespace {
// 跨 rank 交换本端 output 地址 / token 所用的变量槽
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
// 中继用的中转区地址 / token 变量槽。只有在 relayEnabled 时才发布和等待，
// 关掉中继时前同步的 mask 与 base371 完全一致（不多一次变量交换）。
constexpr uint32_t STAGING_XN_ID = 4;
constexpr uint32_t STAGING_TOKEN_XN_ID = 5;
// 前同步与后同步共用的 CKE 序号，用不同的 mask 位区分
constexpr uint32_t CKE_IDX_0 = 0;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint16_t PRE_SYNC_MASK = static_cast<uint16_t>((1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID));
constexpr uint16_t POST_SYNC_MASK = static_cast<uint16_t>(1U << POST_SYNC_ID);
constexpr uint16_t STAGING_SYNC_MASK = static_cast<uint16_t>((1U << STAGING_XN_ID) | (1U << STAGING_TOKEN_XN_ID));
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->root >= kernelArg->rankSize) {
        return CCU_E_PARA;
    }
    if (kernelArg->rankSize > 1 && (kernelArg->channelCount == 0 || kernelArg->channelCount >= kernelArg->rankSize)) {
        return CCU_E_PARA;
    }

    const bool relayOn = (kernelArg->relayEnabled != 0);
    // 拉取式数据路径（小包专用）：注册期常量，分支在注册时就定死，不进微码
    const bool pullMode = (!relayOn && kernelArg->pullMode != 0);

    // 1. 初始化资源：在本 kernel 覆盖的每条 channel 上申请本端用于交换地址的变量槽
    std::vector<ccu::Variable> output(kernelArg->rankSize);
    std::vector<ccu::Variable> token(kernelArg->rankSize);
    std::vector<ccu::Variable> staging(kernelArg->rankSize);      // 对端的中转区基址
    std::vector<ccu::Variable> stagingToken(kernelArg->rankSize); // 对端的中转区 token
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        const uint32_t peer = kernelArg->peerRank[i];
        output[peer] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], OUTPUT_XN_ID);
        token[peer] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], TOKEN_XN_ID);
        if (relayOn) {
            staging[peer] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], STAGING_XN_ID);
            stagingToken[peer] = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], STAGING_TOKEN_XN_ID);
        }
    }

    // 数据面搬运完成事件
    ccu::Event transferEvent;
    // 中继投递单独用一个 event：好处有两个 ——
    //   1) 可以只等中继载荷就发「助手可以转了」的信号，不必等 root 的 mesh 阶段全部发完；
    //   2) 不会和 transferEvent 上的普通写入互相干扰（同一 event 重复 EventWait 的语义不确定）。
    ccu::Event relayEvent;

    // 2. 加载参数：顺序必须与 Host 侧 taskArgs 一致
    ccu::Variable inputAddress; // root 的 sendBuf 基址（已叠加本次分片偏移）
    ccu::Variable inputToken;   // root 的 sendBuf 对应 token
    ccu::Variable chunkBytes;   // 本次分片中每个 rank 传输的字节数
    ccu::Variable sliceStride;  // 每个 rank 完整切片的字节数（跨目的 rank 地址步进用）
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddress, argId++));
    CCU_CHK_RET(ccu::LoadArg(output[kernelArg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(token[kernelArg->rankId], argId++));
    CCU_CHK_RET(ccu::LoadArg(chunkBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(sliceStride, argId++));
    // 本端中转区（只有开中继时才读）。注意 output[kernelArg->rankId] 的写法：
    // 本 rank 不在自己的 peerRank 里，所以这个槽是空的，靠 LoadArg 灌入本地值 —— 中转区同理。
    // 直发段 / 中继段长度：每个被中继的对端都用同一组值。
    ccu::Variable prefixLen;
    ccu::Variable suffixLen;
    if (relayOn) {
        CCU_CHK_RET(ccu::LoadArg(staging[kernelArg->rankId], argId++));
        CCU_CHK_RET(ccu::LoadArg(stagingToken[kernelArg->rankId], argId++));
        CCU_CHK_RET(ccu::LoadArg(prefixLen, argId++));
        CCU_CHK_RET(ccu::LoadArg(suffixLen, argId++));
    }

    // 3. 前同步
    //    总原则：【等待方只等自己真的要写数据的那些对端】（等待缩小永远是安全的），
    //    发布方只在"确实没有读者"时才缩小 —— 这样不会出现"谁在等一个根本不存在的发布"。
    if (relayOn) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], output[kernelArg->rankId], OUTPUT_XN_ID,
                CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], token[kernelArg->rankId], TOKEN_XN_ID,
                CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
            if (kernelArg->relayIsHelper != 0) {
                // 中转区：root 要往助手的中转区里写、助手自己要从那里读，读者只有 root。
                CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], staging[kernelArg->rankId],
                    STAGING_XN_ID, CKE_IDX_0, static_cast<uint16_t>(1U << STAGING_XN_ID)));
                CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], stagingToken[kernelArg->rankId],
                    STAGING_TOKEN_XN_ID, CKE_IDX_0, static_cast<uint16_t>(1U << STAGING_TOKEN_XN_ID)));
            }
        }
        if (kernelArg->rankId == kernelArg->root) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                uint16_t mask = PRE_SYNC_MASK;
                for (uint32_t j = 0; j < kernelArg->relayCount; ++j) {
                    if (kernelArg->peerRank[i] == kernelArg->relayHelper[j]) {
                        mask = static_cast<uint16_t>(mask | STAGING_SYNC_MASK);
                        break;
                    }
                }
                CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, mask));
            }
        } else if (kernelArg->relayIsHelper != 0) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                for (uint32_t j = 0; j < kernelArg->relayCount; ++j) {
                    if (kernelArg->relayHelper[j] != kernelArg->rankId ||
                        kernelArg->relayDest[j] != kernelArg->peerRank[i]) {
                        continue;
                    }
                    CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, PRE_SYNC_MASK));
                    break;
                }
            }
        }
    } else if (pullMode) {
        if (kernelArg->rankId == kernelArg->root) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], inputAddress, OUTPUT_XN_ID,
                    CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
                CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], inputToken, TOKEN_XN_ID,
                    CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
            }
        } else {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->peerRank[i] != kernelArg->root) {
                    continue;
                }
                CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, PRE_SYNC_MASK));
                break;
            }
        }
    } else if (kernelArg->rankId == kernelArg->root) {
        // root 只等，不发布 —— 数据面只有 root 写，没有任何人需要 root 的 output 地址。
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, PRE_SYNC_MASK));
        }
    } else {
        // 非 root 只发布给 root。只在本 kernel 里找 root 的那一条 channel：
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (kernelArg->peerRank[i] != kernelArg->root) {
                continue;
            }
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], output[kernelArg->rankId], OUTPUT_XN_ID,
                CKE_IDX_0, static_cast<uint16_t>(1U << OUTPUT_XN_ID)));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(kernelArg->channels[i], token[kernelArg->rankId], TOKEN_XN_ID,
                CKE_IDX_0, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
            break;
        }
    }

    // 4'. 拉取式数据面：每个 rank（含 root 自己）把 root sendBuf 里属于自己那片弄到自己的 recvBuf。
    if (pullMode) {
        const uint16_t myBit = 1;
        ccu::LocalAddr dst;
        dst.addr = output[kernelArg->rankId];
        dst.token = token[kernelArg->rankId];

        if (kernelArg->rankId != kernelArg->root) {
            uint32_t rootIdx = kernelArg->channelCount;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->peerRank[i] == kernelArg->root) {
                    rootIdx = i;
                    break;
                }
            }
            if (rootIdx >= kernelArg->channelCount) {
                // root 不在这颗 kernel 的对端里（channel 按 die 分组）：本 rank 的搬运由
                // 另一颗含 root 的 kernel 承担，这一颗无事可做。
                return CCU_SUCCESS;
            }
            // output[root]/token[root] 取到的正是【root 发布在它自己槽里的值】= root 的 sendBuf 基址与 token
            ccu::RemoteAddr src;
            src.addr = output[kernelArg->root];
            src.token = token[kernelArg->root];
            for (uint32_t k = 0; k < kernelArg->rankId; ++k) {
                src.addr += sliceStride;
            }
            CCU_CHK_RET(ccu::Read(kernelArg->channels[rootIdx], dst, src, chunkBytes, transferEvent, myBit));
        } else {
            // root 那片在本地：sendBuf -> recvBuf 的本地拷贝，偏移同样是 root * sliceStride。
            if (kernelArg->doSelfCopy == 0) {
                return CCU_SUCCESS;
            }
            ccu::LocalAddr src;
            src.addr = inputAddress;
            src.token = inputToken;
            for (uint32_t k = 0; k < kernelArg->rankId; ++k) {
                src.addr += sliceStride;
            }
            CCU_CHK_RET(ccu::LocalCopy(dst, src, chunkBytes, transferEvent, myBit));
        }
        CCU_CHK_RET(ccu::EventWait(transferEvent, myBit));
        return CCU_SUCCESS;
    }

    // 4. 执行算子数据面传输：仅 root 向各 rank 写入属于它的切片
    //    只等本 kernel 实际置位的 event bit（peer bit + 可能的本端/中继 bit），避免死等别的 kernel 的 bit
    uint16_t waitMask = 0;
    uint32_t remoteChannelIdx = 0;
    if (kernelArg->rankId == kernelArg->root) {
        ccu::LocalAddr src;
        src.addr = inputAddress;
        src.token = inputToken;

        // (a) 中继载荷【先发】。这是让第二跳和 root 的 mesh 阶段重叠的关键：
        //     早发 → 早等到 → 早通知助手 → 助手的 Clos kernel 早开始转发。
        uint16_t relayMask = 0;
        for (uint32_t i = 0; relayOn && i < kernelArg->relayCount; ++i) {
            const uint32_t helper = kernelArg->relayHelper[i];
            const uint32_t dest = kernelArg->relayDest[i];
            uint32_t helperIdx = kernelArg->channelCount;
            for (uint32_t c = 0; c < kernelArg->channelCount; ++c) {
                if (kernelArg->peerRank[c] == helper) {
                    helperIdx = c;
                    break;
                }
            }
            if (helperIdx >= kernelArg->channelCount) {
                continue; // 助手不在本 kernel 的对端里 —— 这颗不是 mesh 组
            }
            ccu::RemoteAddr relDst;
            relDst.addr = staging[helper];
            relDst.token = stagingToken[helper];
            ccu::LocalAddr relSrc;
            relSrc.addr = inputAddress;
            for (uint32_t k = 0; k < dest; ++k) {
                relSrc.addr += sliceStride; // ccu::Variable 不支持乘法，只能累加
            }
            relSrc.token = inputToken;
            // 只把【后缀】那一段转发给助手，前缀由 root 自己直发 Clos。
            // 每个被中继的对端都是一样的：源地址后移 prefixLen、长度取 suffixLen。
            relSrc.addr += prefixLen;
            const uint16_t relMask = static_cast<uint16_t>(1U << dest);
            CCU_CHK_RET(
                ccu::Write(kernelArg->channels[helperIdx], relDst, relSrc, suffixLen, relayEvent, relMask));
            relayMask |= relMask;
        }
        // (b) 等到中继载荷落地，立刻给助手发「可以转了」。
        //     助手那颗 mesh kernel 的 post-sync 等的就是这一条记录，所以它会尽早返回，
        //     它的 Clos kernel 也就能尽早开始第二跳 —— 设备侧仍然不需要任何跨 die 通知。
        if (relayMask != 0) {
            CCU_CHK_RET(ccu::EventWait(relayEvent, relayMask));
            for (uint32_t i = 0; i < kernelArg->relayCount; ++i) {
                for (uint32_t c = 0; c < kernelArg->channelCount; ++c) {
                    if (kernelArg->peerRank[c] == kernelArg->relayHelper[i]) {
                        CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[c], CKE_IDX_0, POST_SYNC_MASK));
                        break;
                    }
                }
            }
        }

        // (c) 再走正常的逐对端写入
        for (uint32_t dstRank = 0; dstRank < kernelArg->rankSize; ++dstRank) {
            const uint16_t dstMask = static_cast<uint16_t>(1U << dstRank);
            if (dstRank == kernelArg->rankId) {
                if (kernelArg->doSelfCopy) {
                    ccu::LocalAddr dst;
                    dst.addr = output[dstRank];
                    dst.token = token[dstRank];
                    CCU_CHK_RET(ccu::LocalCopy(dst, src, chunkBytes, transferEvent, dstMask));
                    waitMask |= dstMask;
                }
            } else if (remoteChannelIdx < kernelArg->channelCount &&
                kernelArg->peerRank[remoteChannelIdx] == dstRank) {
                bool relayed = false;
                if (relayOn) {
                    for (uint32_t i = 0; i < kernelArg->relayCount; ++i) {
                        if (kernelArg->relayDest[i] == dstRank) {
                            relayed = true;
                            break;
                        }
                    }
                }
                ccu::RemoteAddr dst;
                dst.addr = output[dstRank];
                dst.token = token[dstRank];
                if (relayed) {
                    CCU_CHK_RET(ccu::Write(kernelArg->channels[remoteChannelIdx], dst, src, prefixLen,
                        transferEvent, dstMask));
                } else {
                    CCU_CHK_RET(ccu::Write(kernelArg->channels[remoteChannelIdx], dst, src, chunkBytes,
                        transferEvent, dstMask));
                }
                waitMask |= dstMask;
                ++remoteChannelIdx;
            }
            src.addr += sliceStride; // 步进到下一个 rank 的切片
        }
    }

    // 助手的转发：从中转区取出被中继那一片，用本 kernel（Clos 组）转给目的端。
    for (uint32_t i = 0; relayOn && kernelArg->relayIsHelper && i < kernelArg->relayCount; ++i) {
        // 中转区里只有本 rank 负责的那几片；一人一片，所以槽位固定从 0 开始
        if (kernelArg->relayHelper[i] != kernelArg->rankId) {
            continue;
        }
        const uint32_t dest = kernelArg->relayDest[i];
        uint32_t destIdx = kernelArg->channelCount;
        for (uint32_t c = 0; c < kernelArg->channelCount; ++c) {
            if (kernelArg->peerRank[c] == dest) {
                destIdx = c;
                break;
            }
        }
        if (destIdx < kernelArg->channelCount) {
            ccu::RemoteAddr fwdDst;
            fwdDst.addr = output[dest];
            fwdDst.token = token[dest];
            ccu::LocalAddr fwdSrc;
            fwdSrc.addr = staging[kernelArg->rankId];
            fwdSrc.token = stagingToken[kernelArg->rankId];
            // 中转区里放的是【后缀】，且是【从偏移 0 开始】放的（root 写的时候
            // relDst.addr = staging[helper]，没有加偏移；加偏移的是源地址 relSrc）。
            // 所以这里只把【目的端】的落点后移 prefixLen，源地址不动。
            fwdDst.addr += prefixLen;
            const uint16_t fwdMask = static_cast<uint16_t>(1U << dest);
            CCU_CHK_RET(
                ccu::Write(kernelArg->channels[destIdx], fwdDst, fwdSrc, suffixLen, transferEvent, fwdMask));
            waitMask |= fwdMask;
        }
    }

    // 等待本 rank 在本 kernel 发起的全部搬运完成
    if (waitMask != 0) {
        CCU_CHK_RET(ccu::EventWait(transferEvent, waitMask));
    }

    // 5. 后同步：确保数据真的落到接收方之后再退出
    if (relayOn) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            // 助手的 channel 在上面 (b) 已经提前记过了 —— 不能重复记，也不能等它回头，
            // 那条记录就是让助手的 mesh kernel 尽早返回用的。
            bool alreadyRecorded = false;
            if (kernelArg->rankId == kernelArg->root) {
                for (uint32_t j = 0; j < kernelArg->relayCount; ++j) {
                    if (kernelArg->peerRank[i] == kernelArg->relayHelper[j]) {
                        alreadyRecorded = true;
                        break;
                    }
                }
            }
            if (!alreadyRecorded) {
                CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, POST_SYNC_MASK));
            }
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, POST_SYNC_MASK));
        }
    } else if (kernelArg->rankId == kernelArg->root) {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(kernelArg->channels[i], CKE_IDX_0, POST_SYNC_MASK));
        }
    } else {
        // root 不在这颗 kernel 的对端里，说明本 kernel 没有任何来自 root 的搬运
        // （root 一定在本 rank 的另一颗 kernel 里，那颗会等），这里就不必等。
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (kernelArg->peerRank[i] == kernelArg->root) {
                CCU_CHK_RET(ccu::NotifyWait(kernelArg->channels[i], CKE_IDX_0, POST_SYNC_MASK));
                break;
            }
        }
    }

    return CCU_SUCCESS;
}

// TODO: 可编写多个 CCU Kernel 函数，以最大化性能
// CcuResult CcuKernel2(CcuKernelArg arg)
// {
//     return CCU_SUCCESS;
// }

} // namespace ops_hccl
