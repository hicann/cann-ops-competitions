/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of this repository.
 */
#include <algorithm>
#include <vector>
#include "ccu_kernel.h"
#include "custom.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;
#define CCU_TRY(call) do { CcuResult result_ = (call); if (result_ != CCU_SUCCESS) return result_; } while (0)
namespace {
constexpr uint64_t SMALL = CCU_SMALL_BYTES;
constexpr uint64_t MIN_SCRATCH = CCU_MIN_SCRATCH;
// Clos port / Mesh link bandwidth, in 1/100, for the relay byte balance.
constexpr uint64_t Q100 = 398;
// Notify slots on the receiving side of each channel. Every (direction, slot,
// bit) is recorded at most once per invocation and consumed in that invocation.
constexpr uint32_t META = 0;   // root->reader input; helper->root buffers; far->helper output
constexpr uint32_t READY = 1;  // root->helper: relayed tail complete in helper scratch
constexpr uint32_t DONE = 2;   // root->helper: helper output complete; helper->far: tail written
constexpr uint32_t ACK = 3;    // reader->root: every read from root input complete
constexpr uint32_t STAGE = 4;  // reader->root: reader has entered this call
// Staged metadata lives on its own notify and its own variable slots (a channel
// has eight). Root does not wait for a staged reader, so the very next call --
// which reuses slots 0/1 and META for its own addresses -- could otherwise
// overwrite the address a staged reader has not picked up yet.
constexpr uint32_t STAGE_META = 5;
constexpr uint16_t LOCAL_BIT = 0x8000;  // root local copy event bit (helper bits are low)
// Kernel program rules (CCU graph semantics): consecutive async transfers form
// one parallel group; the next wait joins the whole group; a record placed
// directly after a transfer fires at that transfer's completion. Hence every
// record below is issued only after an explicit wait of the group it reports,
// and no wait follows a group it is not meant to join.

struct Route {
    uint32_t helper, target;
    uint64_t length;  // tail bytes [B-length, B) of the target block relayed by helper
};
struct Plan {
    std::vector<uint32_t> near, far;
    std::vector<Route> routes;
};
Plan MakePlan(const CcuKernelArgBase &a)
{
    Plan p;
    std::vector<uint32_t> helpers;
    for (uint32_t r = 0; r < a.rankSize; ++r) {
        if (r == a.root) continue;
        ((a.nearMask & (1U << r)) ? p.near : p.far).push_back(r);
        if (a.relayMask & (1U << r)) helpers.push_back(r);
    }
    const uint64_t h = helpers.size(), f = p.far.size(), length = a.bytes;
    const uint64_t capacity = std::min<uint64_t>(a.scratchCapacity, MAX_DATA_SIZE) -
                              (CcuStageReserved(a.scratchCapacity) ? CCU_STAGE_RESERVE : 0);
    // f <= 4 stays out of relaying entirely. A root with more Mesh peers than
    // Clos peers (8+4 from the eight-card side) was measured on the platform at
    // 1830/1430us relayed against 923/723us direct -- the fluid balance below
    // does not describe that shape, so it is not applied to it.
    if (a.rankSize <= 5 || a.bytes <= SMALL || a.bytes > MAX_DATA_SIZE || capacity < MIN_SCRATCH || h == 0 ||
        f <= 4)
        return p;
    // Q100/100 ~ per-NPU Clos bandwidth over one intra-Server Mesh link. Each
    // helper relays one distinct far block tail; root sends the rest over Clos.
    // Root's Clos port carries f*B - routes*x and its busiest Mesh link B + x.
    if (h == 3 && f == 8) {
        for (uint32_t i = 0; i < h; ++i) {
            const uint64_t fanout = (f + h - 1 - i) / h;
            const uint64_t balanced = ((length / 4) * (100 * f - Q100) / ((100 * h + Q100) * fanout)) * 4;
            const uint64_t bounded = (capacity / fanout / 4) * 4;
            const uint64_t tail = std::min(balanced, bounded) * fanout;
            if (tail > 0 && tail < length && tail <= capacity) p.routes.push_back({helpers[i], p.far[i], tail});
        }
    } else if ((h == 7 && f == 8) || (h == 2 && f == 9)) {
        const uint64_t balanced = ((length / 4) * (100 * f - Q100) / (100 * h + Q100)) * 4;
        const uint64_t tail = std::min(balanced, (capacity / 4) * 4);
        if (tail > 0 && tail < length)
            for (uint32_t i = 0; i < h; ++i) p.routes.push_back({helpers[i], p.far[i], tail});
    }
    return p;
}
// Every rank derives the same wire protocol from the shared call parameters.
bool Uniform(const CcuKernelArgBase &a)
{
    return a.rankSize > 5 && a.bytes > SMALL && a.bytes <= MAX_DATA_SIZE;
}
bool Staged(const CcuKernelArgBase &a)
{
    return CcuStaged(a.bytes, a.rankSize, a.scratchCapacity);
}
// Every receiver of a small call, Mesh or Clos, reads for itself and sends
// nothing back: N readers start N reads in parallel, where root would start
// its writes one after another and then owe each receiver a completion notify.
bool StagedReader(const CcuKernelArgBase &a, uint32_t rank)
{
    (void)rank;
    return Staged(a);
}
struct Context {
    const CcuKernelArgBase &a;
    ccu::Variable input, output, inputToken, outputToken, scratch, scratchToken;
    ccu::Variable offset, size, immediate, pairs, gen;
    ccu::LocalAddr localSrc, localDst;
    ccu::RemoteAddr remote;
    ccu::Event directEvent, relayEvent;
    explicit Context(const CcuKernelArgBase &arg) : a(arg) {}
    ChannelHandle Channel(uint32_t r) const { return a.channels[r]; }
    bool Own(uint32_t r) const { return a.channelDie[r] == a.die; }
};

const Route *RouteOfHelper(const Plan &p, uint32_t helper)
{
    for (const auto &route : p.routes) if (route.helper == helper) return &route;
    return nullptr;
}
const Route *RouteOfTarget(const Plan &p, uint32_t target)
{
    for (const auto &route : p.routes) if (route.target == target) return &route;
    return nullptr;
}
// Root writes a receiver's block instead of the receiver reading it: the
// receiver publishes its output once and waits for a single notify, and root
// needs neither an address broadcast nor a read acknowledgement. Every rank
// decides this the same way from shared call parameters: the Mesh receivers
// that relay nothing. A far rank sees nearMask == 0, which is the same answer
// root gives it, and a Clos receiver is always faster reading for itself --
// several readers pull in parallel where root would write them one at a time.
bool Pushed(const CcuKernelArgBase &a, uint32_t rank, const Plan &p)
{
    if (!(a.nearMask & (1U << rank)) || Staged(a)) return false;
    return a.bytes <= SMALL || (Uniform(a) && RouteOfHelper(p, rank) == nullptr);
}
// Move localSrc from absolute input offset `last` to `from` with the fewest instructions.
void Seek(Context &c, uint64_t &last, uint64_t from)
{
    if (from == last) return;
    if (from > last) {
        c.immediate = from - last;
    } else {
        c.localSrc.addr = c.input;
        if (from == 0) { last = 0; return; }
        c.immediate = from;
    }
    c.localSrc.addr += c.immediate;
    last = from;
}

// ---------------------------------------------------------------- root
// Root's own block. With `wait` false a single-block copy is left for the
// following transfer group's wait; a chunked copy always waits for itself.
CcuResult RootLocalCopy(Context &c, bool wait)
{
    const auto &a = c.a;
    const uint64_t rounds = a.bytes / MAX_DATA_SIZE, tail = a.bytes % MAX_DATA_SIZE;
    uint64_t last = 0;
    c.localSrc.addr = c.input; c.localSrc.token = c.inputToken;
    Seek(c, last, static_cast<uint64_t>(a.root) * a.bytes);
    c.localDst.addr = c.output; c.localDst.token = c.outputToken;
    if (rounds) {
        c.size = MAX_DATA_SIZE; c.immediate = MAX_DATA_SIZE; c.pairs = rounds; c.offset = UINT64_MAX;
        CCU_WHILE(c.pairs != 0) {
            CCU_TRY(ccu::LocalCopy(c.localDst, c.localSrc, c.size, c.directEvent, LOCAL_BIT));
            CCU_TRY(ccu::EventWait(c.directEvent, LOCAL_BIT));
            c.localSrc.addr += c.immediate; c.localDst.addr += c.immediate;
            c.pairs += c.offset;
        }
    }
    if (tail) {
        c.size = tail;
        CCU_TRY(ccu::LocalCopy(c.localDst, c.localSrc, c.size, c.directEvent, LOCAL_BIT));
        if (wait || rounds) CCU_TRY(ccu::EventWait(c.directEvent, LOCAL_BIT));
    }
    return CCU_SUCCESS;
}
CcuResult RootKernel(Context &c)
{
    const auto &a = c.a;
    const bool uniform = Uniform(a);
    const Plan p = uniform ? MakePlan(a) : Plan{};
    // 1. Readers on this die learn root's input first (standalone records):
    //    Mesh readers, then far readers with the largest root flows. A far
    //    reader of a Uniform call learns its relay helper (rankSize = none) and,
    //    only when served, the length of its root part.
    std::vector<uint32_t> readers;
    for (int pass = 0; pass < 3; ++pass) {
        for (uint32_t r = 0; r < a.rankSize; ++r) {
            if (r == a.root || !c.Own(r) || RouteOfHelper(p, r) != nullptr || Pushed(a, r, p)) continue;
            const bool far = uniform && !(a.nearMask & (1U << r));
            const Route *route = far ? RouteOfTarget(p, r) : nullptr;
            if (pass != (!far ? 0 : (route == nullptr ? 1 : 2))) continue;
            if (StagedReader(a, r)) {
                // Published below, in one generation branch for all of them.
            } else {
                CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(r), c.input, 0, META, 1));
                CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(r), c.inputToken, 1, META, 2));
            }
            if (far) {
                c.immediate = route ? route->helper : a.rankSize;
                CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(r), c.immediate, 2, META, 4));
                if (route) {
                    c.immediate = a.bytes - route->length;
                    CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(r), c.immediate, 3, META, 8));
                }
            }
            readers.push_back(r);
        }
    }
    if (!readers.empty() && StagedReader(a, readers[0])) {
        // Consecutive calls must not share notify bits or variable slots: root
        // does not wait for its readers, so it can publish the next call before
        // a reader has consumed this one, and two records on one bit merge into
        // one. Alternating generations keeps them apart; the entry rendezvous
        // at the end of this kernel keeps root within one call of every reader.
        // One branch around the whole run: two instructions per reader.
        CCU_IF(c.gen != 0) {
            for (uint32_t r : readers) {
                CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(r), c.input, 6, STAGE_META, 8));
            }
        }
        CCU_ELSE {
            for (uint32_t r : readers) {
                CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(r), c.input, 4, STAGE_META, 1));
            }
        }
    }
    // 2. Pushed receivers publish their output once; root collects it here.
    std::vector<uint32_t> pushed;
    std::vector<ccu::Variable> pOut, pTok;
    for (uint32_t r = 0; r < a.rankSize; ++r) {
        if (r == a.root || !c.Own(r) || !Pushed(a, r, p)) continue;
        CCU_TRY(ccu::NotifyWait(c.Channel(r), META, 3));
        pOut.push_back(ccu::GetResByChannel<ccu::Variable>(c.Channel(r), 0));
        pTok.push_back(ccu::GetResByChannel<ccu::Variable>(c.Channel(r), 1));
        pushed.push_back(r);
    }
    std::vector<const Route *> helpers;
    for (const auto &route : p.routes) if (c.Own(route.helper)) helpers.push_back(&route);
    const bool local = a.die == 0;
    // True only when the local copy is issued without consuming its own event.
    const bool localPending = local && a.bytes < MAX_DATA_SIZE;
    if (helpers.empty()) {
        if (local && pushed.empty()) CCU_TRY(RootLocalCopy(c, true));
    } else {
        // 2. Relay helpers on this die: Mesh carries tail then own block.
        std::vector<ccu::Variable> hOut, hOutTok, hScr, hScrTok;
        for (const Route *route : helpers) {
            const auto channel = c.Channel(route->helper);
            CCU_TRY(ccu::NotifyWait(channel, META, 15));
            hOut.push_back(ccu::GetResByChannel<ccu::Variable>(channel, 0));
            hOutTok.push_back(ccu::GetResByChannel<ccu::Variable>(channel, 1));
            hScr.push_back(ccu::GetResByChannel<ccu::Variable>(channel, 2));
            hScrTok.push_back(ccu::GetResByChannel<ccu::Variable>(channel, 3));
        }
        const uint16_t bits = static_cast<uint16_t>((1U << helpers.size()) - 1U);
        // Tail group: one write per helper, then the own-block operands are
        // prepared while the tails are in flight (loads join no wait).
        uint64_t last = 0, size = 0;
        c.localSrc.addr = c.input; c.localSrc.token = c.inputToken;
        for (size_t i = 0; i < helpers.size(); ++i) {
            const Route &route = *helpers[i];
            Seek(c, last, static_cast<uint64_t>(route.target) * a.bytes + a.bytes - route.length);
            c.remote.addr = hScr[i]; c.remote.token = hScrTok[i];
            if (size != route.length) { c.size = route.length; size = route.length; }
            CCU_TRY(ccu::Write(c.Channel(route.helper), c.remote, c.localSrc, c.size, c.relayEvent,
                               static_cast<uint16_t>(1U << i)));
        }
        std::vector<ccu::LocalAddr> blockSrc(helpers.size());
        std::vector<ccu::RemoteAddr> blockDst(helpers.size());
        for (size_t i = 0; i < helpers.size(); ++i) {
            blockSrc[i].addr = c.input; blockSrc[i].token = c.inputToken;
            if (helpers[i]->helper) {
                c.immediate = static_cast<uint64_t>(helpers[i]->helper) * a.bytes;
                blockSrc[i].addr += c.immediate;
            }
            blockDst[i].addr = hOut[i]; blockDst[i].token = hOutTok[i];
        }
        c.size = a.bytes;
        CCU_TRY(ccu::EventWait(c.relayEvent, bits));
        // Tails are complete: helpers may serve them; own blocks follow at once.
        for (const Route *route : helpers) CCU_TRY(ccu::NotifyRecord(c.Channel(route->helper), READY, 1));
        for (size_t i = 0; i < helpers.size(); ++i)
            CCU_TRY(ccu::Write(c.Channel(helpers[i]->helper), blockDst[i], blockSrc[i], c.size, c.relayEvent,
                               static_cast<uint16_t>(1U << i)));
        // Root's own block joins the own-block group: it never delays a tail.
        if (local) CCU_TRY(RootLocalCopy(c, false));
        CCU_TRY(ccu::EventWait(c.relayEvent, bits));
        if (localPending) CCU_TRY(ccu::EventWait(c.directEvent, LOCAL_BIT));
        for (const Route *route : helpers) CCU_TRY(ccu::NotifyRecord(c.Channel(route->helper), DONE, 1));
    }
    if (!pushed.empty()) {
        // 3. One write group for every pushed receiver, then one wait, then the
        //    notifies: a receiver returns only after its block is complete.
        uint64_t last = 0;
        uint16_t bits = 0;
        c.localSrc.addr = c.input; c.localSrc.token = c.inputToken;
        c.size = a.bytes;
        for (size_t i = 0; i < pushed.size(); ++i) {
            Seek(c, last, static_cast<uint64_t>(pushed[i]) * a.bytes);
            c.remote.addr = pOut[i]; c.remote.token = pTok[i];
            CCU_TRY(ccu::Write(c.Channel(pushed[i]), c.remote, c.localSrc, c.size, c.directEvent,
                               static_cast<uint16_t>(1U << i)));
            bits |= static_cast<uint16_t>(1U << i);
        }
        const bool copyHere = local && helpers.empty();
        if (copyHere) CCU_TRY(RootLocalCopy(c, false));
        CCU_TRY(ccu::EventWait(c.directEvent,
                               copyHere && a.bytes < MAX_DATA_SIZE ? static_cast<uint16_t>(bits | LOCAL_BIT) : bits));
        for (uint32_t r : pushed) CCU_TRY(ccu::NotifyRecord(c.Channel(r), DONE, 1));
    }
    // 4. A large call keeps root until every reader of its input has finished.
    //    A small call waits only for each reader to have entered this call --
    //    which it records before reading, and which every rank does at launch,
    //    so this costs nothing. It bounds root to one call ahead of its slowest
    //    reader, which is what makes two generations of slots and notify bits
    //    sufficient.
    if (!readers.empty() && StagedReader(a, readers[0])) {
        CCU_IF(c.gen != 0) {
            for (uint32_t r : readers) CCU_TRY(ccu::NotifyWait(c.Channel(r), STAGE, 2));
        }
        CCU_ELSE {
            for (uint32_t r : readers) CCU_TRY(ccu::NotifyWait(c.Channel(r), STAGE, 1));
        }
    } else {
        for (uint32_t r : readers) CCU_TRY(ccu::NotifyWait(c.Channel(r), ACK, 1));
    }
    return CCU_SUCCESS;
}

// ---------------------------------------------------------------- readers
// Direct reader of its whole block from root input, chunked above MAX.
CcuResult PeerPull(Context &c)
{
    const auto &a = c.a;
    const uint64_t rounds = a.bytes / MAX_DATA_SIZE, tail = a.bytes % MAX_DATA_SIZE;
    const auto root = c.Channel(a.root);
    // A staged read takes root's input from the generation's own slots and
    // reports nothing back.
    const bool staged = StagedReader(a, a.rank);
    const uint64_t from = static_cast<uint64_t>(a.rank) * a.bytes;
    // Entering this call is the only thing root needs from this reader: a rank
    // runs its kernels in stream order, so having entered means the previous
    // staged read is complete. Recorded ahead of the metadata wait, it reaches
    // root long before root's closing wait, and nothing reports back after the
    // read. One generation branch holds both the record and the wait.
    // Prepare every local operand before the metadata wait.
    c.localDst.addr = c.output; c.localDst.token = c.outputToken;
    if (from) c.offset = from;
    c.size = rounds ? MAX_DATA_SIZE : tail;
    if (staged) {
        CCU_IF(c.gen != 0) {
            CCU_TRY(ccu::NotifyRecord(root, STAGE, 2));
            CCU_TRY(ccu::NotifyWait(root, STAGE_META, 8));
            c.remote.addr = ccu::GetResByChannel<ccu::Variable>(root, 6);
            c.remote.token = c.outputToken;
        }
        CCU_ELSE {
            CCU_TRY(ccu::NotifyRecord(root, STAGE, 1));
            CCU_TRY(ccu::NotifyWait(root, STAGE_META, 1));
            c.remote.addr = ccu::GetResByChannel<ccu::Variable>(root, 4);
            c.remote.token = c.outputToken;
        }
        if (from) c.remote.addr += c.offset;
    } else {
        CCU_TRY(ccu::NotifyWait(root, META, 3));
        c.remote.addr = ccu::GetResByChannel<ccu::Variable>(root, 0);
        c.remote.token = ccu::GetResByChannel<ccu::Variable>(root, 1);
        if (from) c.remote.addr += c.offset;
    }
    if (rounds) {
        c.immediate = MAX_DATA_SIZE; c.pairs = rounds; c.offset = UINT64_MAX;
        CCU_WHILE(c.pairs != 0) {
            CCU_TRY(ccu::Read(root, c.localDst, c.remote, c.size, c.directEvent, 1));
            CCU_TRY(ccu::EventWait(c.directEvent, 1));
            c.remote.addr += c.immediate; c.localDst.addr += c.immediate;
            c.pairs += c.offset;
        }
        if (tail) c.size = tail;
    }
    if (tail) {
        CCU_TRY(ccu::Read(root, c.localDst, c.remote, c.size, c.directEvent, 1));
        CCU_TRY(ccu::EventWait(c.directEvent, 1));
    }
    return staged ? CCU_SUCCESS : ccu::NotifyRecord(root, ACK, 1);
}
// Receiver whose block root writes: publish the destination once, then wait
// for the notify that root records after its write has completed.
CcuResult PeerPush(Context &c)
{
    const auto root = c.Channel(c.a.root);
    CCU_TRY(ccu::WriteVariableWithNotify(root, c.output, 0, META, 1));
    CCU_TRY(ccu::WriteVariableWithNotify(root, c.outputToken, 1, META, 2));
    return ccu::NotifyWait(root, DONE, 1);
}
// Far reader of a Uniform call. Root part [0, len) is read from root input;
// a sentinel (no helper) reads its whole block. A served far first tells its
// helper where to write the tail [len, B); that write overlaps the root-part
// read, and only its completion is awaited last.
CcuResult FarPull(Context &c)
{
    const auto &a = c.a;
    const auto root = c.Channel(a.root);
    c.localDst.addr = c.output; c.localDst.token = c.outputToken;
    if (a.rank) c.offset = static_cast<uint64_t>(a.rank) * a.bytes;
    c.size = a.bytes;
    CCU_TRY(ccu::NotifyWait(root, META, 7));
    auto input = ccu::GetResByChannel<ccu::Variable>(root, 0);
    auto token = ccu::GetResByChannel<ccu::Variable>(root, 1);
    auto selector = ccu::GetResByChannel<ccu::Variable>(root, 2);
    // Copy it: after this reader's ACK the root may overwrite the slot, and the
    // comparisons below still run.
    c.pairs = selector;
    c.remote.addr = input; c.remote.token = token;
    if (a.rank) c.remote.addr += c.offset;
    std::vector<uint32_t> candidates;
    for (uint32_t h = 0; h < a.rankSize; ++h) {
        // A relay helper shares the root Server, never this reader's own Server.
        if (h != a.root && h != a.rank && c.Own(h) && !(a.localMask & (1U << h))) candidates.push_back(h);
    }
    CCU_IF(c.pairs == a.rankSize) {
        CCU_TRY(ccu::Read(root, c.localDst, c.remote, c.size, c.directEvent, 1));
        CCU_TRY(ccu::EventWait(c.directEvent, 1));
        CCU_TRY(ccu::NotifyRecord(root, ACK, 1));
    }
    CCU_ELSE {
        CCU_TRY(ccu::NotifyWait(root, META, 8));
        auto length = ccu::GetResByChannel<ccu::Variable>(root, 3);
        c.size = length;
        for (uint32_t h : candidates) {
            CCU_IF(c.pairs == h) {
                CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(h), c.output, 0, META, 1));
                CCU_TRY(ccu::WriteVariableWithNotify(c.Channel(h), c.outputToken, 1, META, 2));
            }
        }
        CCU_TRY(ccu::Read(root, c.localDst, c.remote, c.size, c.directEvent, 1));
        CCU_TRY(ccu::EventWait(c.directEvent, 1));
        CCU_TRY(ccu::NotifyRecord(root, ACK, 1));
        for (uint32_t h : candidates) {
            CCU_IF(c.pairs == h) { CCU_TRY(ccu::NotifyWait(c.Channel(h), DONE, 1)); }
        }
    }
    return CCU_SUCCESS;
}

// ---------------------------------------------------------------- helper
// Publish buffers to root and wait until the relayed tail is in scratch.
CcuResult HelperRoot(Context &c)
{
    const auto root = c.Channel(c.a.root);
    CCU_TRY(ccu::WriteVariableWithNotify(root, c.output, 0, META, 1));
    CCU_TRY(ccu::WriteVariableWithNotify(root, c.outputToken, 1, META, 2));
    CCU_TRY(ccu::WriteVariableWithNotify(root, c.scratch, 2, META, 4));
    CCU_TRY(ccu::WriteVariableWithNotify(root, c.scratchToken, 3, META, 8));
    return ccu::NotifyWait(root, READY, 1);
}
// Write the complete scratch tail into the far output; DONE follows the write.
CcuResult HelperFar(Context &c, const Route &route)
{
    const auto far = c.Channel(route.target);
    c.localSrc.addr = c.scratch; c.localSrc.token = c.scratchToken;
    c.size = route.length;
    c.immediate = c.a.bytes - route.length;
    CCU_TRY(ccu::NotifyWait(far, META, 3));
    auto output = ccu::GetResByChannel<ccu::Variable>(far, 0);
    auto token = ccu::GetResByChannel<ccu::Variable>(far, 1);
    c.remote.addr = output; c.remote.addr += c.immediate; c.remote.token = token;
    CCU_TRY(ccu::Write(far, c.remote, c.localSrc, c.size, c.relayEvent, 1));
    CCU_TRY(ccu::EventWait(c.relayEvent, 1));
    return ccu::NotifyRecord(far, DONE, 1);
}
CcuResult HelperDone(Context &c)
{
    // A wait carries no channel: bind (without an instruction) one root channel
    // variable so kernel registration selects the root channel's die.
    auto bind = ccu::GetResByChannel<ccu::Variable>(c.Channel(c.a.root), 0);
    (void)bind;
    return ccu::NotifyWait(c.Channel(c.a.root), DONE, 1);
}
}  // namespace

void ConfigureKernel(CcuKernelArgBase &a)
{
    a.relayDies = 0;
    a.commitDies = 0;
    if (a.rank == a.root) {
        a.activeDies = 1;
        for (uint32_t r = 0; r < a.rankSize; ++r) if (r != a.root) a.activeDies |= 1U << a.channelDie[r];
        return;
    }
    a.activeDies = 1U << a.channelDie[a.root];
    if (!Uniform(a) || !(a.nearMask & (1U << a.rank))) return;
    const Plan p = MakePlan(a);
    if (const Route *route = RouteOfHelper(p, a.rank)) a.relayDies = 1U << a.channelDie[route->target];
}

CcuResult CcuKernel(CcuKernelArg arg)
{
    if (!arg) return CCU_E_PARA;
    const auto &a = *static_cast<CcuKernelArgBase *>(arg);
    Context c(a);
    CCU_TRY(ccu::LoadArg(c.input, 0)); CCU_TRY(ccu::LoadArg(c.output, 1));
    CCU_TRY(ccu::LoadArg(c.inputToken, 2)); CCU_TRY(ccu::LoadArg(c.outputToken, 3));
    if (Staged(a)) {
        CCU_TRY(ccu::LoadArg(c.gen, 4));
    } else if (a.bytes > SMALL) {
        CCU_TRY(ccu::LoadArg(c.scratch, 4)); CCU_TRY(ccu::LoadArg(c.scratchToken, 5));
    }
    if (a.rank == a.root) {
        if (a.phase != 0) return CCU_E_PARA;
        return RootKernel(c);
    }
    if (a.phase != 0 && !Uniform(a)) return CCU_E_PARA;
    if (!Uniform(a)) {
        if (!c.Own(a.root)) return CCU_E_PARA;
        return Pushed(a, a.rank, Plan{}) ? PeerPush(c) : PeerPull(c);
    }
    if (!(a.nearMask & (1U << a.rank))) {
        if (a.phase != 0 || !c.Own(a.root)) return CCU_E_PARA;
        return FarPull(c);
    }
    const Plan p = MakePlan(a);
    const Route *route = RouteOfHelper(p, a.rank);
    if (route == nullptr) {
        if (a.phase != 0 || !c.Own(a.root)) return CCU_E_PARA;
        return PeerPush(c);
    }
    // Helper phases: 4 = root and far channels share this die (one kernel);
    // otherwise 1 (root die) -> 2 (far die) -> 3 (root die) on one host thread.
    switch (a.phase) {
        case 4:
            if (!c.Own(a.root) || !c.Own(route->target)) return CCU_E_PARA;
            CCU_TRY(HelperRoot(c));
            CCU_TRY(HelperFar(c, *route));
            return ccu::NotifyWait(c.Channel(a.root), DONE, 1);
        case 1:
            if (!c.Own(a.root)) return CCU_E_PARA;
            return HelperRoot(c);
        case 2:
            if (!c.Own(route->target)) return CCU_E_PARA;
            return HelperFar(c, *route);
        case 3:
            if (!c.Own(a.root)) return CCU_E_PARA;
            return HelperDone(c);
        default:
            return CCU_E_PARA;
    }
}
#undef CCU_TRY
}  // namespace ops_hccl
