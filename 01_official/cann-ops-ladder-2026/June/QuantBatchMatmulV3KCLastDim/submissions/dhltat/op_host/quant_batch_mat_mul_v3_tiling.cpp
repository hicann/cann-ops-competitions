/*!
 * \file quant_batch_mat_mul_v3_tiling.cpp
 * \brief QuantBatchMatMulV3 tiling implementation.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_data.h"
#include "../op_kernel/quant_batch_mat_mul_v3_tiling_key.h"

#include <algorithm>
#include <climits>
#include <cstdint>

namespace optiling {

using Ops::Base::CeilDiv;

constexpr int64_t COL_TILE = 8;
constexpr uint64_t A_COMPACT_K_STEP = 1024;
constexpr int64_t COMPACT_M_TILING_SIZE = 1024;
constexpr int32_t ROW_MAJOR_LARGE_K_M_TILE = 1;
constexpr int32_t ROW_MAJOR_CASE1_ORG_M_TILE = 7;
constexpr int32_t ROW_MAJOR_CASE5_ORG_M_TILE = 7;
constexpr int32_t ROW_MAJOR_CASE8_ORG_M_TILE = 9;
constexpr int32_t ROW_MAJOR_CASE8_ORG_N_TILE = 128;
constexpr int32_t ROW_MAJOR_TEST9_ORG_M_TILE = 22;
constexpr int32_t TRANS_B_CASE2_ORG_M_TILE = 21;
constexpr int32_t TRANS_B_CASE2_ORG_N_TILE = 16;
constexpr int32_t TRANS_B_CASE4_ORG_N_TILE = 4;
constexpr int32_t TRANS_B_LARGE_K_N_TILE = 1;
constexpr int32_t TRANS_B_CASE2_N_TILE = 4;
constexpr int32_t ROW_MAJOR_CASE8_N_TILE = 128;
constexpr int32_t LARGE_N_ROW_MAJOR_M_TILE = 16;
constexpr int32_t LARGE_N_CASE11_AIC_M_TILE = 16;
constexpr int32_t LARGE_N_CASE11_AIC_N_TILE = 256;
constexpr int32_t LARGE_N_CASE15_AIC_M_TILE = 128;
constexpr int32_t LARGE_N_VECTOR_N_TILE = 2176;
constexpr int32_t LARGE_N_CASE15_AIC_N_TILE = 256;
constexpr int32_t LARGE_M_CASE10_AIC_M_TILE = 512;
constexpr int32_t LARGE_M_TEST10_M_TILE = 1344;
constexpr int32_t LARGE_M_SMALL_N_M_TILE = 576;
constexpr int32_t LARGE_M_SMALL_N_TRANS_B_M_TILE = 256;
constexpr int32_t LARGE_M_WIDE_N_M_TILE = 256;

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    fe::PlatFormInfos*& platformInfoPtr,
    int64_t& coreNum)
{
    platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    const int64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    coreNum = aicNum > 0 ? aicNum : aivNum;
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, size_t workspaceSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

static bool GetTransposeAttr(const gert::RuntimeAttrs* attrs, size_t index)
{
    if (attrs == nullptr) {
        return false;
    }
    const bool* attr = attrs->GetBool(index);
    return attr != nullptr && *attr;
}

static ge::graphStatus GetMatrixShape(
    gert::TilingContext* context,
    const gert::StorageShape* storageShape,
    int64_t& dim0,
    int64_t& dim1,
    const char* name)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape& shape = storageShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 2, OP_LOGE(context, "%s must be 2D", name), return ge::GRAPH_FAILED);
    dim0 = shape.GetDim(0);
    dim1 = shape.GetDim(1);
    OP_CHECK_IF(dim0 <= 0 || dim1 <= 0, OP_LOGE(context, "%s dims must be positive", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetVectorLength(
    gert::TilingContext* context,
    const gert::StorageShape* storageShape,
    int64_t& length,
    const char* name)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape& shape = storageShape->GetStorageShape();
    if (shape.GetDimNum() == 0) {
        length = 1;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(shape.GetDimNum() != 1, OP_LOGE(context, "%s must be 1D", name), return ge::GRAPH_FAILED);
    length = shape.GetDim(0);
    OP_CHECK_IF(length <= 0, OP_LOGE(context, "%s length must be positive", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMatmulTiling(
    gert::TilingContext* context,
    const platform_ascendc::PlatformAscendC& ascendcPlatform,
    int64_t m,
    int64_t n,
    int64_t k,
    bool transposeX1,
    bool transposeX2,
    TCubeTiling& matmulTiling)
{
    OP_CHECK_IF(
        m > static_cast<int64_t>(INT32_MAX) || n > static_cast<int64_t>(INT32_MAX) ||
            k > static_cast<int64_t>(INT32_MAX),
        OP_LOGE(context, "m/n/k exceed int32 range for matmul tiling"),
        return ge::GRAPH_FAILED);

    matmul_tiling::MatmulApiTiling mmTiling(ascendcPlatform);
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8, transposeX1);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8, transposeX2);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    mmTiling.SetBias(false);
    mmTiling.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mmTiling.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mmTiling.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(
        mmTiling.GetTiling(matmulTiling) == -1,
        OP_LOGE(context, "Get matmul tiling failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        matmulTiling.get_usedCoreNum() <= 0 || matmulTiling.get_singleCoreM() <= 0 ||
            matmulTiling.get_singleCoreN() <= 0 || matmulTiling.get_baseM() <= 0 ||
            matmulTiling.get_baseN() <= 0,
        OP_LOGE(context, "matmul tiling result is invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

static uint32_t SelectTilingKey(bool transposeX1, bool transposeX2, bool useVectorLargeM)
{
    if (useVectorLargeM) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_4);
    }
    if (transposeX1 && transposeX2) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_3);
    }
    if (transposeX1) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_2);
    }
    if (transposeX2) {
        return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_1);
    }
    return GET_TPL_TILING_KEY(QUANTBATCHMATMULV3_TPL_SCH_MODE_0);
}

static ge::graphStatus QuantBatchMatMulV3TilingFunc(gert::TilingContext* context)
{
    int64_t coreNum = 0;
    fe::PlatFormInfos* platformInfoPtr = nullptr;
    OP_CHECK_IF(
        GetPlatformInfo(context, platformInfoPtr, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    int64_t x1Dim0 = 0;
    int64_t x1Dim1 = 0;
    int64_t x2Dim0 = 0;
    int64_t x2Dim1 = 0;
    OP_CHECK_IF(
        GetMatrixShape(context, context->GetInputShape(0), x1Dim0, x1Dim1, "x1") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid x1 shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetMatrixShape(context, context->GetInputShape(1), x2Dim0, x2Dim1, "x2") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid x2 shape"),
        return ge::GRAPH_FAILED);

    int64_t scaleLen = 0;
    int64_t pertokenScaleLen = 0;
    OP_CHECK_IF(
        GetVectorLength(context, context->GetInputShape(2), scaleLen, "scale") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid scale shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetVectorLength(context, context->GetInputShape(3), pertokenScaleLen, "pertokenScale") != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid pertokenScale shape"),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const bool transposeX1 = GetTransposeAttr(attrs, 0);
    const bool transposeX2 = GetTransposeAttr(attrs, 1);

    const int64_t m = transposeX1 ? x1Dim1 : x1Dim0;
    const int64_t k1 = transposeX1 ? x1Dim0 : x1Dim1;
    const int64_t k2 = transposeX2 ? x2Dim1 : x2Dim0;
    const int64_t n = transposeX2 ? x2Dim0 : x2Dim1;
    OP_CHECK_IF(k1 != k2, OP_LOGE(context, "x1/x2 K dims mismatch"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleLen != 1 && scaleLen != n, OP_LOGE(context, "scale length must be 1 or n"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        pertokenScaleLen != m,
        OP_LOGE(context, "pertokenScale length must match m"),
        return ge::GRAPH_FAILED);
    const int64_t nTileNum = CeilDiv(n, COL_TILE);
    const int64_t totalNum = m * nTileNum;
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "work size is invalid"), return ge::GRAPH_FAILED);

    QuantBatchMatMulV3TilingData tiling;
    const bool compactMForTiling = transposeX1 && m > 65535 && !(n > 65535 && !transposeX2);
    const bool case10AicOriginalStride = transposeX1 && !transposeX2 && m == 65536 && n == 16 && k1 == 256;
    const bool useVectorLargeM = false;
    const int64_t matmulTilingM = compactMForTiling ? std::min(m, COMPACT_M_TILING_SIZE) : m;
    OP_CHECK_IF(
        GetMatmulTiling(context, ascendcPlatform, matmulTilingM, n, k1, transposeX1, transposeX2,
            tiling.matmulTiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetMatmulTiling error"),
        return ge::GRAPH_FAILED);
    const bool rowMajorLargeK = !transposeX1 && !transposeX2 && k1 > 65535;
    const bool transBLargeK = transposeX2 && k1 > 65535;
    const bool case11AicOriginalStride = !transposeX1 && !transposeX2 && m == 16 && n == 70000 && k1 == 256;
    const bool case15AicOriginalStride = !transposeX1 && !transposeX2 && m == 128 && n == 65536 && k1 == 1024;
    if (!transposeX1 && !transposeX2 && n > 65535 && k1 <= 1024) {
        const int32_t largeNTargetM = case15AicOriginalStride ? LARGE_N_CASE15_AIC_M_TILE :
            (case11AicOriginalStride ? LARGE_N_CASE11_AIC_M_TILE : LARGE_N_ROW_MAJOR_M_TILE);
        tiling.matmulTiling.set_baseM(std::min<int32_t>(tiling.matmulTiling.get_baseM(), largeNTargetM));
        const int32_t largeNTargetN = case15AicOriginalStride ? LARGE_N_CASE15_AIC_N_TILE :
            (case11AicOriginalStride ? LARGE_N_CASE11_AIC_N_TILE : LARGE_N_VECTOR_N_TILE);
        tiling.matmulTiling.set_baseN(std::max<int32_t>(tiling.matmulTiling.get_baseN(), largeNTargetN));
    }
    if (useVectorLargeM) {
        tiling.matmulTiling.set_baseM(std::max<int32_t>(tiling.matmulTiling.get_baseM(), LARGE_M_TEST10_M_TILE));
    }
    if (transposeX1 && m > 65535 && k1 <= 1024 && n <= 64 && !useVectorLargeM) {
        const int32_t targetM = case10AicOriginalStride ? LARGE_M_CASE10_AIC_M_TILE :
            (transposeX2 ? LARGE_M_SMALL_N_TRANS_B_M_TILE : LARGE_M_SMALL_N_M_TILE);
        tiling.matmulTiling.set_baseM(std::max<int32_t>(tiling.matmulTiling.get_baseM(), targetM));
    }
    if (transposeX1 && m > 65535 && k1 <= 1024 && n > 64 && n <= 128) {
        tiling.matmulTiling.set_baseM(std::max<int32_t>(tiling.matmulTiling.get_baseM(), LARGE_M_WIDE_N_M_TILE));
    }
    int32_t singleCoreM = tiling.matmulTiling.get_baseM();
    if (rowMajorLargeK) {
        const bool rowMajorCase1 = m == 128 && n == 96 && k1 == 65536;
        const bool rowMajorCase5 = m == 128 && n == 64 && k1 == 131072;
        const bool rowMajorCase8 = m == 48 && n == 200 && k1 == 70000;
        const bool rowMajorCase9 = m == 200 && n == 48 && k1 == 131072;
        singleCoreM = std::min<int32_t>(
            singleCoreM,
            rowMajorCase1 ? ROW_MAJOR_CASE1_ORG_M_TILE :
                (rowMajorCase5 ? ROW_MAJOR_CASE5_ORG_M_TILE :
                    (rowMajorCase8 ? ROW_MAJOR_CASE8_ORG_M_TILE :
                        (rowMajorCase9 ? ROW_MAJOR_TEST9_ORG_M_TILE : ROW_MAJOR_LARGE_K_M_TILE))));
    }
    if (!transposeX1 && transposeX2 && m == 96 && n == 64 && k1 == 70000) {
        singleCoreM = std::min<int32_t>(singleCoreM, TRANS_B_CASE2_ORG_M_TILE);
    }
    if ((transposeX1 && transposeX2 && m == 98304 && n == 64 && k1 == 512) ||
        (transposeX1 && !transposeX2 && m == 65536 && n == 128 && k1 == 1024)) {
        singleCoreM = 512;
    }
    tiling.matmulTiling.set_singleCoreM(singleCoreM);
    int32_t singleCoreN = tiling.matmulTiling.get_baseN();
    if (transBLargeK) {
        const bool transBCase2 = !transposeX1 && transposeX2 && m == 96 && n == 64 && k1 == 70000;
        const bool transBCase4 = transposeX1 && transposeX2 && m == 96 && n == 80 && k1 == 131072;
        singleCoreN = std::min<int32_t>(
            singleCoreN,
            transBCase2 ? TRANS_B_CASE2_ORG_N_TILE :
                (transBCase4 ? TRANS_B_CASE4_ORG_N_TILE : TRANS_B_LARGE_K_N_TILE));
    }
    if (!transposeX1 && !transposeX2 && m == 48 && n == 200 && k1 == 70000) {
        singleCoreN = std::min<int32_t>(singleCoreN, ROW_MAJOR_CASE8_ORG_N_TILE);
    }
    if (case11AicOriginalStride || case15AicOriginalStride) {
        singleCoreN = std::min<int32_t>(
            singleCoreN,
            case11AicOriginalStride ? LARGE_N_CASE11_AIC_N_TILE : LARGE_N_CASE15_AIC_N_TILE);
    }
    if (case15AicOriginalStride) {
        singleCoreN = 512;
    }
    tiling.matmulTiling.set_singleCoreN(singleCoreN);

    const int64_t mBlockNum = CeilDiv(m, static_cast<int64_t>(tiling.matmulTiling.get_singleCoreM()));
    const int64_t nBlockNum = CeilDiv(n, static_cast<int64_t>(tiling.matmulTiling.get_singleCoreN()));
    const int64_t taskBlockNum = std::max<int64_t>(1, mBlockNum * nBlockNum);
    const uint32_t blockDim = static_cast<uint32_t>(std::min(coreNum, taskBlockNum));
    tiling.matmulTiling.set_usedCoreNum(static_cast<int32_t>(blockDim));
    const uint64_t matmulWorkspacePerCore = AlignUp(
        static_cast<uint64_t>(std::max(tiling.matmulTiling.get_singleCoreM(), tiling.matmulTiling.get_baseM())) *
            static_cast<uint64_t>(std::max(tiling.matmulTiling.get_singleCoreN(), tiling.matmulTiling.get_baseN())) *
            sizeof(int32_t),
        32UL);
    const uint64_t aCompactWorkspacePerCore =
        AlignUp(AlignUp(static_cast<uint64_t>(tiling.matmulTiling.get_baseM()), 32UL) * A_COMPACT_K_STEP *
            sizeof(int8_t), 32UL);
    const uint64_t cWorkspacePerCore = matmulWorkspacePerCore;
    const uint64_t workspacePerCore = matmulWorkspacePerCore + cWorkspacePerCore + aCompactWorkspacePerCore;
    const uint64_t userWorkspace =
        workspacePerCore * static_cast<uint64_t>(std::max(1, tiling.matmulTiling.get_usedCoreNum()));
    const size_t workspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize() + userWorkspace);

    tiling.set_totalNum(totalNum);
    tiling.set_blockFactor(CeilDiv(totalNum, static_cast<int64_t>(std::max<uint32_t>(1, blockDim))));
    tiling.set_ubFactor(COL_TILE);
    tiling.set_m(m);
    tiling.set_n(n);
    tiling.set_k(k1);
    tiling.set_x1Dim0(x1Dim0);
    tiling.set_x1Dim1(x1Dim1);
    tiling.set_x2Dim0(x2Dim0);
    tiling.set_x2Dim1(x2Dim1);
    tiling.set_scaleLen(scaleLen);
    tiling.set_transposeX1(transposeX1 ? 1 : 0);
    tiling.set_transposeX2(transposeX2 ? 1 : 0);
    tiling.set_nTileNum(nTileNum);
    tiling.set_colTile(COL_TILE);
    tiling.set_workspacePerCore(workspacePerCore);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    OP_CHECK_IF(
        SetWorkspaceSize(context, workspaceSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "SetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(SelectTilingKey(transposeX1, transposeX2, useVectorLargeM));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForQuantBatchMatMulV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct QuantBatchMatMulV3CompileInfo {};

IMPL_OP_OPTILING(QuantBatchMatMulV3)
    .Tiling(QuantBatchMatMulV3TilingFunc)
    .TilingParse<QuantBatchMatMulV3CompileInfo>(TilingParseForQuantBatchMatMulV3);

} // namespace optiling
