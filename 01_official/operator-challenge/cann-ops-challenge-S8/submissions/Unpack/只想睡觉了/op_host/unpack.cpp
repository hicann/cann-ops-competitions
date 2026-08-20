
#include "unpack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <cstring>
#include <random>
#include <iostream>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    UnpackTilingData tiling;
    constexpr int32_t max_aiv_num = 40; // Ascend910B的aiv数量

    auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
    int DataTypeSize = GetSizeByDataType(dt);

    const auto runtime_attrs = context->GetAttrs();
    int64_t axis = *(runtime_attrs->GetInt(1));
    const auto input_shape = context->GetInputTensor(0)->GetOriginShape();
    uint8_t nOutputDims = input_shape.GetDimNum();
    if (axis < 0) {
        axis += nOutputDims;
    }
    int32_t D = 1;
    int32_t H = 1;
    int32_t W = 1;

    if (axis == 0) {// 2_0情况
        H = input_shape.GetDim(axis);
        for (int i = axis + 1; i < nOutputDims; ++i) {
            W *= input_shape.GetDim(i);
        }
        int32_t W_512B_aligned = (W * DataTypeSize + 511) / 512 * 512 / DataTypeSize;

        if (W_512B_aligned * DataTypeSize < 32 * 1024) {
            context->SetTilingKey(1);// 2_0 small loop
        } else {
            context->SetTilingKey(2);// 2_0 big loop
        }
    } else if (axis == nOutputDims - 1) {// 2_1情况
        for (int i = 0; i < axis; ++i) {
            H *= input_shape.GetDim(i);
        }
        W = input_shape.GetDim(axis);
        context->SetTilingKey(3);// 2_1
    } else {// 3维情况
        for (int i = 0; i < axis; ++i) {
            D *= input_shape.GetDim(i);
        }
        H = input_shape.GetDim(axis);
        for (int i = axis + 1; i < nOutputDims; ++i) {
            W *= input_shape.GetDim(i);
        }
        context->SetTilingKey(4);// 3维
    }
    tiling.set_D(D);
    tiling.set_H(H);
    tiling.set_W(W);

    int32_t totalRows = D * H;
    int32_t coreNum = std::min(max_aiv_num, totalRows);
    int32_t coreRows = totalRows / coreNum;
    int32_t tailRows = totalRows % coreNum;

    if (axis == nOutputDims - 1 && DataTypeSize != 1) {
        int32_t block_16_rows = (totalRows + 15) / 16;
        coreNum = std::min(max_aiv_num, block_16_rows);
        coreRows = block_16_rows / coreNum * 16;
        tailRows = block_16_rows % coreNum;
    }
    if (axis != 0 && axis != nOutputDims - 1) {// 3维情况,划分D
        if (D < coreNum) {
            coreNum = D;
        }
        coreRows = D / coreNum;
        tailRows = D % coreNum;
    }

    tiling.set_coreRows(coreRows);
    tiling.set_tailRows(tailRows);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t usrSize = 16 * 1024 * 1024; // 设置用户需要使用的workspace大小为256字节。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Unpack : public OpDef {
public:
    explicit Unpack(const char* name) : OpDef(name) {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num").Int();
        this->Attr("axis").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Unpack);
} // namespace ops
