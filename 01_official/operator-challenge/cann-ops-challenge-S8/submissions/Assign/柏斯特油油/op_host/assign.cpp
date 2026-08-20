
#include "assign_tiling.h"
#include "register/op_def_registry.h"


typedef long long ll;

namespace optiling
{
    static constexpr ll BLOCK_NUM = 40;
    static constexpr ll MIN_TILI_SIZE = 16 * 1024;
    static constexpr ll MAX_TILI_SIZE = 24 * 1024;
    static constexpr ll TILI_STEP = 512;

    static ll ChooseTiliSize(ll totalLength)
    {
        ll bestTile = MIN_TILI_SIZE;
        ll bestK = -1;

        for (ll tile = MIN_TILI_SIZE; tile <= MAX_TILI_SIZE; tile += TILI_STEP)
        {
            ll q = totalLength / tile;
            ll k = q % BLOCK_NUM;
            ll b = totalLength % tile;

            if (k == 0 && b == 0)
            {
                return tile;
            }

            if (k > bestK)
            {
                bestK = k;
                bestTile = tile;
            }

            if (k == BLOCK_NUM - 1)
            {
                break;
            }
        }

        return bestTile;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        AssignTilingData tiling;

        auto x_shape = context->GetInputShape(0)->GetOriginShape();

        int dimNum1 = x_shape.GetDimNum();
        uint16_t dimx1[4] = {0};

        ll Xlength = 1;
        for (int i = 0; i < dimNum1; i++)
        {
            dimx1[i] = x_shape.GetDim(i);
            Xlength *= dimx1[i];
        }

        auto y_shape = context->GetInputShape(1)->GetOriginShape();
        int dimNum2 = y_shape.GetDimNum();
        uint16_t dimy[4] = {0};
        ll ylength = 1;
        for (int i = 0; i < dimNum2; i++)
        {
            dimy[i] = y_shape.GetDim(i);
            ylength *= dimy[i];
        }

        int dimNum = 0;
        for (int i = 0; i < dimNum1; i++)
        {
            if (dimx1[i] == 1 && dimy[i] == 1)
            {
                continue;
            }
            dimx1[dimNum] = dimx1[i];
            dimy[dimNum] = dimy[i];
            dimNum++;
        }
        if (dimNum == 0)
        {
            dimNum = 1;
            dimx1[0] = 1;
            dimy[0] = 1;
        }
        // 消除都为1的维度

        int dimMax[4] = {0};
        ll Maxlength = 1;
        for (int i = 0; i < dimNum; i++)
        {
            dimMax[i] = std::max(dimx1[i], dimy[i]);
            Maxlength *= dimMax[i];
        }

        // tiling.set_inputlength(ylength);
        tiling.set_totlength(Maxlength);
        tiling.set_dimNum(dimNum);
        tiling.set_outputShape(dimx1);
        tiling.set_inputShape(dimy);

        // 广播检测：从后向前对齐比较 input(0) 和 input(1) 的 shape
        bool needBroadcast = false;
        bool lastDimBroadcast = false;
        /*
        // 以下 tilisize 计算已下放至 kernel init 中自行推导，host 侧不再传递
        int maxDim = std::max(dimNum1, dimNum2);
        bool consecutiveDims = true;
        int tilisize = 1;
        for (int i = 0; i < maxDim; i++)
        {
            int d1 = (i < dimNum1) ? dimx1[dimNum1 - 1 - i] : 1;
            int d2 = (i < dimNum2) ? dimy[dimNum2 - 1 - i] : 1;
            if (d1 == d2 && consecutiveDims == true)
            {
                tilisize *= d1;
            }
            else
            {
                consecutiveDims = false;
            }
            if (d1 != d2)
            {
                consecutiveDims = false;
                needBroadcast = true;
                if (i == 0)
                {
                    lastDimBroadcast = true;
                }
            }
        }
        */
        // 简化版广播检测：仅保留 tilingkey 判断所需信息
        for (int i = 0; i < dimNum; i++)
        {
            if (dimx1[i] != dimy[i])
            {
                needBroadcast = true;
                if (i == dimNum - 1)
                {
                    lastDimBroadcast = true;
                }
            }
        }

        uint32_t tilingkey;
        if (!needBroadcast)
        {
            ll tilisize = ChooseTiliSize(Maxlength);
            tiling.set_tilisize(tilisize);
            tilingkey = 3; // 不需要广播
        }
        else if (!lastDimBroadcast)
        {
            tilingkey = 1; // 需要广播，最后一维不需要广播
        }
        else
        {
            tilingkey = 2; // 需要广播，最后一维需要广播
        }
        context->SetTilingKey(tilingkey);

        auto tensor = context->GetInputTensor(0);
        auto type = tensor->GetDataType();

        context->SetBlockDim(BLOCK_NUM);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Assign : public OpDef
    {
    public:
        explicit Assign(const char *name) : OpDef(name)
        {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("other")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Assign);
}
