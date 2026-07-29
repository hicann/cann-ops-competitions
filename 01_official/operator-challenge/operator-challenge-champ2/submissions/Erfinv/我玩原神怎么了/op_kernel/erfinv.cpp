#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_ELEMENTS_PER_ITER = 7 * 512;
constexpr uint32_t TMP_TENSOR_PAD_BYTES = 0;
constexpr uint32_t FLOAT_ALIGN_ELEMENTS = 64;
constexpr uint32_t LUT_COEFF_PER_SEGMENT = 4;
constexpr uint32_t LUT_R0_SEGMENTS = 8;
constexpr uint32_t LUT_BODY_SEGMENTS = 96;
constexpr uint32_t LUT_SEGMENTS = LUT_R0_SEGMENTS + LUT_BODY_SEGMENTS;
constexpr uint32_t LUT_R0_COEFF_COUNT = LUT_R0_SEGMENTS * LUT_COEFF_PER_SEGMENT;
constexpr uint32_t LUT_BODY_COEFF_COUNT = LUT_BODY_SEGMENTS * LUT_COEFF_PER_SEGMENT;
constexpr uint32_t LUT_COEFF_COUNT = LUT_SEGMENTS * LUT_COEFF_PER_SEGMENT;
constexpr uint32_t TAIL_BAND_COUNT = 20;
constexpr uint32_t TAIL_PARAM_COUNT = TAIL_BAND_COUNT * 2;
constexpr uint32_t LUT_TABLE_COUNT = LUT_COEFF_COUNT + TAIL_PARAM_COUNT;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int32_t LUT_COEFF_PER_SEGMENT_BYTES = 16;
constexpr int32_t TAIL_PARAM_BYTES_PER_BAND = 8;
constexpr int32_t LUT_COEFF_BYTES = 1664;
constexpr int32_t FP32_INF_BITS_I32 = 0x7f800000;
constexpr uint32_t FP32_QNAN_BITS_U32 = 0x7fc00000U;
constexpr float LUT_R0_SEGMENTS_F = 8.0f;
constexpr float LUT_R1_SEGMENTS_F = 16.0f;
constexpr float LUT_R1_BASE_INDEX_F = 8.0f;
constexpr float LUT_TAIL_BASE_INDEX_F = 24.0f;
constexpr float TAIL_SEGMENTS_PER_BAND_F = 4.0f;
constexpr float TAIL_BAND_MAX_F = 19.0f;
constexpr float MAX_FP32_LESS_THAN_ONE = 0.9999999403953552f;

static constexpr float LUT_R0_COEFFS[LUT_R0_COEFF_COUNT] = {
    0.0f, 0.0830837712f, -1.86975615e-06f, 0.000193972432f,
    0.0832758769f, 0.0836619511f, 0.000575806363f, 0.000213581719f,
    0.167727217f, 0.0854543075f, 0.00121059839f, 0.00025816937f,
    0.254650295f, 0.088650018f, 0.00197513518f, 0.000341670733f,
    0.345617115f, 0.0936252996f, 0.00298100337f, 0.00049544737f,
    0.442718863f, 0.101073645f, 0.00442514103f, 0.000795427943f,
    0.549013078f, 0.112310208f, 0.00670104101f, 0.00145146321f,
    0.669475794f, 0.130066678f, 0.0106831128f, 0.00319422875f,
};

static constexpr float LUT_COEFFS[LUT_BODY_COEFF_COUNT] = {
    0.813419819f, 0.02012695f, 0.000329237344f, 1.05580675e-05f,
    0.833886564f, 0.0208170991f, 0.000360856531f, 1.22711608e-05f,
    0.85507679f, 0.0215756241f, 0.000397523574f, 1.4170485e-05f,
    0.877064109f, 0.022413183f, 0.000439947384f, 1.65752353e-05f,
    0.899933815f, 0.0233428031f, 0.000489579048f, 1.95480588e-05f,
    0.923785746f, 0.0243806057f, 0.000548169133f, 2.32786479e-05f,
    0.9487378f, 0.0255467799f, 0.00061782531f, 2.81785778e-05f,
    0.974930584f, 0.0268669669f, 0.000701798708f, 3.45627814e-05f,
    1.00253391f, 0.0283742528f, 0.0008050467f, 4.2593143e-05f,
    1.03175581f, 0.030112125f, 0.00093230448f, 5.40551518e-05f,
    1.06285429f, 0.032138899f, 0.00109328784f, 6.97619398e-05f,
    1.09615624f, 0.0345347598f, 0.00130099838f, 9.21321553e-05f,
    1.13208413f, 0.0374131538f, 0.00157477439f, 0.000125831211f,
    1.17119789f, 0.0409401953f, 0.00194728002f, 0.000178309638f,
    1.21426368f, 0.0453696847f, 0.00247269799f, 0.000265121344f,
    1.26237118f, 0.0511104465f, 0.00324890204f, 0.000419824413f,
    1.31715035f, 0.0392451473f, 0.00201070728f, 0.0001939293f,
    1.35860014f, 0.0438483506f, 0.00258126552f, 0.000293987599f,
    1.40532374f, 0.0498928465f, 0.00343918195f, 0.000478808011f,
    1.45913458f, 0.0582076311f, 0.00481551839f, 0.000861702953f,
    1.52301943f, 0.0352118909f, 0.00187140855f, 0.000181526048f,
    1.56028426f, 0.0394992828f, 0.00240528397f, 0.000275492901f,
    1.60246432f, 0.0451363325f, 0.00320899789f, 0.000449417508f,
    1.65125906f, 0.0529025793f, 0.00450068992f, 0.00081008405f,
    1.70947242f, 0.0321671069f, 0.00175281405f, 0.000170957035f,
    1.74356329f, 0.0361856036f, 0.0022556358f, 0.000259697699f,
    1.78226423f, 0.0414759703f, 0.00301320781f, 0.000424352998f,
    1.82717776f, 0.0487754457f, 0.00423232419f, 0.000766230747f,
    1.88095176f, 0.0297693927f, 0.0016519092f, 0.000161769029f,
    1.91253483f, 0.0335585177f, 0.0021275389f, 0.00024636474f,
    1.94846725f, 0.0385526903f, 0.00284600817f, 0.000402635167f,
    1.99026859f, 0.0454526134f, 0.00400261581f, 0.0007281875f,
    2.040452f, 0.0278212037f, 0.00156479876f, 0.000154059977f,
    2.06999207f, 0.0314129815f, 0.00201773853f, 0.00023446088f,
    2.10365725f, 0.0361518413f, 0.00270161033f, 0.000383809878f,
    2.14289451f, 0.0427064896f, 0.0038041342f, 0.000694823044f,
    2.19009995f, 0.0261996146f, 0.00148978748f, 0.000146921142f,
    2.21793628f, 0.0296199527f, 0.00192209601f, 0.000224127754f,
    2.24970245f, 0.0341365263f, 0.0025757968f, 0.000367248285f,
    2.28678203f, 0.0403898656f, 0.00363022787f, 0.00066574692f,
    2.33146787f, 0.0248237811f, 0.00142325903f, 0.000141128257f,
    2.35785604f, 0.0280936845f, 0.00183826929f, 0.000214884072f,
    2.38800287f, 0.0324148759f, 0.00246444205f, 0.000352988689f,
    2.42323518f, 0.038402725f, 0.00347775687f, 0.000639325823f,
    2.46575499f, 0.0236381087f, 0.00136477163f, 0.000135736787f,
    2.4908936f, 0.0267748609f, 0.00176373031f, 0.00020682074f,
    2.51963902f, 0.0309227854f, 0.00236712676f, 0.000339267543f,
    2.55326819f, 0.0366748422f, 0.00334096281f, 0.000616444275f,
    2.59390044f, 0.0226030499f, 0.00131356157f, 0.000130286659f,
    2.61794734f, 0.0256210323f, 0.00169652782f, 0.000199996473f,
    2.6454649f, 0.0296140779f, 0.00227938709f, 0.000327375485f,
    2.67768574f, 0.0351549797f, 0.00321934815f, 0.000595190038f,
    2.71665525f, 0.0216896217f, 0.00126602058f, 0.000126374725f,
    2.73973727f, 0.0246007871f, 0.00163787545f, 0.000192658685f,
    2.76616859f, 0.0284545142f, 0.00219980394f, 0.000316969905f,
    2.79713988f, 0.0338050313f, 0.00310906162f, 0.000576273946f,
    2.83463025f, 0.0208759885f, 0.00122428755f, 0.00012200429f,
    2.85685253f, 0.0236905757f, 0.00158298179f, 0.000187161713f,
    2.88231325f, 0.0274180248f, 0.00212812261f, 0.00030719614f,
    2.9121666f, 0.032595858f, 0.00300956261f, 0.000558623986f,
    2.94833064f, 0.0201454274f, 0.00118573185f, 0.000118406177f,
    2.96978021f, 0.0228721108f, 0.00153409713f, 0.000181423995f,
    2.99436784f, 0.026484577f, 0.0020632667f, 0.000297944003f,
    3.02321362f, 0.0315049402f, 0.00291831698f, 0.000542733353f,
    3.05817962f, 0.0194848869f, 0.00114993076f, 0.000115465606f,
    3.0789299f, 0.0221311469f, 0.00148906314f, 0.000176349087f,
    3.10272646f, 0.0256383196f, 0.00200341176f, 0.000289720367f,
    3.13065791f, 0.0305143036f, 0.00283492706f, 0.000527902856f,
    3.16453505f, 0.0188839342f, 0.00111834379f, 0.000111906069f,
    3.18464923f, 0.0214563385f, 0.00144740473f, 0.000171836509f,
    3.20772481f, 0.0248666592f, 0.00194795919f, 0.000282369263f,
    3.2348218f, 0.0296096839f, 0.00275808247f, 0.000514208165f,
    3.26770377f, 0.0183342379f, 0.00108826405f, 0.000109225737f,
    3.2872355f, 0.0208384432f, 0.00140929047f, 0.000167428283f,
    3.30965066f, 0.0241593085f, 0.00189695996f, 0.000275394967f,
    3.33598232f, 0.0287794136f, 0.00268756389f, 0.000501139846f,
    3.36795044f, 0.0178289805f, 0.00106005487f, 0.000106965264f,
    3.38694644f, 0.0202699862f, 0.00137438998f, 0.000163056451f,
    3.40875387f, 0.0235079359f, 0.00184942968f, 0.000269056181f,
    3.43438029f, 0.0280139633f, 0.00262153219f, 0.000489335158f,
    3.46550512f, 0.0173625164f, 0.00103472755f, 0.000104276252f,
    3.48400664f, 0.0197448004f, 0.00134126318f, 0.000159416159f,
    3.50525212f, 0.0229055751f, 0.00180575554f, 0.000262777146f,
    3.53022623f, 0.0273054168f, 0.00255940529f, 0.00047870967f,
    3.56056976f, 0.0169301778f, 0.00101107929f, 0.000101783276f,
    3.5786128f, 0.019257687f, 0.00131037063f, 0.000156000402f,
    3.59933686f, 0.0223464295f, 0.00176491565f, 0.000256940955f,
    3.62370515f, 0.0266470835f, 0.00250167563f, 0.000468550425f,
    3.65332246f, 0.016528042f, -0.049584128f, 0.0330560841f,
    3.65332246f, 0.016528042f, 0.0586830378f, -0.0373561718f,
    3.69117737f, 0.021825606f, 0.0822749361f, -0.0513566323f,
    3.74392128f, 0.0323055834f, -0.0969167501f, 0.0646111667f,
    3.74392128f, 0.0161527917f, -0.048458375f, 0.0323055834f,
    3.74392128f, 0.0161527917f, -0.048458375f, 0.0323055834f,
    3.74392128f, 0.0161527917f, 0.201848149f, -0.129415333f,
    3.8325069f, 0.0316031091f, -0.0948093235f, 0.0632062182f,
};

static constexpr float TAIL_PARAMS[TAIL_PARAM_COUNT] = {
    0.9375f, 128.0f,
    0.96875f, 256.0f,
    0.984375f, 512.0f,
    0.9921875f, 1024.0f,
    0.99609375f, 2048.0f,
    0.998046875f, 4096.0f,
    0.9990234375f, 8192.0f,
    0.99951171875f, 16384.0f,
    0.999755859375f, 32768.0f,
    0.999877929688f, 65536.0f,
    0.999938964844f, 131072.0f,
    0.999969482422f, 262144.0f,
    0.999984741211f, 524288.0f,
    0.999992370605f, 1048576.0f,
    0.999996185303f, 2097152.0f,
    0.999998092651f, 4194304.0f,
    0.999999046326f, 8388608.0f,
    0.999999523163f, 16777216.0f,
    0.999999761581f, 33554432.0f,
    0.999999880791f, 67108864.0f,
};

template <typename T>
class KernelErfinv {
public:
    __aicore__ inline KernelErfinv() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallSize, uint32_t incSize, uint16_t formerNum, TPipe* pipeIn) {
        pipe = pipeIn;

        uint32_t beginIndex = 0;
        if (GetBlockIdx() < formerNum) {
            size = smallSize + incSize;
            beginIndex = size * GetBlockIdx();
        } else {
            size = smallSize;
            beginIndex = size * GetBlockIdx() + formerNum * incSize;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + beginIndex, size);
        yGm.SetGlobalBuffer((__gm__ T*)y + beginIndex, size);

        nElementsPerIter = min(size, MAX_ELEMENTS_PER_ITER);
        loopTimes = nElementsPerIter == 0 ? 0 : (size + nElementsPerIter - 1) / nElementsPerIter;
        nElementsPerIter = nElementsPerIter == 0 ? 1 : nElementsPerIter;

        uint32_t bufferElements = GetCalcSize(nElementsPerIter);
        uint32_t inputBytes = AlignUp(bufferElements * sizeof(T), 32);
        uint32_t outputBytes = AlignUp(bufferElements * sizeof(T), 32);
        uint32_t tmpBytes = GetTmpBytes(bufferElements);
        pipe->InitBuffer(inputBuf, BUFFER_NUM, inputBytes);
        pipe->InitBuffer(outputBuf, BUFFER_NUM, outputBytes);
        pipe->InitBuffer(tmpBuf, tmpBytes);
        pipe->InitBuffer(lutBuf, AlignUp(LUT_TABLE_COUNT * sizeof(float), 32));
        InitLutTable();
    }

    __aicore__ inline void Process() {
        if (loopTimes == 0) {
            return;
        }

        uint32_t currOffset = 0;
        uint32_t currSize = min(nElementsPerIter, size);
        CopyIn(currOffset, currSize);

        if (loopTimes > 1) {
            CopyIn(nElementsPerIter, min(nElementsPerIter, size - nElementsPerIter));
        }

        Compute(currSize);

        for (uint32_t i = 1; i < loopTimes; ++i) {
            CopyOut(currOffset, currSize);

            currOffset = i * nElementsPerIter;
            currSize = min(nElementsPerIter, size - currOffset);
            uint32_t nextOffset = currOffset + currSize;
            if (i + 1 < loopTimes) {
                CopyIn(nextOffset, min(nElementsPerIter, size - nextOffset));
            }

            Compute(currSize);
        }

        CopyOut(currOffset, currSize);
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline uint32_t GetMaskBytes(uint32_t count) {
        return AlignUp((count + 7) / 8, 32);
    }

    __aicore__ inline uint32_t GetFloatStride(uint32_t count) {
        return AlignUp(count * sizeof(float), 32) + TMP_TENSOR_PAD_BYTES;
    }

    __aicore__ inline uint32_t GetTmpBytes(uint32_t count) {
        uint32_t floatStride = GetFloatStride(count);
        uint32_t maskBytes = GetMaskBytes(count);
        if constexpr (std::is_same<T, float>::value) {
            return floatStride * 8 + maskBytes;
        } else {
            return floatStride * 10 + maskBytes;
        }
    }

    __aicore__ inline void InitLutTable() {
        LocalTensor<float> table = lutBuf.Get<float>();
        table.SetSize(LUT_TABLE_COUNT);
        for (uint32_t i = 0; i < LUT_R0_COEFF_COUNT; ++i) {
            table.SetValue(i, LUT_R0_COEFFS[i]);
        }
        for (uint32_t i = 0; i < LUT_BODY_COEFF_COUNT; ++i) {
            table.SetValue(LUT_R0_COEFF_COUNT + i, LUT_COEFFS[i]);
        }
        for (uint32_t i = 0; i < TAIL_PARAM_COUNT; ++i) {
            table.SetValue(LUT_COEFF_COUNT + i, TAIL_PARAMS[i]);
        }
    }

    __aicore__ inline uint32_t GetCalcSize(uint32_t count) {
        return AlignUp(count, FLOAT_ALIGN_ELEMENTS);
    }

    __aicore__ inline bool IsAlignedCount(uint32_t count) {
        return (count * sizeof(T) % 32) == 0;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t iterSize) {
        LocalTensor<T> xLocal = inputBuf.AllocTensor<T>();
        uint32_t calcSize = GetCalcSize(iterSize);
        if (calcSize != iterSize) {
            Duplicate(xLocal, static_cast<T>(0), calcSize);
        }
        if (IsAlignedCount(iterSize)) {
            DataCopy(xLocal, xGm[offset], iterSize);
        } else {
            uint16_t blockCount = 1;
            uint32_t inputBytes = iterSize * sizeof(T);
            uint8_t rightPadding = (AlignUp(inputBytes, 32) - inputBytes) / sizeof(T);
            DataCopyExtParams copyParams{blockCount, inputBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, rightPadding, static_cast<T>(0)};
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inputBuf.EnQue<T>(xLocal);
    }

    __aicore__ inline void Compute(uint32_t iterSize) {
        LocalTensor<T> xLocal = inputBuf.DeQue<T>();
        LocalTensor<T> yLocal = outputBuf.AllocTensor<T>();
        LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
        ComputeCore(yLocal, xLocal, tmpLocal, GetCalcSize(iterSize));
        inputBuf.FreeTensor<T>(xLocal);
        outputBuf.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t iterSize) {
        LocalTensor<T> outLocal = outputBuf.DeQue<T>();
        if (IsAlignedCount(iterSize)) {
            DataCopy(yGm[offset], outLocal, iterSize);
        } else {
            uint16_t blockCount = 1;
            uint32_t outputBytes = iterSize * sizeof(T);
            DataCopyExtParams storeParams{blockCount, outputBytes, 0, 0, 0};
            DataCopyPad(yGm[offset], outLocal, storeParams);
        }
        outputBuf.FreeTensor<T>(outLocal);
    }

    __aicore__ inline void ComputeCore(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                       const LocalTensor<uint8_t>& tmpLocal, uint32_t iterSize) {
        uint32_t floatStride = GetFloatStride(iterSize);
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<float> yFp32 = yLocal.template ReinterpretCast<float>();
            LocalTensor<float> absX = tmpLocal.template ReinterpretCast<float>();
            LocalTensor<float> raw = tmpLocal[floatStride].template ReinterpretCast<float>();
            LocalTensor<float> floorValue = tmpLocal[floatStride * 2].template ReinterpretCast<float>();
            LocalTensor<float> c0 = tmpLocal[floatStride * 3].template ReinterpretCast<float>();
            LocalTensor<float> c1 = tmpLocal[floatStride * 4].template ReinterpretCast<float>();
            LocalTensor<float> c2 = tmpLocal[floatStride * 5].template ReinterpretCast<float>();
            LocalTensor<float> c3 = tmpLocal[floatStride * 6].template ReinterpretCast<float>();
            LocalTensor<uint32_t> offset = tmpLocal[floatStride * 7].template ReinterpretCast<uint32_t>();
            LocalTensor<uint8_t> mask = tmpLocal[floatStride * 8].template ReinterpretCast<uint8_t>();
            ComputeFloat(yFp32, xLocal, absX, raw, floorValue, c0, c1, c2, c3, offset, mask, iterSize);
        } else {
            LocalTensor<float> xFp32 = tmpLocal.template ReinterpretCast<float>();
            LocalTensor<float> yFp32 = tmpLocal[floatStride].template ReinterpretCast<float>();
            LocalTensor<float> absX = tmpLocal[floatStride * 2].template ReinterpretCast<float>();
            LocalTensor<float> raw = tmpLocal[floatStride * 3].template ReinterpretCast<float>();
            LocalTensor<float> floorValue = tmpLocal[floatStride * 4].template ReinterpretCast<float>();
            LocalTensor<float> c0 = tmpLocal[floatStride * 5].template ReinterpretCast<float>();
            LocalTensor<float> c1 = tmpLocal[floatStride * 6].template ReinterpretCast<float>();
            LocalTensor<float> c2 = tmpLocal[floatStride * 7].template ReinterpretCast<float>();
            LocalTensor<float> c3 = tmpLocal[floatStride * 8].template ReinterpretCast<float>();
            LocalTensor<uint32_t> offset = tmpLocal[floatStride * 9].template ReinterpretCast<uint32_t>();
            LocalTensor<uint8_t> mask = tmpLocal[floatStride * 10].template ReinterpretCast<uint8_t>();

            Cast(xFp32, xLocal, RoundMode::CAST_NONE, iterSize);
            ComputeFloat(yFp32, xFp32, absX, raw, floorValue, c0, c1, c2, c3, offset, mask, iterSize);
            Cast(yLocal, yFp32, RoundMode::CAST_RINT, iterSize);
        }
    }

    __aicore__ inline void ComputeFloat(const LocalTensor<float>& yLocal, const LocalTensor<float>& xLocal,
                                        const LocalTensor<float>& absX, const LocalTensor<float>& raw,
                                        const LocalTensor<float>& floorValue, const LocalTensor<float>& c0,
                                        const LocalTensor<float>& c1, const LocalTensor<float>& c2,
                                        const LocalTensor<float>& c3, const LocalTensor<uint32_t>& offset,
                                        const LocalTensor<uint8_t>& mask, uint32_t iterSize) {
        Abs(absX, xLocal, iterSize);
        ApplyDynamicLut(yLocal, absX, raw, floorValue, c0, c1, c2, c3, offset, mask, iterSize);

        CompareScalar(mask, absX, 1.0f, CMPMODE::EQ, iterSize);
        Duplicate(raw.template ReinterpretCast<uint32_t>(), 0x7f800000U, iterSize);
        Select(floorValue, mask, raw, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Adds(yLocal, floorValue, 0.0f, iterSize);

        CompareScalar(mask, absX, 1.0f, CMPMODE::GT, iterSize);
        Duplicate(raw.template ReinterpretCast<uint32_t>(), 0x7fc00000U, iterSize);
        Select(floorValue, mask, raw, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Adds(yLocal, floorValue, 0.0f, iterSize);

        CompareScalar(mask, xLocal, 0.0f, CMPMODE::LT, iterSize);
        Muls(raw, yLocal, -1.0f, iterSize);
        Select(floorValue, mask, raw, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Adds(yLocal, floorValue, 0.0f, iterSize);

        FixInputNan(yLocal, xLocal, raw, floorValue, c0, c1, mask, iterSize);
    }

    __aicore__ inline void FixInputNan(const LocalTensor<float>& yLocal, const LocalTensor<float>& xLocal,
                                       const LocalTensor<float>& raw, const LocalTensor<float>& floorValue,
                                       const LocalTensor<float>& c0, const LocalTensor<float>& c1,
                                       const LocalTensor<uint8_t>& mask, uint32_t iterSize) {
        Adds(raw, xLocal, 0.0f, iterSize);
        LocalTensor<uint32_t> bits = raw.template ReinterpretCast<uint32_t>();
        ShiftLeft(bits, bits, 1U, iterSize);
        ShiftRight(bits, bits, 1U, iterSize);

        LocalTensor<int32_t> diff = raw.template ReinterpretCast<int32_t>();
        Adds(diff, diff, -FP32_INF_BITS_I32, iterSize);
        Maxs(diff, diff, 0, iterSize);

        LocalTensor<int16_t> nanFlagInt = c1.template ReinterpretCast<int16_t>();
        Cast(nanFlagInt, diff.template ReinterpretCast<float>(), RoundMode::CAST_CEIL, iterSize);
        Cast(c0, nanFlagInt, RoundMode::CAST_NONE, iterSize);
        CompareScalar(mask, c0, 0.0f, CMPMODE::GT, iterSize);

        Duplicate(raw.template ReinterpretCast<uint32_t>(), FP32_QNAN_BITS_U32, iterSize);
        Select(floorValue, mask, raw, yLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Adds(yLocal, floorValue, 0.0f, iterSize);
    }

    __aicore__ inline void ApplyDynamicLut(const LocalTensor<float>& yLocal, const LocalTensor<float>& absX,
                                           const LocalTensor<float>& raw,
                                           const LocalTensor<float>& floorValue,
                                           const LocalTensor<float>& c0, const LocalTensor<float>& c1,
                                           const LocalTensor<float>& c2, const LocalTensor<float>& c3,
                                           const LocalTensor<uint32_t>& offset,
                                           const LocalTensor<uint8_t>& mask, uint32_t iterSize) {
        LocalTensor<int32_t> offsetInt = offset.template ReinterpretCast<int32_t>();
        LocalTensor<float> table = lutBuf.Get<float>();
        table.SetSize(LUT_TABLE_COUNT);

        Duplicate(c0, 0.75f, iterSize);
        Duplicate(c1, 85.3333333333f, iterSize);
        Duplicate(c2, LUT_R1_BASE_INDEX_F, iterSize);

        Muls(raw, absX, -1.0f, iterSize);
        Adds(raw, raw, 1.0f, iterSize);
        Muls(raw, raw, MAX_FP32_LESS_THAN_ONE, iterSize);
        ShiftRight(offset, raw.template ReinterpretCast<uint32_t>(), 23U, iterSize);
        Cast(c3, offsetInt, RoundMode::CAST_NONE, iterSize);
        Muls(c3, c3, -1.0f, iterSize);
        Adds(c3, c3, 122.0f, iterSize);
        Maxs(c3, c3, 0.0f, iterSize);
        Mins(c3, c3, TAIL_BAND_MAX_F, iterSize);
        Cast(offsetInt, c3, RoundMode::CAST_FLOOR, iterSize);
        Muls(offsetInt, offsetInt, TAIL_PARAM_BYTES_PER_BAND, iterSize);
        Adds(offsetInt, offsetInt, LUT_COEFF_BYTES, iterSize);
        Gather(floorValue, table, offset, 0, iterSize);
        Adds(offsetInt, offsetInt, FLOAT_BYTES, iterSize);
        Gather(raw, table, offset, 0, iterSize);
        Muls(c3, c3, TAIL_SEGMENTS_PER_BAND_F, iterSize);
        Adds(c3, c3, LUT_TAIL_BASE_INDEX_F, iterSize);

        CompareScalar(mask, absX, 0.75f, CMPMODE::GE, iterSize);
        Select(c0, mask, c0, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);
        Select(c1, mask, c1, 10.6666666667f, SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);
        Select(c2, mask, c2, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);

        CompareScalar(mask, absX, 0.9375f, CMPMODE::GE, iterSize);
        Select(c0, mask, floorValue, c0, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Select(c1, mask, raw, c1, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Select(c2, mask, c3, c2, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Duplicate(c3, TAIL_SEGMENTS_PER_BAND_F, iterSize);
        Select(floorValue, mask, c3, LUT_R1_SEGMENTS_F, SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);
        CompareScalar(mask, absX, 0.75f, CMPMODE::LT, iterSize);
        Duplicate(c3, LUT_R0_SEGMENTS_F, iterSize);
        Select(floorValue, mask, c3, floorValue, SELMODE::VSEL_TENSOR_TENSOR_MODE, iterSize);
        Adds(floorValue, floorValue, -1.0e-4f, iterSize);

        Sub(raw, absX, c0, iterSize);
        Mul(raw, raw, c1, iterSize);
        CompareScalar(mask, absX, 1.0f, CMPMODE::LT, iterSize);
        Select(raw, mask, raw, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, iterSize);
        Maxs(raw, raw, 0.0f, iterSize);
        Min(raw, raw, floorValue, iterSize);
        Cast(offsetInt, raw, RoundMode::CAST_FLOOR, iterSize);
        Cast(floorValue, offsetInt, RoundMode::CAST_NONE, iterSize);
        Sub(raw, raw, floorValue, iterSize);

        Add(floorValue, floorValue, c2, iterSize);
        Cast(offsetInt, floorValue, RoundMode::CAST_FLOOR, iterSize);
        Muls(offsetInt, offsetInt, LUT_COEFF_PER_SEGMENT_BYTES, iterSize);
        Gather(c0, table, offset, 0, iterSize);
        Adds(offsetInt, offsetInt, FLOAT_BYTES, iterSize);
        Gather(c1, table, offset, 0, iterSize);
        Adds(offsetInt, offsetInt, FLOAT_BYTES, iterSize);
        Gather(c2, table, offset, 0, iterSize);
        Adds(offsetInt, offsetInt, FLOAT_BYTES, iterSize);
        Gather(c3, table, offset, 0, iterSize);

        Mul(c3, c3, raw, iterSize);
        Add(c3, c3, c2, iterSize);
        Mul(c3, c3, raw, iterSize);
        Add(c3, c3, c1, iterSize);
        Mul(c3, c3, raw, iterSize);
        Add(c3, c3, c0, iterSize);

        Adds(yLocal, c3, 0.0f, iterSize);
    }

private:
    TPipe* pipe;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> lutBuf;
    uint32_t size;
    uint32_t loopTimes;
    uint32_t nElementsPerIter;
};

template <typename T>
__aicore__ inline void RunErfinv(GM_ADDR x, GM_ADDR y, uint32_t smallSize, uint32_t incSize, uint16_t formerNum) {
    TPipe pipe;
    KernelErfinv<T> op;
    op.Init(x, y, smallSize, incSize, formerNum, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void erfinv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.totalSize == 0) {
        return;
    }

    if (TILING_KEY_IS(1)) {
        RunErfinv<half>(x, y, tilingData.smallSize, tilingData.incSize, tilingData.formerNum);
    } else if (TILING_KEY_IS(2)) {
        RunErfinv<bfloat16_t>(x, y, tilingData.smallSize, tilingData.incSize, tilingData.formerNum);
    } else if (TILING_KEY_IS(3)) {
        RunErfinv<float>(x, y, tilingData.smallSize, tilingData.incSize, tilingData.formerNum);
    }
}
