#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include "acl/acl_base.h"
#include "aclnn_cumsum.h"

static uint16_t FloatToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exponent = ((bits >> 23) & 0xFFu);
    uint32_t mantissa = bits & 0x007FFFFFu;
    uint16_t half;
    if (exponent == 255u) {
        half = static_cast<uint16_t>(sign | 0x7C00u | (mantissa ? 0x0200u : 0));
    } else if (exponent > 142u) {
        half = static_cast<uint16_t>(sign | 0x7C00u);
    } else if (exponent < 113u) {
        uint32_t shifted = (mantissa | 0x00800000u) >> (114u - exponent);
        half = static_cast<uint16_t>(sign | (shifted + 0x00001000u >> 13u));
    } else {
        uint32_t exp16 = exponent - 112u;
        uint32_t man16 = mantissa >> 13u;
        half = static_cast<uint16_t>(sign | (exp16 << 10u) | man16);
    }
    return half;
}

static float HalfToFloat(uint16_t half) {
    uint32_t sign = (half & 0x8000u) << 16u;
    uint32_t exponent = (half & 0x7C00u) >> 10u;
    uint32_t mantissa = half & 0x03FFu;
    uint32_t bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            exponent = 1u;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                exponent--;
            }
            mantissa &= 0x03FFu;
            bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1Fu) {
        bits = sign | 0x7F800000u | (mantissa << 13u);
    } else {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t FloatToBFloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static float BFloat16ToFloat(uint16_t raw) {
    uint32_t bits = static_cast<uint32_t>(raw) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static int64_t GetTypeByteSize(int dtype) {
    switch (dtype) {
        case ACL_FLOAT: return 4;
        case ACL_FLOAT16: return 2;
#ifdef ACL_FLOAT_BF16
        case ACL_FLOAT_BF16: return 2;
#endif
        case ACL_INT32: return 4;
        case ACL_INT64: return 8;
        case ACL_UINT8: return 1;
        case ACL_BOOL: return 1;
        default: return 0;
    }
}

static void WriteValue(void* dest, int dtype, double value) {
    switch (dtype) {
        case ACL_FLOAT:
            *reinterpret_cast<float*>(dest) = static_cast<float>(value);
            break;
        case ACL_FLOAT16:
            *reinterpret_cast<uint16_t*>(dest) = FloatToHalf(static_cast<float>(value));
            break;
#ifdef ACL_FLOAT_BF16
        case ACL_FLOAT_BF16:
            *reinterpret_cast<uint16_t*>(dest) = FloatToBFloat16(static_cast<float>(value));
            break;
#endif
        case ACL_INT32:
            *reinterpret_cast<int32_t*>(dest) = static_cast<int32_t>(value);
            break;
        case ACL_INT64:
            *reinterpret_cast<int64_t*>(dest) = static_cast<int64_t>(value);
            break;
        default:
            break;
    }
}

static double ReadValue(const void* src, int dtype) {
    switch (dtype) {
        case ACL_FLOAT:
            return static_cast<double>(*reinterpret_cast<const float*>(src));
        case ACL_FLOAT16:
            return static_cast<double>(HalfToFloat(*reinterpret_cast<const uint16_t*>(src)));
#ifdef ACL_FLOAT_BF16
        case ACL_FLOAT_BF16:
            return static_cast<double>(BFloat16ToFloat(*reinterpret_cast<const uint16_t*>(src)));
#endif
        case ACL_INT32:
            return static_cast<double>(*reinterpret_cast<const int32_t*>(src));
        case ACL_INT64:
            return static_cast<double>(*reinterpret_cast<const int64_t*>(src));
        default:
            return 0.0;
    }
}

static std::vector<double> CpuCumsum(
        const std::vector<double>& input,
        const std::vector<int64_t>& dims,
        int dim,
        bool exclusive,
        bool reverse) {
    size_t total = input.size();
    std::vector<double> output(total, 0.0);
    int rank = static_cast<int>(dims.size());
    int64_t left = 1;
    for (int i = 0; i < dim; ++i) {
        left *= dims[i];
    }
    int64_t axis = dims[dim];
    int64_t right = 1;
    for (int i = dim + 1; i < rank; ++i) {
        right *= dims[i];
    }
    for (int64_t i = 0; i < left; ++i) {
        for (int64_t j = 0; j < right; ++j) {
            for (int64_t k = 0; k < axis; ++k) {
                int64_t idx = i * axis * right + k * right + j;
                if (!reverse) {
                    if (k == 0) {
                        output[idx] = exclusive ? 0.0 : input[idx];
                    } else {
                        output[idx] = output[i * axis * right + (k - 1) * right + j]
                                      + (exclusive ? input[i * axis * right + (k - 1) * right + j]
                                                   : input[idx]);
                    }
                    if (exclusive && k == 0) {
                        output[idx] = 0.0;
                    } else if (exclusive && k > 0) {
                        double prev = output[i * axis * right + (k - 1) * right + j];
                        output[idx] = prev + input[i * axis * right + (k - 1) * right + j];
                    }
                } else {
                    if (k == axis - 1) {
                        output[idx] = exclusive ? 0.0 : input[idx];
                    } else {
                        output[idx] = output[i * axis * right + (k + 1) * right + j]
                                      + (exclusive ? input[i * axis * right + (k + 1) * right + j]
                                                   : input[idx]);
                    }
                    if (exclusive && k == axis - 1) {
                        output[idx] = 0.0;
                    } else if (exclusive && k < axis - 1) {
                        double next = output[i * axis * right + (k + 1) * right + j];
                        output[idx] = next + input[i * axis * right + (k + 1) * right + j];
                    }
                }
            }
        }
    }
    return output;
}

static bool ExecuteCumsumOp(
        const void* input_raw,
        const std::vector<int64_t>& dims,
        int dim,
        int input_dtype,
        int output_dtype,
        bool use_v2,
        bool exclusive,
        bool reverse,
        void* output_raw) {
    aclTensorDesc* input_desc = aclCreateTensorDesc(
            static_cast<aclDataType>(input_dtype),
            static_cast<int>(dims.size()),
            dims.data(),
            ACL_FORMAT_ND);
    aclTensorDesc* output_desc = aclCreateTensorDesc(
            static_cast<aclDataType>(output_dtype),
            static_cast<int>(dims.size()),
            dims.data(),
            ACL_FORMAT_ND);
    if (input_desc == nullptr || output_desc == nullptr) {
        if (input_desc) aclDestroyTensorDesc(input_desc);
        if (output_desc) aclDestroyTensorDesc(output_desc);
        return false;
    }
    aclError ret = ACL_ERROR_NONE;
    if (use_v2) {
        ret = aclnnCumsumV2(input_desc,
                            input_raw,
                            dim,
                            output_dtype,
                            exclusive,
                            reverse,
                            output_desc,
                            output_raw);
    } else {
        ret = aclnnCumsum(input_desc,
                          input_raw,
                          dim,
                          output_dtype,
                          output_desc,
                          output_raw);
    }
    aclDestroyTensorDesc(input_desc);
    aclDestroyTensorDesc(output_desc);
    return ret == ACL_ERROR_NONE;
}

static bool RunTestCase(
        const std::string& name,
        const std::vector<double>& input_values,
        const std::vector<int64_t>& dims,
        int dim,
        int input_dtype,
        int output_dtype,
        bool use_v2,
        bool exclusive,
        bool reverse,
        double atol,
        double rtol) {
    size_t element_count = input_values.size();
    int64_t in_size = GetTypeByteSize(input_dtype) * element_count;
    int64_t out_size = GetTypeByteSize(output_dtype) * element_count;
    std::vector<uint8_t> input_buffer(in_size);
    std::vector<uint8_t> output_buffer(out_size);
    for (size_t i = 0; i < element_count; ++i) {
        WriteValue(input_buffer.data() + i * GetTypeByteSize(input_dtype),
                   input_dtype,
                   input_values[i]);
    }
    bool ok = ExecuteCumsumOp(
            input_buffer.data(),
            dims,
            dim,
            input_dtype,
            output_dtype,
            use_v2,
            exclusive,
            reverse,
            output_buffer.data());
    if (!ok) {
        std::cout << name << " [FAIL] operator execution failed\n";
        return false;
    }
    std::vector<double> expected = CpuCumsum(input_values, dims, dim, exclusive, reverse);
    double max_error = 0.0;
    size_t max_index = 0;
    bool passed = true;
    for (size_t i = 0; i < element_count; ++i) {
        double actual = ReadValue(output_buffer.data() + i * GetTypeByteSize(output_dtype), output_dtype);
        double diff = std::fabs(actual - expected[i]);
        if (diff > max_error) {
            max_error = diff;
            max_index = i;
        }
        if (diff > atol + rtol * std::fabs(expected[i])) {
            passed = false;
        }
    }
    std::cout << "Test case: " << name << "\n"
              << "  Shape: [";
    for (size_t i = 0; i < dims.size(); ++i) {
        std::cout << dims[i] << (i + 1 < dims.size() ? ", " : "");
    }
    std::cout << "] dim=" << dim
              << " input_dtype=" << input_dtype
              << " output_dtype=" << output_dtype
              << " exclusive=" << std::boolalpha << exclusive
              << " reverse=" << std::boolalpha << reverse
              << "\n"
              << "  Max error: " << std::scientific << max_error
              << " at index " << max_index << "\n"
              << "  Result: " << (passed ? "[PASS]" : "[FAIL]") << "\n\n";
    return passed;
}

int main() {
    std::vector<std::pair<std::string, bool>> test_entries;
    int passed = 0;
    int total = 0;
    std::vector<double> seq10(10);
    for (int i = 0; i < 10; ++i) {
        seq10[i] = static_cast<double>(i + 1);
    }
    std::vector<double> alternating;
    for (int i = 0; i < 100; ++i) {
        alternating.push_back((i % 2 == 0) ? 1e8 : 1e-6);
    }
    std::vector<double> long_seq(10000, 1.0);
    std::vector<double> mixed_small = {1.0, -1.0, 2.0, -2.0, 0.5, -0.5};
    std::vector<double> int_values = {1, -2, 3, -4, 5, 6, -7, 8, -9, 10};

    struct Case {
        std::string name;
        std::vector<double> input;
        std::vector<int64_t> dims;
        int dim;
        int input_dtype;
        int output_dtype;
        bool use_v2;
        bool exclusive;
        bool reverse;
        double atol;
        double rtol;
    };
    std::vector<Case> cases = {
        // 基础数据类型和形状测试
        {"float32 basic", seq10, {10}, 0, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        {"float32 long sequence", long_seq, {10000}, 0, ACL_FLOAT, ACL_FLOAT, false, false, false, 2e-3, 1e-5},
        {"float32 2D dim=1", {1,2,3,4,5, 1,2,3,4,5}, {2,5}, 1, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        {"float32 3D dim=2", {2,3,4, 2,3,4, 2,3,4}, {2,3,3}, 2, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        
        // float16数据类型测试
        {"float16 reverse exclusive", mixed_small, {6}, 0, ACL_FLOAT16, ACL_FLOAT16, true, true, true, 1e-3, 1e-3},
        {"float16 2D normal", {1,2,3,4}, {2,2}, 1, ACL_FLOAT16, ACL_FLOAT16, false, false, false, 1e-3, 1e-3},
        
        // int32数据类型测试
        {"int32 exact path", int_values, {10}, 0, ACL_INT32, ACL_INT32, true, false, false, 0.0, 0.0},
        {"int32 2D", {1,-2,3,-4, 5,6,-7,8}, {2,4}, 1, ACL_INT32, ACL_INT32, false, false, false, 0.0, 0.0},
        {"int32 large dim", long_seq, {1000}, 0, ACL_INT32, ACL_INT32, false, false, false, 0.0, 0.0},
        
        // int64数据类型测试
        {"int64 output from int32 input", int_values, {10}, 0, ACL_INT32, ACL_INT64, true, false, true, 0.0, 0.0},
        {"int64 native", {1LL,2LL,3LL,4LL}, {4}, 0, ACL_INT64, ACL_INT64, false, false, false, 0.0, 0.0},
        
        // uint8数据类型测试
        {"uint8 small", {1,2,3,4,5}, {5}, 0, ACL_UINT8, ACL_UINT8, false, false, false, 0.0, 0.0},
        {"uint8 exclusive reverse", {10,20,30,40}, {4}, 0, ACL_UINT8, ACL_UINT8, true, true, false, 0.0, 0.0},
        
        // int8数据类型测试
        {"int8", {-10,-20,30,40}, {4}, 0, ACL_INT8, ACL_INT8, false, false, false, 0.0, 0.0},
        
        // 边界值测试
        {"0 dim tensor", {1}, {}, 0, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        {"empty tensor", {2,0}, {2,0}, 0, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        
        // 大shape测试（触发不同的tiling策略）
        {"large shape M", long_seq, {100, 100}, 0, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-3},
        {"large shape N", long_seq, {10, 1000}, 1, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-3},
        {"large shape MN", long_seq, {100, 50, 20}, 1, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-3},
        
        // exclusive和reverse组合测试
        {"exclusive only", {1,2,3,4}, {4}, 0, ACL_FLOAT, ACL_FLOAT, true, false, false, 1e-5, 1e-5},
        {"reverse only", {4,3,2,1}, {4}, 0, ACL_FLOAT, ACL_FLOAT, false, true, false, 1e-5, 1e-5},
        {"both exclusive and reverse", {1,2,3,4}, {4}, 0, ACL_FLOAT, ACL_FLOAT, true, true, false, 1e-5, 1e-5},
        
        // V2 API测试
        {"v2 api float", {1,2,3,4,5}, {5}, 0, ACL_FLOAT, ACL_FLOAT, true, false, false, 1e-5, 1e-5},
        {"v2 api float16", {1,2,3}, {3}, 0, ACL_FLOAT16, ACL_FLOAT16, true, true, true, 1e-3, 1e-3},
        
        // 边界dim值测试
        {"dim negative", {1,2,3,4}, {4}, -1, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        {"dim large", {1,2,3,4,5,6,7,8}, {8}, 7, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        
#ifdef ACL_FLOAT_BF16
        // BF16数据类型测试
        {"bf16 mixed magnitude", alternating, {100}, 0, ACL_FLOAT_BF16, ACL_FLOAT_BF16, true, false, false, 1e-2, 1e-2},
        {"bf16 2D", {1,2,3,4, 5,6,7,8}, {2,4}, 1, ACL_FLOAT_BF16, ACL_FLOAT_BF16, false, true, false, 1e-2, 1e-2},
#endif
        
        // 8D张量测试（接近MAX_DIM_LEN边界）
        {"8D tensor", std::vector<double>(16, 1.0), {2,2,2,2,1,1,2,2}, 3, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-5, 1e-5},
        
        // 复杂形状测试（触发不同tiling策略）
        {"complex shape 1", long_seq, {16, 64, 32}, 1, ACL_FLOAT, ACL_FLOAT, false, false, false, 1e-3, 1e-3},
        {"complex shape 2", long_seq, {8, 128, 16}, 2, ACL_FLOAT, ACL_FLOAT, true, false, true, 1e-3, 1e-3},
    };

    for (auto& c : cases) {
        ++total;
        bool ok = RunTestCase(c.name,
                              c.input,
                              c.dims,
                              c.dim,
                              c.input_dtype,
                              c.output_dtype,
                              c.use_v2,
                              c.exclusive,
                              c.reverse,
                              c.atol,
                              c.rtol);
        if (ok) {
            ++passed;
        }
    }

    std::cout << "Summary: " << passed << " passed, " << (total - passed) << " failed\n";
    return (passed == total) ? 0 : 1;
}
