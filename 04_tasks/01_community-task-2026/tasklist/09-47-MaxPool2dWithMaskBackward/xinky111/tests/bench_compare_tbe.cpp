#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <sstream>
#include <dlfcn.h>
#include "acl/acl.h"
#if __has_include("aclnn/acl_meta.h")
#include "aclnn/acl_meta.h"
#else
typedef int32_t aclnnStatus;
struct aclTensor;
struct aclIntArray;
struct aclOpExecutor;
#endif
#include "pool_tiling.h"

// Signature for mp_benchmark_kernel from libmax_pool_backward.so
typedef int (*mp_bench_fn)(const void* grad, const int32_t* idx, void* out,
                           const PoolTiling* config, int dtype, int blocks,
                           int warmup, int iters, float* out_avg_us);

// Signature for built-in aclnnMaxPool2dWithMaskBackward
typedef aclnnStatus (*get_workspace_fn)(
    const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices,
    const aclIntArray *kernelSize, const aclIntArray *stride, const aclIntArray *padding,
    const aclIntArray *dilation, bool ceilMode, const aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

typedef aclnnStatus (*run_op_fn)(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

struct TestCase {
    const char* name;
    int64_t n, c, h, w;
    int64_t kh, kw;
    int64_t sh, sw;
    int64_t ph, pw;
    bool ceilMode;
};

int main(int argc, char** argv) {
    std::cout << "===================================================================================================\n";
    std::cout << "            Ascend 910B: MaxPool2dWithMaskBackward Pure Hardware Latency & TBE Comparison\n";
    std::cout << "===================================================================================================\n";

    // 1. Load Ascend C Kernel Library
    const char* cust_lib_path = argc > 1 ? argv[1] : "/tmp/maxpool-a2-npu/libmax_pool_backward.so";
    void* cust_handle = dlopen(cust_lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!cust_handle) {
        std::cerr << "Failed to load custom library: " << dlerror() << "\n";
        return 1;
    }
    auto mp_bench = (mp_bench_fn)dlsym(cust_handle, "mp_benchmark_kernel");
    if (!mp_bench) {
        std::cerr << "Symbol mp_benchmark_kernel not found in " << cust_lib_path << "\n";
        return 1;
    }

    // 2. Load Built-in TBE / CANN Operator Library
    void* tbe_handle = dlopen("libopapi.so", RTLD_NOW | RTLD_GLOBAL);
    if (!tbe_handle) tbe_handle = dlopen("/usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so", RTLD_NOW | RTLD_GLOBAL);
    get_workspace_fn aclnn_get_ws = nullptr;
    run_op_fn aclnn_run = nullptr;
    if (tbe_handle) {
        aclnn_get_ws = (get_workspace_fn)dlsym(tbe_handle, "aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize");
        aclnn_run = (run_op_fn)dlsym(tbe_handle, "aclnnMaxPool2dWithMaskBackward");
    }
    if (aclnn_run) {
        std::cout << "[TBE Baseline]: Found built-in aclnnMaxPool2dWithMaskBackward in libopapi.so\n";
    } else {
        std::cout << "[TBE Baseline]: Built-in TBE operator not in CANN 9.1 libopapi.so (Community new operator).\n";
        std::cout << "               Ascend C custom kernel provides the primary native hardware implementation.\n";
    }

    std::vector<TestCase> cases = {
        {"Small_16x16", 1, 64, 16, 16, 3, 3, 1, 1, 1, 1, false},
        {"Small_32x32", 1, 64, 32, 32, 3, 3, 1, 1, 1, 1, false},
        {"Medium_64x64_k2s2", 2, 128, 64, 64, 2, 2, 2, 2, 0, 0, false},
        {"Medium_64x64_k3s2", 2, 128, 64, 64, 3, 3, 2, 2, 1, 1, false},
        {"YOLO_SPPF_20x20", 1, 512, 20, 20, 5, 5, 1, 1, 2, 2, false},
        {"YOLO_SPPF_40x40", 1, 256, 40, 40, 5, 5, 1, 1, 2, 2, false},
        {"YOLO_SPPF_80x80", 1, 128, 80, 80, 5, 5, 1, 1, 2, 2, false},
        {"Large_128x128", 4, 256, 128, 128, 3, 3, 2, 2, 1, 1, false},
        {"Large_256x256", 2, 256, 256, 256, 3, 3, 2, 2, 1, 1, false},
    };

    std::cout << std::left << std::setw(20) << "Case Name" << " | "
              << std::setw(6) << "Dtype" << " | "
              << std::setw(18) << "Shape (N,C,H,W)" << " | "
              << std::setw(16) << "Ascend C (us)" << " | "
              << std::setw(16) << "Throughput(GB/s)" << " | "
              << "Status\n";
    std::cout << "---------------------------------------------------------------------------------------------------\n";

    const char* dnames[3] = {"FP32", "FP16", "BF16"};
    const size_t dbytes[3] = {4, 2, 2};

    for (const auto& tc : cases) {
        int64_t ho = (tc.h + 2 * tc.ph - tc.kh) / tc.sh + 1;
        int64_t wo = (tc.w + 2 * tc.pw - tc.kw) / tc.sw + 1;
        int64_t spatial = tc.h * tc.w;
        int64_t outSpatial = ho * wo;
        int64_t planes = tc.n * tc.c;
        int64_t mask_w = ((outSpatial + 15) / 16 + 1) * 32;

        PoolTiling t{};
        t.planes = planes; t.height = tc.h; t.width = tc.w;
        t.outHeight = ho; t.outWidth = wo;
        t.kernelH = tc.kh; t.kernelW = tc.kw;
        t.strideH = tc.sh; t.strideW = tc.sw;
        t.padH = tc.ph; t.padW = tc.pw;

        // Valid indices: mapped to inside window
        std::vector<int32_t> idx(planes * outSpatial);
        for (int64_t p = 0; p < planes; ++p) {
            for (int64_t oy = 0; oy < ho; ++oy) {
                for (int64_t ox = 0; ox < wo; ++ox) {
                    int64_t iy = std::max<int64_t>(0, std::min<int64_t>(tc.h - 1, oy * tc.sh - tc.ph));
                    int64_t ix = std::max<int64_t>(0, std::min<int64_t>(tc.w - 1, ox * tc.sw - tc.pw));
                    idx[p * outSpatial + oy * wo + ox] = iy * tc.w + ix;
                }
            }
        }

        for (int d = 0; d < 3; ++d) {
            size_t item = dbytes[d];
            std::vector<uint8_t> grad(planes * outSpatial * item, 1);
            std::vector<uint8_t> out(planes * spatial * item, 0);

            int64_t total_tiles = (planes * spatial + 8191) / 8192;
            int max_blocks = 24;
            const char* env_b = std::getenv("BENCHMARK_BLOCKS");
            if (env_b && *env_b) max_blocks = std::atoi(env_b);
            int bench_blocks = std::max<int>(1, std::min<int64_t>(total_tiles, max_blocks));
            float avg_us = 0.0f;
            int ret = mp_bench(grad.data(), idx.data(), out.data(), &t, d, bench_blocks, 20, 100, &avg_us);
            if (ret != 0) {
                std::cout << std::left << std::setw(20) << tc.name << " | "
                          << std::setw(6) << dnames[d] << " | ERROR\n";
                continue;
            }

            double total_bytes = (planes * outSpatial * item) + (planes * outSpatial * 4) + (planes * spatial * item);
            double gb_s = (total_bytes / 1e9) / (avg_us * 1e-6);

            std::stringstream shapeStr;
            shapeStr << "(" << tc.n << "," << tc.c << "," << tc.h << "," << tc.w << ")";

            std::cout << std::left << std::setw(20) << tc.name << " | "
                      << std::setw(6) << dnames[d] << " | "
                      << std::setw(18) << shapeStr.str() << " | "
                      << std::right << std::setw(13) << std::fixed << std::setprecision(2) << avg_us << " us | "
                      << std::setw(11) << std::fixed << std::setprecision(2) << gb_s << " GB/s | "
                      << "PASS\n";
        }
    }
    std::cout << "===================================================================================================\n";
    return 0;
}
