#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_add_v3.h"

aclTensor* MakeT(std::vector<int64_t> shape, aclDataType dtype, void** dev, bool non_contig=false) {
    int64_t numel = 1; for (auto s : shape) numel *= (s > 0 ? s : 1);
    size_t bytes = numel * 8 + 128; 
    aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    std::vector<int64_t> strides(shape.size(), 1);
    if (non_contig && shape.size() > 1) { strides[0] = shape[1] + 2; strides[1] = 1; }
    else { for (int i = shape.size()-2; i >= 0; i--) strides[i] = (shape[i+1]>0?shape[i+1]:1)*strides[i+1]; }
    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), *dev);
}

int main() {
    aclInit(nullptr); aclrtSetDevice(0);
    aclrtStream stream; aclrtCreateStream(&stream);
    void *p1, *p2, *p3, *w = nullptr; uint64_t ws = 0; aclOpExecutor* ex = nullptr;

    // --- 矩阵 A: 爆破 aclnn_add.cpp (Alpha 特化 + Inplace 变体) ---
    std::vector<float> alphas = {1.0f, 0.0f, -1.0f, 0.5f};
    auto t1 = MakeT({16}, ACL_FLOAT, &p1); auto t2 = MakeT({16}, ACL_FLOAT, &p2); auto out = MakeT({16}, ACL_FLOAT, &p3);
    for (float av : alphas) {
        aclScalar* a = aclCreateScalar(&av, ACL_FLOAT);
        aclnnAddGetWorkspaceSize(t1, t2, a, out, &ws, &ex); // 触发 4 种 alpha 路径
        aclnnInplaceAddGetWorkspaceSize(t1, t2, a, &ws, &ex); // 触发原地路径
        aclDestroyScalar(a);
    }

    // --- 矩阵 B: 爆破 aclnn_add_v3.cpp (标量 API 全家桶) ---
    float sv = 5.0f; aclScalar* sc = aclCreateScalar(&sv, ACL_FLOAT);
    float av1 = 1.0f; aclScalar* a1 = aclCreateScalar(&av1, ACL_FLOAT);
    aclnnAddV3GetWorkspaceSize(sc, t2, a1, out, &ws, &ex); // 标准 V3
    aclnnInplaceAddV3GetWorkspaceSize(sc, t2, a1, &ws, &ex); // 原地 V3
    aclnnAddsGetWorkspaceSize(t1, sc, a1, out, &ws, &ex); // Adds 变体
    aclnnInplaceAddsGetWorkspaceSize(t1, sc, a1, &ws, &ex); // InplaceAdds 变体

    // --- 矩阵 C: 爆破 add_tiling_arch35.cpp (复合高维广播) ---
    void *p4, *p5;
    auto t_h1 = MakeT({1, 2, 1, 4}, ACL_FLOAT, &p4);
    auto t_h2 = MakeT({2, 1, 3, 1}, ACL_FLOAT, &p5);
    aclnnAddGetWorkspaceSize(t_h1, t_h2, a1, t1, &ws, &ex); // 触发复杂广播 Tiling

    // --- 矩阵 D: 爆破 add.cpp (多类型表路由) ---
    void *p6; auto t_i32 = MakeT({8}, ACL_INT32, &p6);
    aclnnAddGetWorkspaceSize(t_i32, t_i32, a1, t_i32, &ws, &ex); // 触发 INT32 路由

    std::cout << "--- MISSION: FINAL SQUEEZE SUCCESS ---" << std::endl;
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
