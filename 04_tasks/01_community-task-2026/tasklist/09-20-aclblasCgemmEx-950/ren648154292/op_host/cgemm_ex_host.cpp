/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_blas.h"

// ==========================================================================
//  aclblasCgemmEx -- arch35 host side (iteration 4: upstream thin wrapper)
//
//  C = alpha * op(A) * op(B) + beta * C, column major (CblasColMajor).
//
//  Iteration 1-3 shipped a hand-written 3-GEMM (Karatsuba) complex kernel
//  (cgemm_ex_kernel.cpp / _kernel.h / _tiling_data.h). It produced wrong
//  numerics and was withdrawn; those three files are deleted. aclblasCgemm
//  and aclblasSgemm in blas/gemm/arch35/gemm_host.cpp are already validated
//  in this repository, so this file is now a thin wrapper:
//      parameter validation -> type dispatch -> delegate to upstream.
//  There is no tiling, no workspace planning and no device launch of our own.
//
//  Supported type sets:
//    ACLBLAS_C_32 -> aclblasCgemm (complex64 in, complex64 out)
//    ACLBLAS_R_32 -> aclblasSgemm (real part of alpha/beta, float in/out)
//  typeA / typeB / typeC must all be the same type, otherwise
//  ACLBLAS_STATUS_INVALID_VALUE. ACLBLAS_H_R_32 / ACLBLAS_H_C_32 are out of
//  scope and return ACLBLAS_STATUS_NOT_SUPPORTED.
//
//  alpha / beta are host-resident scalars for both aclblasCgemmEx and
//  aclblasCgemm, so the C_32 path forwards them untouched. aclblasSgemm takes
//  host float* scalars, so the R_32 path copies alpha->real / beta->real into
//  stack locals before delegating.
//
//  Pointer validation deliberately mirrors the upstream contract rather than
//  requiring every buffer unconditionally: with k == 0 the GEMM degenerates to
//  C = beta * C and A / B (and, when beta == 0, C) may be nullptr. Requiring
//  them would reject calls that upstream aclblasCgemm / aclblasSgemm handle
//  correctly, so the checks are conditional on k > 0 and on alpha / beta.
// ==========================================================================

namespace {

const char* const CGEMM_EX_LOG_TAG = "aclblasCgemmEx";

// ---------------------------------------------------------------------------
//  Upstream declarations
//  Both definitions live in blas/gemm/arch35/gemm_host.cpp, which is compiled
//  into the same libops_blas target, so the symbols are always present. They
//  are also already declared in cann_ops_blas.h (inside its extern "C" block);
//  the restatements below keep this wrapper's dependency explicit and do not
//  change linkage.
// ---------------------------------------------------------------------------
extern "C" aclblasStatus_t aclblasCgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C, int ldc);

extern "C" aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);

inline bool CgemmExIsLegalOp(aclblasOperation_t op)
{
    return (op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C);
}

// Recognised aclblasType_t values only. 0xFF (and any other unknown value) is
// rejected as INVALID_ENUM, which is distinct from the mixed-dtype INVALID_VALUE
// case below.
inline bool CgemmExIsKnownType(aclblasType_t type)
{
    return (type == ACLBLAS_R_32 || type == ACLBLAS_C_32 || type == ACLBLAS_H_R_32 ||
            type == ACLBLAS_H_C_32);
}

// ---------------------------------------------------------------------------
//  ValidateCgemmExParams
//  Order follows the upstream ValidateGemmParams / ValidateGemmPointers style:
//  handle, trans, dimensions, leading dims, scalars, then buffers.
// ---------------------------------------------------------------------------
aclblasStatus_t ValidateCgemmExParams(aclblasHandle_t handle, aclblasOperation_t transa,
    aclblasOperation_t transb, int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* beta, aclblasType_t typeA, aclblasType_t typeB, aclblasType_t typeC,
    const void* A, const void* B, void* C, int lda, int ldb, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!CgemmExIsLegalOp(transa)) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "invalid transa: %d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (!CgemmExIsLegalOp(transb)) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "invalid transb: %d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (!CgemmExIsKnownType(typeA) || !CgemmExIsKnownType(typeB) || !CgemmExIsKnownType(typeC)) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "unsupported type combination: typeA=%d, typeB=%d, typeC=%d",
                static_cast<int>(typeA), static_cast<int>(typeB), static_cast<int>(typeC));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }

    if (m < 0 || n < 0 || k < 0) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "m, n, k must be non-negative: m=%d, n=%d, k=%d", m, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // A/B/C must all be the same type. Kept separate from the enum check above
    // so that out-of-range values keep returning INVALID_ENUM.
    if (typeA != typeB || typeA != typeC) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "mixed dtype is not supported: typeA=%d, typeB=%d, typeC=%d",
                static_cast<int>(typeA), static_cast<int>(typeB), static_cast<int>(typeC));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // m == 0 or n == 0 is a legal no-op: nothing is computed. This matches
    // aclblasCgemm / aclblasSgemm and keeps the no-op precedence above the
    // pointer and dtype-scope checks.
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Half-precision variants are declared in the public API but out of scope
    // for this wrapper.
    if (typeA != ACLBLAS_C_32 && typeA != ACLBLAS_R_32) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "dtype %d is not supported by aclblasCgemmEx, "
                "only ACLBLAS_C_32 and ACLBLAS_R_32 are", static_cast<int>(typeA));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    int ldaMin = (transa == ACLBLAS_OP_N) ? std::max(1, m) : std::max(1, k);
    if (lda < ldaMin) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "lda=%d must be >= %d", lda, ldaMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int ldbMin = (transb == ACLBLAS_OP_N) ? std::max(1, k) : std::max(1, n);
    if (ldb < ldbMin) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "ldb=%d must be >= %d", ldb, ldbMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "ldc=%d must be >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (alpha == nullptr) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Pointer requirements are conditional, mirroring the upstream
    // ValidateGemmPointers contract exactly so that k == 0 callers (which
    // legitimately omit A / B / C) reach the upstream beta-only fast path
    // instead of being rejected here.
    //   A / B required only when k > 0 and alpha != 0
    //   C    required when k > 0, or when beta != 0
    float alphaAbs = std::abs(alpha->real) + std::abs(alpha->imag);
    float betaAbs = std::abs(beta->real) + std::abs(beta->imag);
    if (k > 0 && alphaAbs != 0.0f) {
        if (A == nullptr) {
            OP_LOGE(CGEMM_EX_LOG_TAG, "A must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (B == nullptr) {
            OP_LOGE(CGEMM_EX_LOG_TAG, "B must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if ((k > 0 || betaAbs != 0.0f) && C == nullptr) {
        OP_LOGE(CGEMM_EX_LOG_TAG, "C must not be nullptr when k > 0 or beta != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

}  // namespace

// ==========================================================================
//  aclblasCgemmEx -- public API entry
//  Signature stays exactly as declared in include/cann_ops_blas.h.
// ==========================================================================
extern "C" aclblasStatus_t aclblasCgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const aclblasComplex* alpha,
    const void* A, aclblasType_t typeA, int lda,
    const void* B, aclblasType_t typeB, int ldb,
    const aclblasComplex* beta, void* C, aclblasType_t typeC, int ldc)
{
    aclblasStatus_t st = ValidateCgemmExParams(
        handle, transa, transb, m, n, k, alpha, beta, typeA, typeB, typeC, A, B, C, lda, ldb, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    OP_LOGI(CGEMM_EX_LOG_TAG, "entry: typeA=%d, transa=%d, transb=%d, m=%d, n=%d, k=%d",
            static_cast<int>(typeA), static_cast<int>(transa), static_cast<int>(transb), m, n, k);

    if (typeA == ACLBLAS_C_32) {
        // alpha / beta are host pointers for both interfaces: forward directly.
        return aclblasCgemm(handle, transa, transb, m, n, k, alpha,
            reinterpret_cast<const aclblasComplex*>(A), lda,
            reinterpret_cast<const aclblasComplex*>(B), ldb,
            beta, reinterpret_cast<aclblasComplex*>(C), ldc);
    }

    // ACLBLAS_R_32: real-valued GEMM. Take the real parts of the complex
    // scalars; alpha->imag / beta->imag are intentionally dropped, matching
    // the real GEMM semantics of aclblasSgemm.
    float alphaReal = alpha->real;
    float betaReal = beta->real;
    return aclblasSgemm(handle, transa, transb, m, n, k, &alphaReal,
        reinterpret_cast<const float*>(A), lda,
        reinterpret_cast<const float*>(B), ldb,
        &betaReal, reinterpret_cast<float*>(C), ldc);
}

#endif  // ASC_DEVKIT_GE_9_1
