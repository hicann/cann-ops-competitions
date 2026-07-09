#!/usr/bin/env python3
"""Standalone cuSPARSE SDDMM GPU test suite (200 cases from cases/sddmm_cases.json).

Usage:
  python3 run_sddmm_gpu.py
  python3 run_sddmm_gpu.py --start 0 --end 20
  python3 run_sddmm_gpu.py --skip-perf-only
  python3 run_sddmm_gpu.py --m 512 --n 512 --k 64 --dtype fp16

Requires: PyTorch(CUDA), numpy, libcusparse.so.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'lib'))

import torch

from case_builder import build_sddmm_case
from runner_common import default_cases_path, load_case_specs, run_case_suite, slice_cases
from sparse_common import (
    logical_dense_from_tensor,
    max_abs_diff,
    require_cuda,
    sddmm_masked_golden,
    to_result_tensor,
)
from sparse_gpu_cusparse import cusparse_sddmm, resolve_cusparse_lib


def cpu_reference(m, n, k, alpha, beta, dense_a, dense_b, dense_c,
                  op_a, op_b, c_row, c_col, ab_dtype, c_dtype,
                  out_dtype=torch.float32):
    A = logical_dense_from_tensor(dense_a, op_a, m, k)
    B = logical_dense_from_tensor(dense_b, op_b, k, n)
    uniform_fp32 = ab_dtype == 'fp32' and c_dtype == 'fp32'
    high_precision = uniform_fp32

    # Masked golden: O(nnz*k) — required for S2 (10⁵×10⁵) accuracy without ~80GB RAM.
    result = sddmm_masked_golden(
        A, B, m, n, c_row, c_col, alpha, beta=beta, dense_c=dense_c,
        ab_dtype=ab_dtype, c_dtype=c_dtype, high_precision=high_precision,
    )
    if high_precision:
        return to_result_tensor(result, out_dtype, high_precision=True)
    return torch.from_numpy(result).to(out_dtype)


def gpu_call(case):
    if case['nnz_c'] == 0:
        return torch.zeros(case['m'], case['n'], dtype=torch.float32)
    return cusparse_sddmm(
        case['m'], case['n'], case['k'], case['alpha'], case['beta'],
        case['dense_a'], case['dense_b'],
        case['op_a'], case['op_b'], case['ld_a'], case['ld_b'],
        case['order_a'], case['order_b'],
        case['c_row'], case['c_col'], case['c_vals'],
        ab_dtype_name=case['ab_dtype'], c_dtype_name=case['c_dtype'],
    )


def run_one(case, name, spec=None, rtol=1e-3, atol=1e-3):
    cat = case.get('category', spec.get('category', '') if spec else '')
    print(f"\n=== [{name}] cat={cat} m={case['m']} n={case['n']} k={case['k']} "
          f"ab={case['ab_dtype']} c={case['c_dtype']} beta={case['beta']} nnz={case['nnz_c']} ===")

    t0 = time.time()
    out = gpu_call(case)
    gpu_ms = (time.time() - t0) * 1000

    ref = cpu_reference(
        case['m'], case['n'], case['k'], case['alpha'], case['beta'],
        case['dense_a'], case['dense_b'], case['dense_c'],
        case['op_a'], case['op_b'], case['c_row'], case['c_col'],
        case['ab_dtype'], case['c_dtype'], out_dtype=torch.float32,
    )
    diff = max_abs_diff(out, ref)
    ok = diff <= atol + rtol * max(ref.abs().max().item(), 1.0)
    print(f"  max_abs_diff={diff:.6e} gpu_ms={gpu_ms:.1f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(f"SDDMM case failed: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description='cuSPARSE SDDMM GPU test suite')
    parser.add_argument('--cases', default=default_cases_path('sddmm', ROOT))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--skip-perf-only', action='store_true')
    parser.add_argument('--m', type=int, default=0, help='single custom case mode')
    parser.add_argument('--n', type=int, default=0)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32')
    args = parser.parse_args()

    require_cuda()
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"libcusparse: {resolve_cusparse_lib()}")

    if args.m > 0:
        m, n, k = args.m, args.n or args.m, args.k or 64
        ab = args.dtype
        c = 'fp32' if ab == 'fp16' else args.dtype
        spec = dict(name='custom', category='custom', seed=0,
                    m=m, n=n, k=k, sparsity=0.995, alpha=1.0, beta=0.5,
                    op_a=0, op_b=0, ab_dtype=ab, c_dtype=c)
        run_one(build_sddmm_case(spec), 'custom', spec=spec)
    else:
        specs = slice_cases(load_case_specs(args.cases), args.start, args.end)
        print(f"Loaded {len(specs)} SDDMM cases from {args.cases}")
        run_case_suite(specs, build_sddmm_case, run_one, skip_perf_only=args.skip_perf_only)

    print("\nAll SDDMM GPU tests passed.")


if __name__ == '__main__':
    main()
