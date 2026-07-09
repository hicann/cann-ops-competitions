#!/usr/bin/env python3
"""Standalone cuSPARSE SpGEMM GPU test suite (200 cases from cases/spgemm_cases.json).

Usage:
  python3 run_spgemm_gpu.py
  python3 run_spgemm_gpu.py --start 0 --end 20
  python3 run_spgemm_gpu.py --m 512 --dtype fp16 --beta 1.0

Requires: PyTorch(CUDA), numpy, scipy (for beta!=0), libcusparse.so.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'lib'))

import numpy as np
import torch

from case_builder import build_spgemm_case
from runner_common import default_cases_path, load_case_specs, run_case_suite, slice_cases
from sparse_common import max_abs_diff, require_cuda, to_result_tensor
from sparse_gpu_cusparse import cusparse_spgemm, resolve_cusparse_lib

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


def cpu_reference(m, k, n, alpha, beta, a_row, a_col, a_vals,
                  b_row, b_col, b_vals, c_row, c_col, c_vals, dtype_name,
                  out_dtype=torch.float32):
    if sp is None:
        raise RuntimeError("scipy is required for SpGEMM CPU reference")
    acc = np.float64 if dtype_name == 'fp32' else np.float32
    A = sp.csr_matrix((a_vals.astype(acc), a_col, a_row), shape=(m, k))
    B = sp.csr_matrix((b_vals.astype(acc), b_col, b_row), shape=(k, n))
    C = sp.csr_matrix((c_vals.astype(acc), c_col, c_row), shape=(m, n))
    out = alpha * (A @ B) + beta * C
    dense = out.toarray()
    if dtype_name == 'fp16':
        dense = dense.astype('float32').astype('float16').astype('float32')
    elif dtype_name == 'bf16':
        dense = torch.from_numpy(dense.astype('float32')).to(torch.bfloat16).to(torch.float32).numpy()
    if dtype_name == 'fp32':
        return to_result_tensor(dense, out_dtype, high_precision=True)
    return torch.from_numpy(np.asarray(dense, dtype=np.float32)).to(out_dtype)


def zero_nnz_result(case):
    if case['nnz_a'] > 0 and case['nnz_b'] > 0:
        return None
    out_dtype = {
        'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16,
    }[case['dtype_name']]
    if case['beta'] == 0.0:
        return torch.zeros(case['m'], case['n'], dtype=out_dtype)
    return cpu_reference(
        case['m'], case['k'], case['n'], case['alpha'], case['beta'],
        case['a_row'], case['a_col'], case['a_vals'],
        case['b_row'], case['b_col'], case['b_vals'],
        case['c_row'], case['c_col'], case['c_vals'],
        case['dtype_name'], out_dtype=out_dtype,
    )


def gpu_call(case):
    empty = zero_nnz_result(case)
    if empty is not None:
        return empty

    out_row = out_col = None
    if case['beta'] != 0.0:
        if sp is None:
            raise RuntimeError("scipy required for SpGEMM GPU when beta != 0")
        m, k, n = case['m'], case['k'], case['n']
        out_csr = (
            case['alpha'] * sp.csr_matrix(
                (case['a_vals'], case['a_col'], case['a_row']), shape=(m, k)) @
            sp.csr_matrix((case['b_vals'], case['b_col'], case['b_row']), shape=(k, n)) +
            case['beta'] * sp.csr_matrix(
                (case['c_vals'], case['c_col'], case['c_row']), shape=(m, n))
        ).tocsr()
        out_row = out_csr.indptr.astype('int32')
        out_col = out_csr.indices.astype('int32')

    out_dtype = {
        'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16,
    }[case['dtype_name']]
    return cusparse_spgemm(
        case['m'], case['k'], case['n'],
        case['alpha'], case['beta'],
        case['a_row'], case['a_col'], case['a_vals'],
        case['b_row'], case['b_col'], case['b_vals'],
        case['c_row'], case['c_col'], case['c_vals'],
        dtype_name=case['dtype_name'],
        out_dtype=out_dtype,
        out_row_off=out_row,
        out_col_ind=out_col,
    )


def _needs_scipy(spec):
    return float(spec.get('beta', 0)) != 0.0


def run_one(case, name, spec=None, rtol=1e-3, atol=1e-3):
    cat = case.get('category', spec.get('category', '') if spec else '')
    print(f"\n=== [{name}] cat={cat} m={case['m']} k={case['k']} n={case['n']} "
          f"dtype={case['dtype_name']} alpha={case['alpha']} beta={case['beta']} "
          f"nnz_a={case['nnz_a']} nnz_b={case['nnz_b']} ===")

    t0 = time.time()
    ref = zero_nnz_result(case)
    if ref is not None:
        out = ref
    else:
        out = gpu_call(case)
        ref = cpu_reference(
            case['m'], case['k'], case['n'], case['alpha'], case['beta'],
            case['a_row'], case['a_col'], case['a_vals'],
            case['b_row'], case['b_col'], case['b_vals'],
            case['c_row'], case['c_col'], case['c_vals'],
            case['dtype_name'],
            out_dtype={
                'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16,
            }[case['dtype_name']],
        )
    gpu_ms = (time.time() - t0) * 1000

    diff = max_abs_diff(out, ref)
    ok = diff <= atol + rtol * max(ref.abs().max().item(), 1.0)
    print(f"  max_abs_diff={diff:.6e} gpu_ms={gpu_ms:.1f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(f"SpGEMM case failed: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description='cuSPARSE SpGEMM GPU test suite')
    parser.add_argument('--cases', default=default_cases_path('spgemm', ROOT))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--n', type=int, default=0)
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()

    require_cuda()
    if sp is None:
        print("WARNING: scipy not installed; cases with beta!=0 will be skipped.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"libcusparse: {resolve_cusparse_lib()}")

    if args.m > 0:
        m, k, n = args.m, args.k or args.m, args.n or args.m
        spec = dict(name='custom', category='custom', seed=0,
                    m=m, k=k, n=n, sp_a=0.99, sp_b=0.99,
                    alpha=1.0, beta=args.beta, dtype_name=args.dtype)
        run_one(build_spgemm_case(spec), 'custom', spec=spec)
    else:
        specs = slice_cases(load_case_specs(args.cases), args.start, args.end)
        print(f"Loaded {len(specs)} SpGEMM cases from {args.cases}")
        run_case_suite(
            specs, build_spgemm_case, run_one,
            skip_scipy_required=(sp is None),
            needs_scipy_fn=_needs_scipy,
        )

    print("\nAll SpGEMM GPU tests passed.")


if __name__ == '__main__':
    main()
