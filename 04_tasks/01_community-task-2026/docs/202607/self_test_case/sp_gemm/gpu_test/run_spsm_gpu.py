#!/usr/bin/env python3
"""Standalone cuSPARSE SpSM GPU test suite (200 cases from cases/spsm_cases.json).

Usage:
  python3 run_spsm_gpu.py
  python3 run_spsm_gpu.py --start 0 --end 20
  python3 run_spsm_gpu.py --m 512 --nrhs 8 --update-matrix

Requires: PyTorch(CUDA), numpy, libcusparse.so.
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

from case_builder import build_spsm_case
from runner_common import default_cases_path, load_case_specs, run_case_suite, slice_cases
from sparse_common import csr_to_dense, max_abs_diff, require_cuda, to_result_tensor
from sparse_gpu_cusparse import cusparse_spsm, resolve_cusparse_lib


def triangular_solve_numpy(A, B, lower=True, unit_diag=False, acc=np.float64):
    A = A.astype(acc, copy=False)
    m = A.shape[0]
    nrhs = B.shape[1]
    X = np.zeros((m, nrhs), dtype=acc)
    rhs = B.astype(acc, copy=False)
    if lower:
        for i in range(m):
            diag = acc(1.0) if unit_diag else A[i, i]
            X[i, :] = (rhs[i, :] - A[i, :i] @ X[:i, :]) / diag
    else:
        for i in range(m - 1, -1, -1):
            diag = acc(1.0) if unit_diag else A[i, i]
            X[i, :] = (rhs[i, :] - A[i, i + 1:] @ X[i + 1:, :]) / diag
    return X


def apply_op_b(B, op_b):
    if op_b == 0:
        return B
    return B.T.contiguous()


def cpu_reference(m, nrhs, alpha, a_row, a_col, a_vals, dense_b,
                  op_a, op_b, fill_mode, diag_type, update_matrix,
                  out_dtype=torch.float32):
    acc = np.float64
    a_vals_work = a_vals
    if update_matrix:
        a_vals_work = a_vals.copy().astype(np.float64)
        a_vals_work = a_vals_work + np.random.RandomState(0).uniform(
            -0.1, 0.1, size=a_vals_work.shape)
        a_vals_work = a_vals_work.astype(np.float32)

    A_dense = csr_to_dense(a_row, a_col, a_vals_work, (m, m), acc_dtype=acc)
    A_op = A_dense.T if op_a == 1 else A_dense
    is_lower = (fill_mode == 0 and op_a == 0) or (fill_mode == 1 and op_a == 1)
    B_in = apply_op_b(dense_b[:m, :nrhs], op_b).numpy().astype(acc, copy=False)
    X = triangular_solve_numpy(
        A_op, alpha * B_in, lower=is_lower, unit_diag=(diag_type == 1), acc=acc)
    return to_result_tensor(X, out_dtype, high_precision=True)


def gpu_call(case):
    m, nrhs = case['m'], case['nrhs']
    B_in = apply_op_b(case['dense_b'][:m, :nrhs], case['op_b'])
    C_out = B_in if case['in_place'] else case['dense_c'][:m, :nrhs]
    return cusparse_spsm(
        m, nrhs, case['alpha'],
        case['a_row'], case['a_col'], case['a_vals'],
        B_in, C_out,
        case['op_a'], case['op_b'], case['fill_mode'], case['diag_type'],
        in_place=case['in_place'],
        null_values=case['null_values'],
        update_matrix=case['update_matrix'],
    )


def run_one(case, name, spec=None, rtol=1e-4, atol=1e-4):
    flags = []
    if case['null_values']:
        flags.append('nullValues')
    if case['update_matrix']:
        flags.append('updateMatrix')
    if case['in_place']:
        flags.append('inPlace')
    flag_str = ','.join(flags) if flags else 'none'
    cat = case.get('category', spec.get('category', '') if spec else '')
    print(f"\n=== [{name}] cat={cat} m={case['m']} nrhs={case['nrhs']} "
          f"avg_deg={spec.get('avg_degree', '?') if spec else '?'} flags={flag_str} ===")

    t0 = time.time()
    out = gpu_call(case)
    gpu_ms = (time.time() - t0) * 1000
    ref = cpu_reference(
        case['m'], case['nrhs'], case['alpha'],
        case['a_row'], case['a_col'], case['a_vals'],
        case['dense_b'],
        case['op_a'], case['op_b'], case['fill_mode'], case['diag_type'],
        case['update_matrix'],
        out_dtype=torch.float32,
    )
    diff = max_abs_diff(out, ref)
    ok = diff <= atol + rtol * max(ref.abs().max().item(), 1.0)
    print(f"  max_abs_diff={diff:.6e} gpu_ms={gpu_ms:.1f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(f"SpSM case failed: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description='cuSPARSE SpSM GPU test suite')
    parser.add_argument('--cases', default=default_cases_path('spsm', ROOT))
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--nrhs', type=int, default=8)
    parser.add_argument('--null-values', action='store_true')
    parser.add_argument('--update-matrix', action='store_true')
    parser.add_argument('--in-place', action='store_true')
    args = parser.parse_args()

    require_cuda()
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"libcusparse: {resolve_cusparse_lib()}")

    if args.m > 0:
        spec = dict(
            name='custom', category='custom', seed=0,
            m=args.m, nrhs=args.nrhs, alpha=1.0, avg_degree=10.0,
            op_a=0, op_b=0, fill_mode=0, diag_type=0,
            in_place=args.in_place, null_values=args.null_values,
            update_matrix=args.update_matrix,
        )
        run_one(build_spsm_case(spec), 'custom', spec=spec)
    else:
        specs = slice_cases(load_case_specs(args.cases), args.start, args.end)
        print(f"Loaded {len(specs)} SpSM cases from {args.cases}")
        run_case_suite(specs, build_spsm_case, run_one)

    print("\nAll SpSM GPU tests passed.")


if __name__ == '__main__':
    main()
