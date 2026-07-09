#!/usr/bin/env python3
"""Benchmark cuSPARSE GPU performance for task-book fixed reference cases (3 per op).

GPU-only script. Times the same scope as ATK executor ``_gpu_call`` (full cuSPARSE
API path via ``run_*_gpu.gpu_call``). Paste results into task doc「GPU 参考耗时」.

Usage:
  python3 run_perf_gpu.py                  # all ops
  python3 run_perf_gpu.py --op sddmm
  python3 run_perf_gpu.py --op spgemm --warmup 5 --repeat 30
  python3 run_perf_gpu.py --markdown       # print paste-friendly table

Requires: PyTorch(CUDA), numpy, scipy (SpGEMM), libcusparse.so.
Correctness: use run_*_gpu.py (not this script).
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'lib'))

import torch

from case_builder import build_sddmm_case, build_spgemm_case, build_spsm_case
from gpu_timing import cuda_event_benchmark
from perf_cases import all_perf_cases
from sparse_gpu_cusparse import resolve_cusparse_lib

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


def _sddmm_fn(case):
    from run_sddmm_gpu import gpu_call
    def _run():
        gpu_call(case)
    return _run


def _spgemm_fn(case):
    from run_spgemm_gpu import gpu_call
    def _run():
        gpu_call(case)
    return _run


def _spsm_fn(case):
    from run_spsm_gpu import gpu_call

    def _run():
        gpu_call(case)

    return _run


def benchmark_op(op: str, warmup: int, repeat: int) -> list:
    results = []
    for i, spec in enumerate(all_perf_cases(op)):
        print(f"\n--- [{op.upper()} {spec.get('name', i + 1)}] ---")
        if op == 'sddmm':
            case = build_sddmm_case(spec)
            desc = (f"m={case['m']} n={case['n']} k={case['k']} nnz={case['nnz_c']} "
                    f"dtype={case['ab_dtype']}/{case['c_dtype']}")
            fn = _sddmm_fn(case)
        elif op == 'spgemm':
            if sp is None:
                raise RuntimeError('scipy required for SpGEMM perf (beta=1)')
            case = build_spgemm_case(spec)
            desc = (f"m={case['m']} k={case['k']} n={case['n']} "
                    f"nnz_a={case['nnz_a']} nnz_b={case['nnz_b']} dtype={case['dtype_name']}")
            fn = _spgemm_fn(case)
        else:
            spec = dict(spec)
            spec['banded'] = spec['m'] >= 8000
            case = build_spsm_case(spec)
            desc = (f"m={case['m']} nrhs={case['nrhs']} nnz={len(case['a_vals'])} "
                    f"avg_deg={spec.get('avg_degree')}")
            fn = _spsm_fn(case)

        print(f"  build: {desc}")
        print('  scope: ATK _gpu_call equivalent (full cuSPARSE invoke)')
        stats = cuda_event_benchmark(fn, warmup=warmup, repeat=repeat)
        print(f"  GPU time (ms): min={stats['min_ms']:.3f}  median={stats['median_ms']:.3f}  "
              f"avg={stats['avg_ms']:.3f}  max={stats['max_ms']:.3f}  (repeat={int(stats['repeat'])})")
        results.append({'op': op, 'idx': i + 1, 'name': spec.get('name', ''), 'desc': desc, **stats})
    return results


def print_markdown(all_results: list):
    print('\n## GPU reference times (paste into task doc)\n')
    print('| 算子 | 编号 | GPU min (ms) | GPU median (ms) | GPU avg (ms) | 备注 |')
    print('|------|------|-------------:|----------------:|-------------:|------|')
    for r in all_results:
        print(f"| {r['op'].upper()} | {r['idx']:02d} | {r['min_ms']:.3f} | {r['median_ms']:.3f} | "
              f"{r['avg_ms']:.3f} | {r['desc']} |")


def main():
    parser = argparse.ArgumentParser(description='cuSPARSE fixed-case GPU performance benchmark')
    parser.add_argument('--op', choices=['sddmm', 'spgemm', 'spsm', 'all'], default='all')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--markdown', action='store_true', help='print markdown table at end')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available')
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"libcusparse: {resolve_cusparse_lib()}")
    print(f"warmup={args.warmup} repeat={args.repeat}")

    ops = ['sddmm', 'spgemm', 'spsm'] if args.op == 'all' else [args.op]
    all_results = []
    for op in ops:
        all_results.extend(benchmark_op(op, args.warmup, args.repeat))

    if args.markdown:
        print_markdown(all_results)
    else:
        print('\nTip: re-run with --markdown to get a table for the task doc.')

    print('\nPerf benchmark finished.')


if __name__ == '__main__':
    main()
