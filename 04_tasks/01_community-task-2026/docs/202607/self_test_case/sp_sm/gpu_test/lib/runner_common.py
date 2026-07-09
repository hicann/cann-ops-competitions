"""Shared CLI helpers for gpu_test run scripts."""
from __future__ import annotations

import json
import os
import time
from typing import Callable, List, Tuple


def load_case_specs(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'case file must be a JSON list: {path}')
    return data


def default_cases_path(op: str, root: str) -> str:
    return os.path.join(root, 'cases', f'{op}_cases.json')


def slice_cases(cases: list, start: int, end: int) -> list:
    if end < 0:
        return cases[start:]
    return cases[start:end]


def run_case_suite(
    specs: list,
    build_fn: Callable,
    run_one_fn: Callable,
    *,
    skip_perf_only: bool = False,
    skip_scipy_required: bool = False,
    needs_scipy_fn: Callable | None = None,
) -> Tuple[int, int, int]:
    passed = failed = skipped = 0
    t0 = time.time()
    for idx, spec in enumerate(specs):
        name = spec.get('name', f'case_{idx}')
        if skip_perf_only and spec.get('perf_only'):
            print(f'\n=== [{idx+1}/{len(specs)}] {name} SKIP (perf-only) ===')
            skipped += 1
            continue
        if skip_scipy_required and needs_scipy_fn and needs_scipy_fn(spec):
            print(f'\n=== [{idx+1}/{len(specs)}] {name} SKIP (needs scipy) ===')
            skipped += 1
            continue
        try:
            case = build_fn(spec)
            run_one_fn(case, name, spec=spec)
            passed += 1
        except Exception as exc:
            failed += 1
            print(f'\n=== [{idx+1}/{len(specs)}] {name} FAIL: {exc} ===')
            raise
    elapsed = time.time() - t0
    print(f'\nSummary: passed={passed} failed={failed} skipped={skipped} elapsed={elapsed:.1f}s')
    return passed, failed, skipped
