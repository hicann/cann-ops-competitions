"""Task-book fixed reference cases (3 per op) for cuSPARSE GPU benchmarking.

Same specs as case_generator.*_fixed_cases() — paste results into task doc
「固定参考用例 → GPU 参考耗时」.
"""
from __future__ import annotations

from typing import Any, Dict, List

from case_generator import sddmm_fixed_cases, spgemm_fixed_cases, spsm_fixed_cases


def all_perf_cases(op: str) -> List[Dict[str, Any]]:
    if op == 'sddmm':
        return sddmm_fixed_cases()[:3]
    if op == 'spgemm':
        return spgemm_fixed_cases()[:3]
    if op == 'spsm':
        return spsm_fixed_cases()[:3]
    raise ValueError(f'unknown op: {op}')
