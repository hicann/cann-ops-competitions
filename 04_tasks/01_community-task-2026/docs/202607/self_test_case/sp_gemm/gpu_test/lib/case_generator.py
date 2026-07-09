"""Generate gpu_test case specs (200 per op) aligned with task-book coverage.

Each spec is a JSON-serializable dict consumed by case_builder.build_*_case().
First 3 entries are small fixed anchor cases (core semantics, not perf scale);
then functional TC + random cases fill to 200 total.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

MEM_LIMIT = 20 * 1024 * 1024 * 1024

SDDMM_K_CHOICES = [16, 64, 128, 256, 512]
SPGEMM_DTYPE_CYCLE = ['fp32', 'fp16', 'bf16']
SDDMM_DTYPE_CYCLE = [
    {'ab_dtype': 'fp32', 'c_dtype': 'fp32'},
    {'ab_dtype': 'fp16', 'c_dtype': 'fp32'},
    {'ab_dtype': 'fp16', 'c_dtype': 'fp16'},
]
SPGEMM_NRHS = [1, 8, 32, 128]


def _spec(name: str, category: str, seed: int, perf_only: bool = False, **params) -> Dict[str, Any]:
    return {'name': name, 'category': category, 'seed': seed, 'perf_only': perf_only, **params}


def _check_sddmm_mem(m, k, n, elem_bytes=4):
    items = m * k + k * n + max(m * n // 100, 1)
    mem = items * elem_bytes
    if mem > MEM_LIMIT:
        scale = (MEM_LIMIT / max(mem, 1)) ** 0.5
        m = max(int(m * scale), 256)
        n = max(int(n * scale), 256)
        k = max(int(k * scale), 16)
    return m, k, n


def _check_spgemm_mem(m, k, n, sp_a, sp_b, elem_bytes=4):
    nnz_a = max(1, int(m * k * (1 - sp_a)))
    nnz_b = max(1, int(k * n * (1 - sp_b)))
    items = nnz_a + nnz_b + m * n // 10
    mem = items * elem_bytes
    if mem > MEM_LIMIT:
        m = max(int(m * 0.7), 128)
        k = max(int(k * 0.7), 128)
        n = max(int(n * 0.7), 128)
    return m, k, n, sp_a, sp_b


def _check_spsm_mem(m, nrhs, avg_degree):
    nnz = max(1, int(m * avg_degree))
    mem = (nnz * 2 + m * nrhs * 2) * 4
    if mem > MEM_LIMIT:
        m = max(int(m * 0.7), 1)
    return m, nrhs, avg_degree


def sddmm_fixed_cases() -> List[Dict[str, Any]]:
    """Three small fixed anchors: fp32 core, β accumulation, fp16 mixed (M1)."""
    cases = [
        _spec('FIX01_fp32_beta0', 'fixed', 1001,
              m=256, n=256, k=64, sparsity=0.995, alpha=1.0, beta=0.0,
              op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('FIX02_fp32_beta', 'fixed', 1002,
              m=128, n=128, k=32, sparsity=0.99, alpha=2.0, beta=1.5,
              op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('FIX03_fp16_m1', 'fixed', 1003,
              m=256, n=256, k=64, sparsity=0.995, alpha=1.0, beta=0.0,
              op_a=0, op_b=0, ab_dtype='fp16', c_dtype='fp32'),
        _spec('TC03_outer_m1', 'functional', 1103, m=1, n=4096, k=64, sparsity=0.95,
              alpha=1.0, beta=0.0, op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('TC03_outer_n1', 'functional', 1104, m=4096, n=1, k=128, sparsity=0.95,
              alpha=1.0, beta=0.0, op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('TC04_gnn', 'functional', 1105, m=512, n=512, k=1, sparsity=0.98,
              alpha=1.0, beta=0.0, op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('TC05_nonsquare', 'functional', 1106, m=512, n=256, k=128, sparsity=0.992,
              alpha=1.0, beta=0.0, op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('TC06_zero_nnz', 'functional', 1107, m=128, n=128, k=64, sparsity=1.0,
              alpha=1.0, beta=0.0, op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('M2_fp16', 'functional', 1109, m=256, n=256, k=64, sparsity=0.995,
              alpha=1.0, beta=0.0, op_a=0, op_b=0, ab_dtype='fp16', c_dtype='fp16'),
        _spec('TC_transpose', 'functional', 1110, m=128, n=64, k=128, sparsity=0.99,
              alpha=1.0, beta=0.0, op_a=1, op_b=0, ab_dtype='fp32', c_dtype='fp32'),
        _spec('TC_noncontig', 'functional', 1111, m=256, n=256, k=64, sparsity=0.995,
              alpha=1.0, beta=0.0, op_a=0, op_b=0, ab_dtype='fp32', c_dtype='fp32',
              ld_a_pad=32, ld_b_pad=16),
    ]
    return cases


def spgemm_fixed_cases() -> List[Dict[str, Any]]:
    """Three small fixed anchors: fp32 SpGEMM, zero nnz, non-trivial α/β."""
    cases = [
        _spec('FIX01_fp32', 'fixed', 2001,
              m=128, k=128, n=128, sp_a=0.95, sp_b=0.95, alpha=1.0, beta=1.0, dtype_name='fp32'),
        _spec('FIX02_zero_nnz', 'fixed', 2002,
              m=128, k=128, n=128, sp_a=1.0, sp_b=1.0, alpha=1.0, beta=0.0, dtype_name='fp32'),
        _spec('FIX03_alpha_beta', 'fixed', 2003,
              m=128, k=64, n=128, sp_a=0.98, sp_b=0.98, alpha=2.0, beta=0.5, dtype_name='fp32'),
        _spec('TC02_zero_nnz_b1', 'functional', 2103, m=128, k=128, n=128, sp_a=1.0, sp_b=0.9999,
              alpha=1.0, beta=1.0, dtype_name='fp32'),
        _spec('TC_fp16', 'functional', 2105, m=128, k=128, n=128, sp_a=0.99, sp_b=0.99,
              alpha=1.0, beta=1.0, dtype_name='fp16'),
        _spec('TC_bf16', 'functional', 2106, m=128, k=128, n=128, sp_a=0.99, sp_b=0.99,
              alpha=1.0, beta=1.0, dtype_name='bf16'),
        _spec('TC_nnz_inflate', 'functional', 2107, m=256, k=256, n=256, sp_a=0.88, sp_b=0.88,
              alpha=1.0, beta=1.0, dtype_name='fp32'),
        _spec('TC_high_sparsity', 'functional', 2108, m=512, k=512, n=512, sp_a=0.999, sp_b=0.999,
              alpha=1.0, beta=1.0, dtype_name='fp32'),
    ]
    return cases


def spsm_fixed_cases() -> List[Dict[str, Any]]:
    """Three small fixed anchors: basic solve, updateMatrix, nullValues."""
    cases = [
        _spec('FIX01_lower_nrhs8', 'fixed', 3001,
              m=256, nrhs=8, alpha=1.0, avg_degree=10.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=0),
        _spec('FIX02_update_matrix', 'fixed', 3002,
              m=128, nrhs=4, alpha=1.0, avg_degree=10.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=0, update_matrix=True),
        _spec('FIX03_null_values', 'fixed', 3003,
              m=128, nrhs=4, alpha=1.0, avg_degree=10.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=0, null_values=True),
        _spec('TC02_upper_trans', 'functional', 3102, m=128, nrhs=4, alpha=1.0, avg_degree=10.0,
              op_a=1, op_b=0, fill_mode=1, diag_type=0),
        _spec('TC03_unit_diag', 'functional', 3103, m=128, nrhs=1, alpha=-2.0, avg_degree=10.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=1),
        _spec('TC05_in_place', 'functional', 3105, m=128, nrhs=8, alpha=1.0, avg_degree=10.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=0, in_place=True),
        _spec('TC06_null_update', 'functional', 3107, m=128, nrhs=4, alpha=1.0, avg_degree=10.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=0, null_values=True, update_matrix=True),
        _spec('TC_boundary_m1', 'functional', 3108, m=1, nrhs=1, alpha=1.0, avg_degree=5.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=0),
        _spec('TC_near_singular', 'functional', 3109, m=512, nrhs=8, alpha=1.0, avg_degree=12.0,
              op_a=0, op_b=0, fill_mode=0, diag_type=0, near_singular=True),
    ]
    return cases


def _sddmm_random(i: int, rng: random.Random) -> Dict[str, Any]:
    dc = SDDMM_DTYPE_CYCLE[i % len(SDDMM_DTYPE_CYCLE)]
    elem = 2 if dc['ab_dtype'] == 'fp16' else 4
    m = rng.randint(1000, 50000)
    n = rng.randint(1000, 50000)
    k = rng.choice(SDDMM_K_CHOICES)
    sparsity = round(rng.uniform(0.999, 0.99999), 5)
    beta = 0.0 if rng.random() < 0.55 else round(rng.uniform(-3, 3), 4)
    op_a = rng.choice([0, 1])
    op_b = rng.choice([0, 1])
    ld_a_pad = ld_b_pad = 0
    if i % 23 == 0:
        ld_a_pad = rng.randint(1, 64)
        ld_b_pad = rng.randint(1, 64)
    if i % 17 == 0:
        m, n, k = 1, rng.randint(500, 8000), rng.choice(SDDMM_K_CHOICES)
    elif i % 19 == 0:
        e = rng.randint(2000, 30000)
        m = n = e
        k = 1
        sparsity = round(rng.uniform(0.96, 0.999), 4)
    elif i % 29 == 0:
        m = rng.randint(8000, 40000)
        n = rng.randint(32, 512)
        sparsity = round(rng.uniform(0.99, 0.9999), 4)
    elif i % 31 == 0:
        sparsity = 1.0
        beta = 0.0
    m, k, n = _check_sddmm_mem(m, k, n, elem)
    if beta != 0.0:
        m = min(m, 2048)
        n = min(n, 2048)
    return _spec(
        f'R{i:03d}', 'random', 420000 + i,
        m=m, n=n, k=k, sparsity=sparsity, alpha=round(rng.uniform(-2, 2), 4) if rng.random() < 0.3 else 1.0,
        beta=beta, op_a=op_a, op_b=op_b, ld_a_pad=ld_a_pad, ld_b_pad=ld_b_pad,
        **dc,
    )


def _spgemm_random(i: int, rng: random.Random) -> Dict[str, Any]:
    dtype_name = SPGEMM_DTYPE_CYCLE[i % len(SPGEMM_DTYPE_CYCLE)]
    elem = 2 if dtype_name != 'fp32' else 4
    m = rng.randint(128, 5000)
    k = rng.randint(128, 5000)
    n = rng.randint(128, 5000)
    sp_a = round(rng.uniform(0.99, 0.9999), 4)
    sp_b = round(rng.uniform(0.99, 0.9999), 4)
    alpha = 1.0
    beta = 1.0
    if i % 7 == 0:
        alpha = round(rng.choice([-2.0, -0.5, 0.5, 2.0, 3.0]), 4)
        beta = round(rng.choice([-1.0, 0.0, 0.5, 2.0]), 4)
    if i % 13 == 0:
        sp_a = sp_b = 1.0
        beta = 0.0
    elif i % 11 == 0:
        sp_a = round(rng.uniform(0.85, 0.92), 4)
        sp_b = round(rng.uniform(0.85, 0.92), 4)
    elif i % 37 == 0:
        m = k = n = rng.randint(4000, 8000)
        sp_a = sp_b = round(rng.uniform(0.9995, 0.99999), 5)
    m, k, n, sp_a, sp_b = _check_spgemm_mem(m, k, n, sp_a, sp_b, elem)
    return _spec(
        f'R{i:03d}', 'random', 430000 + i,
        m=m, k=k, n=n, sp_a=sp_a, sp_b=sp_b, alpha=alpha, beta=beta, dtype_name=dtype_name,
    )


def _spsm_random(i: int, rng: random.Random) -> Dict[str, Any]:
    m = rng.randint(100, 20000)
    nrhs = rng.choice(SPGEMM_NRHS)
    avg_degree = round(rng.uniform(5, 50), 2)
    fill_mode = rng.choice([0, 1])
    op_a = rng.choice([0, 1])
    if i % 11 == 0:
        fill_mode = 1
        op_a = 1
    diag_type = rng.choice([0, 1])
    in_place = i % 41 == 0
    null_values = i % 43 == 0
    update_matrix = i % 47 == 0
    near_singular = i % 53 == 0
    if i % 59 == 0:
        m = 1
        nrhs = 1
    m, nrhs, avg_degree = _check_spsm_mem(m, nrhs, avg_degree)
    return _spec(
        f'R{i:03d}', 'random', 440000 + i,
        m=m, nrhs=nrhs, alpha=round(rng.uniform(-2, 2), 4),
        avg_degree=avg_degree, op_a=op_a, op_b=rng.choice([0, 1]),
        fill_mode=fill_mode, diag_type=diag_type,
        in_place=in_place, null_values=null_values, update_matrix=update_matrix,
        near_singular=near_singular,
    )


def generate_cases(op: str, total: int = 200, random_seed: int = 42) -> List[Dict[str, Any]]:
    if op == 'sddmm':
        fixed = sddmm_fixed_cases()
        random_fn = _sddmm_random
    elif op == 'spgemm':
        fixed = spgemm_fixed_cases()
        random_fn = _spgemm_random
    elif op == 'spsm':
        fixed = spsm_fixed_cases()
        random_fn = _spsm_random
    else:
        raise ValueError(f'unknown op: {op}')

    n_random = max(0, total - len(fixed))
    rng = random.Random(random_seed)
    random_cases = [random_fn(i, rng) for i in range(n_random)]
    return fixed + random_cases
