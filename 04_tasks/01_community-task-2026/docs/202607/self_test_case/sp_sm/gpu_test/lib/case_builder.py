"""Build runnable gpu_test cases from JSON specs."""
from __future__ import annotations

import numpy as np
import torch

from sparse_common import (
    generate_csr_data,
    generate_triangular_csr,
    nnz_from_sparsity,
    seed_from_dims,
)


def _pad_dense(base: torch.Tensor, ld: int, rows: int, cols: int) -> torch.Tensor:
    if ld <= cols:
        return base[:rows, :cols].contiguous()
    out = torch.zeros(rows, ld, dtype=base.dtype, device=base.device)
    out[:, :cols] = base[:rows, :cols]
    return out


def build_sddmm_case(spec: dict) -> dict:
    m = int(spec['m'])
    n = int(spec['n'])
    k = int(spec['k'])
    sparsity = float(spec['sparsity'])
    alpha = float(spec['alpha'])
    beta = float(spec['beta'])
    op_a = int(spec.get('op_a', 0))
    op_b = int(spec.get('op_b', 0))
    ab_dtype = spec.get('ab_dtype', 'fp32')
    c_dtype = spec.get('c_dtype', 'fp32')
    seed = int(spec.get('seed', 0))
    ld_a_pad = int(spec.get('ld_a_pad', 0))
    ld_b_pad = int(spec.get('ld_b_pad', 0))

    torch.manual_seed(seed)
    ab_dt = {'fp32': torch.float32, 'fp16': torch.float16}[ab_dtype]

    if op_a == 0:
        dense_a = torch.randn(m, k, dtype=ab_dt)
        a_rows, a_cols = m, k
    else:
        dense_a = torch.randn(k, m, dtype=ab_dt)
        a_rows, a_cols = k, m
    if op_b == 0:
        dense_b = torch.randn(k, n, dtype=ab_dt)
        b_rows, b_cols = k, n
    else:
        dense_b = torch.randn(n, k, dtype=ab_dt)
        b_rows, b_cols = n, k

    nnz_c = nnz_from_sparsity(m, n, sparsity)
    _, rng, np_rng = seed_from_dims('SDDMM', m, n, max(nnz_c, 1), seed)
    c_row, c_col, _ = generate_csr_data(m, n, max(nnz_c, 0), c_dtype, rng=rng, np_rng=np_rng)

    if beta == 0.0 or nnz_c == 0:
        dense_c = None
        c_vals = np.zeros(max(nnz_c, 0), dtype=np.float32)
    else:
        c_dt = {'fp32': torch.float32, 'fp16': torch.float16}[c_dtype]
        dense_c = torch.randn(m, n, dtype=c_dt)
        c_vals = np.zeros(max(nnz_c, 0), dtype=np.float32)
        for i in range(m):
            for p in range(c_row[i], c_row[i + 1]):
                c_vals[p] = float(dense_c[i, c_col[p]].item())

    ld_a = (k if op_a == 0 else m) + ld_a_pad
    ld_b = (n if op_b == 0 else k) + ld_b_pad
    if ld_a_pad > 0:
        dense_a = _pad_dense(dense_a, ld_a, a_rows, a_cols)
    if ld_b_pad > 0:
        dense_b = _pad_dense(dense_b, ld_b, b_rows, b_cols)

    return dict(
        m=m, n=n, k=k, alpha=alpha, beta=beta, op_a=op_a, op_b=op_b,
        ld_a=ld_a, ld_b=ld_b, order_a=0, order_b=0,
        dense_a=dense_a, dense_b=dense_b, dense_c=dense_c,
        c_row=c_row, c_col=c_col, c_vals=c_vals,
        ab_dtype=ab_dtype, c_dtype=c_dtype, nnz_c=nnz_c,
        perf_only=bool(spec.get('perf_only', False)),
        name=spec.get('name', 'case'),
        category=spec.get('category', ''),
    )


def build_spgemm_case(spec: dict) -> dict:
    m = int(spec['m'])
    k = int(spec['k'])
    n = int(spec['n'])
    sp_a = float(spec['sp_a'])
    sp_b = float(spec['sp_b'])
    alpha = float(spec['alpha'])
    beta = float(spec['beta'])
    dtype_name = spec.get('dtype_name', 'fp32')
    seed = int(spec.get('seed', 0))

    nnz_a = max(nnz_from_sparsity(m, k, sp_a), 0)
    nnz_b = max(nnz_from_sparsity(k, n, sp_b), 0)
    nnz_c = max(nnz_from_sparsity(m, n, 0.999), 0)

    _, rng_a, np_rng_a = seed_from_dims('A', m, k, max(nnz_a, 1), seed)
    _, rng_b, np_rng_b = seed_from_dims('B', k, n, max(nnz_b, 1), seed + 1)
    _, rng_c, np_rng_c = seed_from_dims('C', m, n, max(nnz_c, 1), seed + 2)

    a_row, a_col, a_vals = generate_csr_data(m, k, nnz_a, dtype_name, rng=rng_a, np_rng=np_rng_a)
    b_row, b_col, b_vals = generate_csr_data(k, n, nnz_b, dtype_name, rng=rng_b, np_rng=np_rng_b)
    c_row, c_col, c_vals = generate_csr_data(m, n, nnz_c, dtype_name, rng=rng_c, np_rng=np_rng_c)

    return dict(
        m=m, k=k, n=n, alpha=alpha, beta=beta, dtype_name=dtype_name,
        nnz_a=nnz_a, nnz_b=nnz_b,
        a_row=a_row, a_col=a_col, a_vals=a_vals,
        b_row=b_row, b_col=b_col, b_vals=b_vals,
        c_row=c_row, c_col=c_col, c_vals=c_vals,
        perf_only=bool(spec.get('perf_only', False)),
        name=spec.get('name', 'case'),
        category=spec.get('category', ''),
    )


def build_spsm_case(spec: dict) -> dict:
    m = int(spec['m'])
    nrhs = int(spec['nrhs'])
    alpha = float(spec['alpha'])
    avg_degree = float(spec.get('avg_degree', 10.0))
    op_a = int(spec.get('op_a', 0))
    op_b = int(spec.get('op_b', 0))
    fill_mode = int(spec.get('fill_mode', 0))
    diag_type = int(spec.get('diag_type', 0))
    in_place = bool(spec.get('in_place', False))
    null_values = bool(spec.get('null_values', False))
    update_matrix = bool(spec.get('update_matrix', False))
    near_singular = bool(spec.get('near_singular', False))
    seed = int(spec.get('seed', 0))
    # Functional TC keeps random pattern; large fixed/random perf cases use banded fast path.
    use_banded = bool(spec.get('banded', m >= 8000 and not near_singular))

    torch.manual_seed(seed)
    dense_b = torch.randn(m, nrhs, dtype=torch.float32)
    dense_c = torch.randn(m, nrhs, dtype=torch.float32)

    _, rng, np_rng = seed_from_dims('SpSM', m, int(avg_degree * 100), nrhs, seed)
    a_row, a_col, a_vals = generate_triangular_csr(
        m, avg_degree=avg_degree, fill_mode=fill_mode, diag_type=diag_type,
        dtype='fp32', rng=rng, np_rng=np_rng, near_singular=near_singular, banded=use_banded)

    return dict(
        m=m, nrhs=nrhs, alpha=alpha, op_a=op_a, op_b=op_b,
        fill_mode=fill_mode, diag_type=diag_type,
        in_place=in_place, null_values=null_values, update_matrix=update_matrix,
        dense_b=dense_b, dense_c=dense_c,
        a_row=a_row, a_col=a_col, a_vals=a_vals,
        perf_only=bool(spec.get('perf_only', False)),
        name=spec.get('name', 'case'),
        category=spec.get('category', ''),
    )
