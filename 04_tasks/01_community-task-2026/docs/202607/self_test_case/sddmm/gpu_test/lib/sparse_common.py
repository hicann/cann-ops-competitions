"""Shared helpers for standalone GPU cuSPARSE tests (extracted from ATK)."""
import hashlib
import random

import numpy as np
import torch


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Need PyTorch with GPU and a visible CUDA device."
        )


def seed_from_dims(*parts):
    seed_str = "_".join(str(p) for p in parts)
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    return seed, random.Random(seed), np.random.RandomState(seed)


def nnz_from_sparsity(rows, cols, sparsity):
    if rows <= 0 or cols <= 0:
        return 0
    return int(rows * cols * (1.0 - float(sparsity)))


def _vals_np_dtype(dtype):
    if dtype == 'fp16':
        return np.float16
    if dtype == 'int8':
        return np.int8
    return np.float32


def _sample_vals(nnz, dtype, np_rng):
    if dtype == 'int8':
        return np_rng.randint(-5, 6, size=nnz).astype(np.int8)
    if dtype == 'fp16':
        return np_rng.uniform(-5, 5, size=nnz).astype(np.float32).astype(np.float16)
    return np_rng.uniform(-5, 5, size=nnz).astype(np.float32)


def generate_csr_data(m, k, nnz, dtype='fp32', rng=None, np_rng=None, allow_zero_nnz=True):
    if rng is None:
        rng = random.Random()
    if np_rng is None:
        np_rng = np.random.RandomState()

    if m == 0 or k == 0 or nnz <= 0:
        return np.zeros(m + 1, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=_vals_np_dtype(dtype))

    row_off = np.zeros(m + 1, dtype=np.int32)
    row_nnz = np.zeros(m, dtype=np.int64)
    base = nnz // m
    r = nnz % m
    row_nnz[:] = base
    if r > 0:
        row_nnz[:r] += 1
    row_off[1:] = np.cumsum(row_nnz, dtype=np.int64).astype(np.int32)

    col_ind = np.zeros(nnz, dtype=np.int32)
    vals = _sample_vals(nnz, dtype, np_rng)

    counts = np.minimum(row_nnz, k).astype(np.int64)
    total = int(counts.sum())
    row_ids = np.repeat(np.arange(m, dtype=np.int64), counts)
    block_start = np.repeat(np.cumsum(counts) - counts, counts)
    within = np.arange(total, dtype=np.int64) - block_start
    dest = np.repeat(row_off[:m].astype(np.int64), counts) + within
    cols = np.empty(total, dtype=np.int32)

    full_mask = (counts == k)
    slot_full = np.repeat(full_mask, counts)
    if full_mask.any():
        cols[slot_full] = np.tile(np.arange(k, dtype=np.int32), int(full_mask.sum()))

    part_idx = np.nonzero(~slot_full)[0]
    if part_idx.size > 0:
        p_rows = row_ids[part_idx]
        rowbase = p_rows * np.int64(k)
        p_cols = np_rng.randint(0, k, size=part_idx.size).astype(np.int32)
        for _ in range(64):
            key = rowbase + p_cols
            order = np.argsort(key, kind='stable')
            sk = key[order]
            dup_sorted = np.zeros(sk.size, dtype=bool)
            dup_sorted[1:] = (sk[1:] == sk[:-1])
            dup = np.zeros(sk.size, dtype=bool)
            dup[order] = dup_sorted
            if int(dup.sum()) == 0:
                break
            p_cols[dup] = np_rng.randint(0, k, size=int(dup.sum())).astype(np.int32)
        else:
            for br in np.unique(p_rows[dup]).tolist():
                mask = (p_rows == br)
                c = int(mask.sum())
                p_cols[mask] = np.array(rng.sample(range(k), c), dtype=np.int32)
        cols[part_idx] = p_cols

    order = np.argsort(row_ids * np.int64(k) + cols, kind='stable')
    col_ind[dest] = cols[order]
    return row_off, col_ind, vals


def shuffle_col_indices_per_row(row_off, col_ind, rng=None):
    if rng is None:
        rng = random.Random()
    col_ind = col_ind.copy()
    m = len(row_off) - 1
    for i in range(m):
        s, e = row_off[i], row_off[i + 1]
        if e > s:
            seg = col_ind[s:e].copy()
            rng.shuffle(seg)
            col_ind[s:e] = seg
    return col_ind


def generate_triangular_csr(m, avg_degree, fill_mode, diag_type, dtype='fp32', rng=None, np_rng=None,
                            near_singular=False, banded=None):
    """Generate a nonsingular triangular CSR matrix [m,m].

    For large m with moderate avg_degree, uses a banded pattern (same nnz order,
    much faster than row-wise random sampling). Set banded=True/False to force.
    """
    if banded is None:
        banded = (m >= 8000 and not near_singular and avg_degree <= 64)
    if banded:
        return _generate_triangular_csr_banded(
            m, avg_degree, fill_mode, diag_type, dtype=dtype, np_rng=np_rng)
    return _generate_triangular_csr_random(
        m, avg_degree, fill_mode, diag_type, dtype=dtype, rng=rng, np_rng=np_rng,
        near_singular=near_singular)


def _generate_triangular_csr_banded(m, avg_degree, fill_mode, diag_type, dtype='fp32', np_rng=None):
    """Fast banded triangular CSR: O(m * bw) for large perf cases (S2/S3)."""
    if np_rng is None:
        np_rng = np.random.RandomState()
    bw = max(1, int(avg_degree))
    row_off = np.zeros(m + 1, dtype=np.int32)
    col_parts = []
    val_parts = []
    for i in range(m):
        if fill_mode == 0:
            lo = max(0, i - bw + 1)
            cols = np.arange(lo, i + 1, dtype=np.int32)
        else:
            hi = min(m, i + bw)
            cols = np.arange(i, hi, dtype=np.int32)
        n = len(cols)
        vals = np_rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
        diag_pos = np.where(cols == i)[0]
        if diag_type == 1:
            vals[diag_pos] = 1.0
        else:
            vals[diag_pos] = np_rng.uniform(2.0, 5.0, size=len(diag_pos)).astype(np.float32)
        col_parts.append(cols)
        val_parts.append(vals)
        row_off[i + 1] = row_off[i] + n
    col_ind = np.concatenate(col_parts) if col_parts else np.zeros(0, dtype=np.int32)
    vals = np.concatenate(val_parts) if val_parts else np.zeros(0, dtype=np.float32)
    if dtype == 'fp16':
        vals = vals.astype(np.float16)
    return row_off, col_ind, vals


def _generate_triangular_csr_random(m, avg_degree, fill_mode, diag_type, dtype='fp32', rng=None,
                                    np_rng=None, near_singular=False):
    if rng is None:
        rng = random.Random()
    if np_rng is None:
        np_rng = np.random.RandomState()

    row_off = np.zeros(m + 1, dtype=np.int32)
    col_list = []
    val_list = []

    for i in range(m):
        if fill_mode == 0:
            candidates = list(range(0, i + 1))
        else:
            candidates = list(range(i, m))
        target = min(max(1, int(avg_degree)), len(candidates))
        cols = sorted(rng.sample(candidates, target))
        if fill_mode == 0 and i not in cols:
            cols.append(i)
            cols = sorted(set(cols))
        if fill_mode == 1 and i not in cols:
            cols.append(i)
            cols = sorted(set(cols))

        for j in cols:
            if i == j:
                if diag_type == 1:
                    val = 1.0
                else:
                    val = float(np_rng.uniform(2.0, 5.0))
                    if near_singular:
                        val = float(np_rng.uniform(1e-4, 1e-2))
            else:
                val = float(np_rng.uniform(-2.0, 2.0))
            col_list.append(j)
            if dtype == 'fp16':
                val = np.float16(val).item()
            val_list.append(val)
        row_off[i + 1] = len(col_list)

    col_ind = np.array(col_list, dtype=np.int32)
    if dtype == 'fp16':
        vals = np.array(val_list, dtype=np.float16)
    else:
        vals = np.array(val_list, dtype=np.float32)
    return row_off, col_ind, vals


def logical_dense_from_tensor(tensor, op, rows, cols):
    mat = tensor.reshape(tensor.shape[0], tensor.shape[1])
    if op == 0:
        return mat[:rows, :cols]
    return mat[:cols, :rows].T


def csr_to_dense(row_off, col_ind, vals, shape, acc_dtype=np.float64):
    dense = np.zeros(shape, dtype=acc_dtype)
    for i in range(shape[0]):
        for p in range(row_off[i], row_off[i + 1]):
            dense[i, col_ind[p]] = acc_dtype(vals[p])
    return dense


def apply_sddmm_mask(full, row_off, col_ind):
    out = np.zeros_like(full)
    for i in range(len(row_off) - 1):
        for p in range(row_off[i], row_off[i + 1]):
            out[i, col_ind[p]] = full[i, col_ind[p]]
    return out


def sddmm_masked_golden(A, B, m, n, c_row, c_col, alpha, beta=0.0, dense_c=None,
                        ab_dtype='fp32', c_dtype='fp32', high_precision=False):
    """CPU golden at CSR nnz positions only — O(nnz*k) memory, exact for SDDMM.

    Use for large m×n (e.g. S2 10⁵×10⁵) where full dense matmul would need O(m×n) RAM.
    """
    import numpy as np
    out = np.zeros((m, n), dtype=np.float64 if high_precision else np.float32)
    use_fp64 = high_precision and ab_dtype == 'fp32' and c_dtype == 'fp32'

    if use_fp64:
        Aacc = A.to(torch.float64)
        Bacc = B.to(torch.float64)
    elif ab_dtype == 'fp16':
        Aacc = A.to(torch.float16).to(torch.float32)
        Bacc = B.to(torch.float16).to(torch.float32)
    else:
        Aacc = A.to(torch.float32)
        Bacc = B.to(torch.float32)

    acc_np = np.float64 if use_fp64 else np.float32
    for i in range(m):
        s, e = int(c_row[i]), int(c_row[i + 1])
        if s >= e:
            continue
        cols = c_col[s:e].astype(np.int64, copy=False)
        dots = (Aacc[i] @ Bacc[:, cols]).cpu().numpy().astype(acc_np, copy=False)
        vals = alpha * dots
        if beta != 0.0 and dense_c is not None:
            c_slice = dense_c[i, cols]
            if use_fp64:
                vals = vals + beta * c_slice.to(torch.float64).cpu().numpy()
            else:
                c32 = c_slice.to(torch.float16 if c_dtype == 'fp16' else torch.float32)
                if c_dtype == 'fp16':
                    c32 = c32.to(torch.float16).to(torch.float32)
                vals = vals + beta * c32.cpu().numpy().astype(acc_np, copy=False)
        out[i, cols] = vals

    if c_dtype == 'fp16' and not use_fp64:
        out = out.astype(np.float32).astype(np.float16).astype(np.float32)
    return out


def scatter_csr_to_dense(m, n, row_off, col_ind, vals, out_dtype):
    dense = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for p in range(row_off[i], row_off[i + 1]):
            dense[i, col_ind[p]] = vals[p]
    return torch.from_numpy(dense).to(out_dtype)


def to_result_tensor(arr, out_dtype, high_precision=False):
    """Match ATK executor: fp32 output uses fp64 tensor as high-precision golden."""
    if high_precision and out_dtype == torch.float32:
        return torch.from_numpy(np.asarray(arr)).to(torch.float64)
    return torch.from_numpy(np.asarray(arr)).to(out_dtype)


def max_abs_diff(a, b):
    return float((a.float() - b.float()).abs().max().item())
