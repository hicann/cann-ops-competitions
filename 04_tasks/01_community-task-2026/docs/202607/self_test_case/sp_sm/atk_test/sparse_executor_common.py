"""Shared helpers for aclSparse SDDMM / SpGEMM / SpSM ATK executors."""
import ctypes
import glob
import hashlib
import os
import random

import numpy as np
import torch


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; GPU path must not fall back to CPU")


def seed_from_dims(*parts):
    seed_str = "_".join(str(p) for p in parts)
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    return seed, random.Random(seed), np.random.RandomState(seed)


def resolve_sparse_lib():
    lib_sparse_path = os.environ.get("OPS_SPARSE_LIB_PATH", "")
    if lib_sparse_path and os.path.exists(lib_sparse_path):
        return lib_sparse_path
    for pattern in (
        "/home/z00889627/ops-sparse/build/libops_sparse.so",
        "/home/z00889627/ops-sparse/build_out/lib64/libops_sparse.so",
        "**/libops_sparse.so",
    ):
        candidates = glob.glob(pattern, recursive=True)
        if candidates:
            return candidates[0]
    return ""


def torch_dtype_from_name(name):
    return {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'int32': torch.int32,
        'int8': torch.int8,
    }[name]


def infer_value_dtype(tensor):
    if tensor.dtype == torch.int8:
        return 'int8'
    if tensor.dtype == torch.float16:
        return 'fp16'
    if tensor.dtype == torch.bfloat16:
        return 'bf16'
    return 'fp32'


def nnz_from_sparsity(rows, cols, sparsity):
    if rows <= 0 or cols <= 0:
        return 0
    return int(rows * cols * (1.0 - float(sparsity)))


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
                            near_singular=False):
    """Generate a nonsingular triangular CSR matrix [m,m]. fill_mode: 0=lower, 1=upper."""
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

    nnz = len(val_list)
    col_ind = np.array(col_list, dtype=np.int32)
    if dtype == 'fp16':
        vals = np.array(val_list, dtype=np.float16)
    else:
        vals = np.array(val_list, dtype=np.float32)
    return row_off, col_ind, vals


def logical_dense_from_tensor(tensor, op, rows, cols):
    mat = tensor.reshape(-1)
    mat = mat.reshape(tensor.shape[0], tensor.shape[1])
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


def to_result_tensor(arr, out_dtype, high_precision=False):
    if high_precision and out_dtype == torch.float32:
        return torch.from_numpy(np.asarray(arr)).to(torch.float64)
    return torch.from_numpy(np.asarray(arr)).to(out_dtype)


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
