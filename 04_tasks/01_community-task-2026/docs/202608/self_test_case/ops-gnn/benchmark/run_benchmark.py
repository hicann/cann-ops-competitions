#!/usr/bin/env python3
"""GNN operators benchmark — NPU latency / throughput.

Usage:
    python benchmark/run_benchmark.py                          # all ops, NPU
    python benchmark/run_benchmark.py --op fps,scatter_sum     # selected
    python benchmark/run_benchmark.py --iter 100 --warmup 20   # custom
    python benchmark/run_benchmark.py --device cpu             # CPU fallback
"""

import argparse
import os
import sys
import time

import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "python"))
import ops_gnn

WRITE = sys.stdout.write
FLUSH = sys.stdout.flush


def setup_device(device):
    if hasattr(torch, "npu"):
        torch.npu.set_device(4)
    return torch.device(device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def bold(s):
    return f"\033[1m{s}\033[0m"


@torch.no_grad()
def bench(fn, args, warmup, iters, device):
    for _ in range(warmup):
        fn(*args)
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    synchronize(device)
    return (time.perf_counter() - t0) / iters * 1000  # ms


# ── per-op bench config: [(display_name, shape_dims, kwargs), ...] ──────────

SHAPES = {
    "fps": [
        (" N=  1K F=  3 r=0.50",  1024,   3, 0.50),
        (" N=  4K F=  3 r=0.50",  4096,   3, 0.50),
        (" N= 16K F=  3 r=0.25", 16384,   3, 0.25),
        (" N= 32K F=  3 r=0.25", 32768,   3, 0.25),
    ],
    "gather_coo": [
        ("16384, 65536",   16384,   65536),
        ("65536, 65536",   65536,   65536),
        ("65536, 262144",  65536,  262144),
        ("262144, 524288", 262144, 524288),
        ("262144, 1048576", 262144, 1048576),
    ],
    "gather_csr": [
        (" 256K,  512",   262144,   512),
        (" 512K, 1024",   524288,  1024),
        ("1024K, 1024",  1048576,  1024),
        ("1024K, 2048",  1048576,  2048),
        ("2048K, 4096",  2097152,  4096),
        ("4096K, 4096",  4194304,  4096),
    ],
    "grid_cluster": [
        (" N=   8M D=  3 s=0.005",    (8388608, 3),    0.005),
        (" N=  16M D=  3 s=0.002",    (16777216, 3),   0.002),
        (" N=   4M D= 64 s=0.010",    (4194304, 64),   0.010),
        (" N=   8M D= 32 s=0.005",    (8388608, 32),   0.005),
        (" N=  32M D=  3 s=0.001",    (33554432, 3),   0.001),
    ],
    "knn": [
        (" N=  1K F=  3 k= 4",  (1024, 3),   4),
        (" N=  1K F= 16 k= 8",  (1024, 16),  8),
        (" N=  4K F= 32 k= 8",  (4096, 32),  8),
        (" N=  4K F= 64 k=16",  (4096, 64), 16),
        (" N= 16K F= 32 k=16",  (16384, 32), 16),
    ],
    "nearest": [
        (" N=  1K M=  1K F=  3",  1024,  1024,   3),
        (" N=  4K M=  4K F=  3",  4096,  4096,   3),
        (" N= 16K M= 16K F=  3", 16384, 16384,   3),
        (" N= 32K M= 32K F=  3", 32768, 32768,   3),
        (" N=  8K M=  8K F= 64",  8192,  8192,  64),
    ],
    "random_walk": [
        (" E= 64K walk=128 start= 8K nodes= 16K",  65536, 128,  8192, 16384),
        (" E=256K walk=128 start=16K nodes= 64K", 262144, 128, 16384, 65536),
        (" E=512K walk=128 start=32K nodes=128K", 524288, 128, 32768, 131072),
        (" E=512K walk=256 start=32K nodes=128K", 524288, 256, 32768, 131072),
    ],
    "scatter": [
        (" 16384 x  512", 16384, 512, 4096),
        (" 32768 x  512", 32768, 512, 8192),
        (" 65536 x  512", 65536, 512, 16384),
        (" 65536 x 1024", 65536, 1024, 16384),
        (" 65536 x 2048", 65536, 2048, 16384),
    ],
    "segment_coo": [
        (" N= 256K seg= 16K",   262144,  16384),
        (" N= 512K seg= 32K",   524288,  32768),
        (" N=1024K seg= 32K",  1048576,  32768),
        (" N=2048K seg= 64K",  2097152,  65536),
        (" N=4096K seg= 64K",  4194304,  65536),
    ],
    "segment_csr": [
        (" N= 256K seg= 16K",   262144,  16384),
        (" N= 512K seg= 32K",   524288,  32768),
        (" N=1024K seg= 32K",  1048576,  32768),
        (" N=2048K seg= 64K",  2097152,  65536),
        (" N=4096K seg= 64K",  4194304,  65536),
    ],
    "graclus_cluster": [
        (" V=   4 E=  12",      4,    12),
        (" V=   8 E=  56",      8,    56),
        (" V=  16 E= 240",     16,   240),
        (" V=  32 E= 992",     32,   992),
        (" V=  64 E=4032",     64,  4032),
    ],
}

DTYPES = {
    "random_walk": [None],
    "graclus_cluster": [None],
    "default": [torch.float32, torch.float16],
}


def dtype_str(dt):
    return str(dt).replace("torch.", "") if dt else "long"


# ── factory functions ─────────────────────────────────────────────────────────

def _fps(N, F, r, dtype, device):
    return [torch.randn(N, F, dtype=dtype, device=device),
            None, r, False]

def _gather_coo(N, nidx, dtype, device):
    src = torch.randn(N, 128, dtype=dtype, device=device)
    idx = torch.randint(0, N - 1, (nidx,), device=device).long()
    return [src, idx]

def _gather_csr(N, nseg, dtype, device):
    src = torch.randn(nseg, 128, dtype=dtype, device=device)
    step = N // nseg
    indptr = torch.arange(0, N + 1, step, device=device).long()
    if indptr.numel() < nseg + 1:
        indptr = torch.linspace(0, N, nseg + 1, device=device).long()
    elif indptr.numel() > nseg + 1:
        indptr = indptr[:nseg + 1]
    return [src, indptr]

def _grid(dims, s, dtype, device):
    N, D = dims
    return [torch.randn(N, D, dtype=dtype, device=device),
            torch.full((D,), s, device=device)]

def _knn(dims, k, dtype, device):
    N, F = dims
    x = torch.randn(N, F, dtype=dtype, device=device)
    return [x, x, k]

def _nearest(Nx, Ny, F, dtype, device):
    return [torch.randn(Nx, F, dtype=dtype, device=device),
            torch.randn(Ny, F, dtype=dtype, device=device)]

def _random_walk(E, wl, S, nodes, _, device):
    return [torch.randint(0, nodes, (E,), dtype=torch.long, device=device),
            torch.randint(0, nodes, (E,), dtype=torch.long, device=device),
            torch.arange(S, dtype=torch.long, device=device), wl]

def _scatter(N, C, seg, reduce, dtype, device):
    src = torch.randn(N, C, dtype=dtype, device=device)
    idx = torch.randint(0, seg, (N,), device=device).long()
    return [src, idx, 0, None, seg, reduce]

def _segment_coo(N, nseg, reduce, dtype, device):
    src = torch.randn(N, 128, dtype=dtype, device=device)
    seg_size = N // nseg
    idx = torch.arange(nseg, device=device).repeat_interleave(seg_size)
    if idx.size(0) < N:
        idx = torch.cat([idx, torch.full((N - idx.size(0),), nseg - 1, device=device)])
    return [src, idx, None, nseg, reduce]

def _segment_csr(N, nseg, reduce, dtype, device):
    src = torch.randn(N, 128, dtype=dtype, device=device)
    seg_size = N // nseg
    indptr = torch.arange(0, N + 1, seg_size, device=device).long()
    if indptr.size(0) - 1 < nseg:
        indptr = torch.linspace(0, N, nseg + 1, device=device).long()
    elif indptr.size(0) - 1 > nseg:
        indptr = indptr[:nseg + 1]
        indptr[-1] = N
    return [src, indptr, None, reduce]


def _graclus_cluster(V, E, _, device):
    r = torch.arange(V, device=device)
    row = r.repeat_interleave(V - 1)
    col = torch.cat([torch.cat([r[:i], r[i + 1:]]) for i in range(V)])
    w = torch.ones(V * (V - 1), device=device)
    return [row, col, w]


FACTORIES = {
    "fps":               lambda N, F, r, dt, dev: _fps(N, F, r, dt, dev),
    "gather_coo":        lambda N, nidx, dt, dev: _gather_coo(N, nidx, dt, dev),
    "gather_csr":        lambda N, ns, dt, dev: _gather_csr(N, ns, dt, dev),
    "grid_cluster":      lambda d, s, dt, dev: _grid(d, s, dt, dev),
    "knn":               lambda d, k, dt, dev: _knn(d, k, dt, dev),
    "nearest":           lambda Nx, Ny, F, dt, dev: _nearest(Nx, Ny, F, dt, dev),
    "random_walk":       lambda E, wl, S, n, dt, dev: _random_walk(E, wl, S, n, dt, dev),
    "scatter":           lambda N, C, seg, dt, dev: _scatter(N, C, seg, args.reduce, dt, dev),
    "segment_coo":       lambda N, ns, dt, dev: _segment_coo(N, ns, args.reduce, dt, dev),
    "segment_csr":       lambda N, ns, dt, dev: _segment_csr(N, ns, args.reduce, dt, dev),
    "graclus_cluster":    lambda V, E, _, dev: _graclus_cluster(V, E, _, dev),
}


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", type=str, default=None)
    p.add_argument("--iter", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", type=str, default="npu")
    p.add_argument("--reduce", type=str, default="sum",
                   choices=["sum", "mean", "min", "max"])
    args = p.parse_args()

    device = setup_device(args.device)

    targets = set(args.op.split(",")) if args.op else set(SHAPES.keys())

    configs = [(k, v) for k, v in SHAPES.items() if k in targets]
    if not configs:
        print(f"No ops selected. Available: {list(SHAPES.keys())}")
        return

    WRITE(f"device={device}  warmup={args.warmup}  iters={args.iter}\n"); FLUSH()

    fn_lookup = {k: getattr(ops_gnn, k) for k in SHAPES}

    for op_name, shape_list in configs:
        fn = fn_lookup[op_name]
        dtypes = DTYPES.get(op_name, DTYPES["default"])

        WRITE(f"\n{bold(op_name)}\n"); FLUSH()
        header = f"{'shape':>22s}"
        for dt in dtypes:
            header += f" {dtype_str(dt):>10s}"
        WRITE(header + "\n"); FLUSH()
        WRITE("-" * (24 + 12 * len(dtypes)) + "\n"); FLUSH()

        factory = FACTORIES.get(op_name)

        for entry in shape_list:
            label = entry[0]
            shape_dims = entry[1:]
            row = f"{label:>22s}"

            for dtype in dtypes:
                try:
                    if factory is not None and dtype is not None:
                        tensor_args = factory(*shape_dims, dtype, device)
                    elif factory is not None:
                        tensor_args = factory(*shape_dims, torch.float32, device)
                    else:
                        tensor_args = list(shape_dims)

                    lat = bench(fn, tensor_args, args.warmup, args.iter, device)
                    row += f" {lat:>8.3f}ms"
                except Exception as e:
                    row += f" {'ERR':>10s}"

            WRITE(row + "\n"); FLUSH()

    WRITE(f"\nDone.\n"); FLUSH()


if __name__ == "__main__":
    main()
