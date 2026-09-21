import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'operator/python'))
from max_pool_backward import MaxPoolBackward

spec = importlib.util.spec_from_file_location('golden', ROOT / 'max_pool2d_with_mask_backward_golden.py')
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

LIBRARY = os.getenv('BENCHMARK_LIBRARY', '')
if not LIBRARY or not os.path.exists(LIBRARY):
    for candidate in [
        '/tmp/maxpool-310p-npu/libmax_pool_backward.so',
        '/tmp/maxpool-a2-npu/libmax_pool_backward.so',
        str(ROOT / 'operator_build/libcust_opapi.so')
    ]:
        if os.path.exists(candidate):
            LIBRARY = candidate
            break

if not LIBRARY or not os.path.exists(LIBRARY):
    raise RuntimeError(f"Cannot find compiled kernel library. Looked in /tmp/maxpool-310p-npu and /tmp/maxpool-a2-npu")

op = MaxPoolBackward(LIBRARY)

is_310p = '310p' in LIBRARY or os.getenv('BENCHMARK_PLATFORM') == '310p'
# Default to 8 cores on 310P, 24 cores on 910B
BLOCKS = int(os.getenv('BENCHMARK_BLOCKS', '8' if is_310p else '24'))

TEST_CASES = [
    # (Category, Name, N, C, H, W, k, s, p, d, ceil)
    ("Small", "Small_16x16", (1, 64, 16, 16), [3, 3], [1, 1], [1, 1], [1], False),
    ("Small", "Small_32x32", (1, 64, 32, 32), [3, 3], [1, 1], [1, 1], [1], False),
    ("Medium", "Medium_64x64_k2s2", (2, 128, 64, 64), [2, 2], [2, 2], [0, 0], [1], False),
    ("Medium", "Medium_64x64_k3s2", (2, 128, 64, 64), [3, 3], [2, 2], [1, 1], [1], False),
    ("YOLOv11", "YOLO_SPPF_20x20", (1, 512, 20, 20), [5, 5], [1, 1], [2, 2], [1], False),
    ("YOLOv11", "YOLO_SPPF_40x40", (1, 256, 40, 40), [5, 5], [1, 1], [2, 2], [1], False),
    ("YOLOv11", "YOLO_SPPF_80x80", (1, 128, 80, 80), [5, 5], [1, 1], [2, 2], [1], False),
    ("Large", "Large_128x128_k3s2", (4, 256, 128, 128), [3, 3], [2, 2], [1, 1], [1], False),
    ("Large", "Large_256x256_k3s2", (2, 256, 256, 256), [3, 3], [2, 2], [1, 1], [1], False),
]

DTYPES = [
    ("FP32", np.float32),
    ("FP16", np.float16),
]
if not is_310p:
    DTYPES.append(("BF16", ml_dtypes.bfloat16))

print("=" * 114, flush=True)
print(f"| {'Case Name':<22} | {'Dtype':<6} | {'Shape (N,C,H,W)':<18} | {'Kernel':<6} | {'Blocks':<6} | {'Latency (us)':<14} | {'Throughput':<14} |", flush=True)
print("=" * 114, flush=True)

results = []
for category, name, shape, k, s, p, d, ceil in TEST_CASES:
    attrs = [k, s, p, d, ceil]
    for dtype_str, dtype in DTYPES:
        x = np.random.uniform(-5, 5, size=shape).astype(dtype)
        pooled, mask = golden.maxpool2d_with_argmax_expect(x, *attrs)
        grad = np.random.uniform(-5, 5, size=pooled.shape).astype(dtype)
        
        total_elements = int(np.prod(shape))
        total_tiles = (total_elements + 8192 - 1) // 8192
        active_blocks = max(1, min(total_tiles, BLOCKS))

        # Hardware timing via aclrtEvent (warmup=20, iters=100)
        avg_us, actual = op.benchmark(grad, x, mask, *attrs, blocks=active_blocks, warmup=20, iters=100)
        
        # Verify correctness
        expected = golden.max_pool2d_with_mask_backward_golden(grad, x, mask, *attrs)[0]
        np.testing.assert_array_equal(actual, expected)
        assert actual.tobytes() == expected.tobytes()
        
        # Memory traffic: gradOutput (in) + indices (in) + gradInput (out)
        bytes_io = grad.nbytes + (grad.size * 4) + x.nbytes
        gb_s = (bytes_io / 1e9) / (avg_us * 1e-6) if avg_us > 0 else 0
        
        results.append({
            "category": category, "name": name, "dtype": dtype_str, "shape": list(shape),
            "kernel": k, "stride": s, "padding": p, "ceil_mode": ceil, "blocks": active_blocks,
            "latency_us": round(avg_us, 2), "throughput_gb_s": round(gb_s, 2), "status": "PASS"
        })
        
        print(f"| {name:<22} | {dtype_str:<6} | {str(shape):<18} | {str(k):<6} | {active_blocks:<6} | {avg_us:>10.2f} us   | {gb_s:>9.2f} GB/s  |", flush=True)

print("=" * 114, flush=True)

platform_name = "Atlas 300V Pro (Ascend 310P)" if is_310p else "Atlas 800T A2 (Ascend 910B)"
out_fname = '310p_hardware_benchmark.json' if is_310p else 'a2_hardware_benchmark.json'
out_path = ROOT / 'operator/reports' / out_fname
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "hardware": platform_name,
    "measurement": "Hardware ACL Event Timing (pure kernel execution, excludes memory allocation & PCIe transfers)",
    "blocks": BLOCKS,
    "warmup_iterations": 20,
    "benchmark_iterations": 100,
    "results": results
}, indent=2), encoding='utf-8')
print(f"\n[OK] Benchmark completed and saved to {out_path}", flush=True)
