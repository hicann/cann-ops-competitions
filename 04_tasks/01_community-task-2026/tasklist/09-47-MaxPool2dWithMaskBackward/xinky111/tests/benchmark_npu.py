import importlib.util
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

LIBRARY = '/tmp/maxpool-a2-npu/libmax_pool_backward.so'
op = MaxPoolBackward(LIBRARY)

TEST_CASES = [
    ("Small_32x32", (1, 64, 32, 32), [3, 3], [1, 1], [1, 1], [1], False),
    ("Medium_64x64", (2, 128, 64, 64), [3, 3], [2, 2], [1, 1], [1], False),
    ("YOLOv11_SPPF_20x20", (1, 512, 20, 20), [5, 5], [1, 1], [2, 2], [1], False),
    ("YOLOv11_SPPF_40x40", (1, 256, 40, 40), [5, 5], [1, 1], [2, 2], [1], False),
    ("Large_64x64", (4, 256, 64, 64), [3, 3], [2, 2], [1, 1], [1], False),
]

DTYPES = [
    ("FP32", np.float32),
    ("FP16", np.float16),
    ("BF16", ml_dtypes.bfloat16)
]

print("=" * 85, flush=True)
print(f"{'Case Name':<20} | {'Dtype':<6} | {'Shape (N,C,H,W)':<18} | {'Kernel':<6} | {'Stride':<6} | {'Latency':<10}", flush=True)
print("=" * 85, flush=True)

for name, shape, k, s, p, d, ceil in TEST_CASES:
    attrs = [k, s, p, d, ceil]
    for dtype_str, dtype in DTYPES:
        x = np.random.uniform(-5, 5, size=shape).astype(dtype)
        pooled, mask = golden.maxpool2d_with_argmax_expect(x, *attrs)
        grad = np.random.uniform(-5, 5, size=pooled.shape).astype(dtype)
        
        # Warmup
        for _ in range(5):
            op(grad, x, mask, *attrs, blocks=8)
            
        # Benchmark iterations
        iters = 50
        start = time.perf_counter()
        for _ in range(iters):
            op(grad, x, mask, *attrs, blocks=8)
        elapsed = (time.perf_counter() - start) / iters * 1000.0
        
        print(f"{name:<20} | {dtype_str:<6} | {str(shape):<18} | {str(k):<6} | {str(s):<6} | {elapsed:>8.3f} ms", flush=True)

print("=" * 85, flush=True)
