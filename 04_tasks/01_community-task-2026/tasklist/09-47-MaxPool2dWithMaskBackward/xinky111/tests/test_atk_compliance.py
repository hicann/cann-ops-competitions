"""Test operator compliance against Huawei Official ATK (Ascend Test Kit) protocol.
This script replicates the exact data packing and golden verification from:
MethodTorchNnMaxPool2DWithMaskBackwardApi
"""
import math
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'operator/python'))
from max_pool_backward import MaxPoolBackward

LIBRARY = '/tmp/maxpool-a2-npu/libmax_pool_backward.so'
if not os.path.exists(LIBRARY):
    # Try finding build lib in local or deliverables
    for candidate in [
        ROOT / 'operator_build/libcust_opapi.so',
        ROOT / 'operator/build/libmax_pool_backward.so'
    ]:
        if candidate.exists():
            LIBRARY = str(candidate)
            break

print("======================================================================")
print("     Huawei Official ATK Test Protocol Compliance Verification        ")
print("======================================================================")

has_torch = False
try:
    import torch
    has_torch = True
    print(f"[Environment]: PyTorch {torch.__version__} detected.")
except ImportError:
    print("[Environment]: PyTorch not installed in Python environment, using NumPy equivalent.")

TEST_CASES = [
    # (Name, N, C, H, W, k, s, p, d, ceil)
    ("ATK_Standard_k3s2", (2, 16, 32, 32), [3, 3], [2, 2], [1, 1], [1, 1], False),
    ("ATK_SPPF_k5s1", (1, 32, 20, 20), [5, 5], [1, 1], [2, 2], [1, 1], False),
    ("ATK_CeilMode_k2s2", (2, 8, 15, 17), [2, 2], [2, 2], [0, 0], [1, 1], True),
]

if os.path.exists(LIBRARY):
    op = MaxPoolBackward(LIBRARY)
    all_passed = True

    for name, shape, k, s, p, d, ceil in TEST_CASES:
        n, c, h, w = shape
        kh, kw = k
        sh, sw = s
        ph, pw = p
        
        if has_torch:
            # Replicate exact ATK init_by_input_data logic
            x_cpu = torch.randn(shape, dtype=torch.float32)
            m = torch.nn.MaxPool2d(kernel_size=k, stride=s, padding=p, dilation=d, return_indices=True, ceil_mode=ceil)
            out_cpu, indices_cpu = m(x_cpu)
            
            ho, wo = indices_cpu.shape[2], indices_cpu.shape[3]
            indices_npu_shape = [n, c, kh * kw, (math.ceil(ho * wo / 16) + 1) * 32]
            pad_size = n * c * kh * kw * (math.ceil(ho * wo / 16) + 1) * 8 - indices_cpu.numel()
            flatten_indices_cpu = indices_cpu.to(torch.int32).flatten()
            zero_padding = torch.zeros(int(pad_size), dtype=torch.int32)
            indices_npu = torch.cat((flatten_indices_cpu, zero_padding)).view(torch.int8).reshape(indices_npu_shape)
            
            gradOut = torch.randn_like(out_cpu)
            
            # ATK Golden: torch.ops.aten.max_pool2d_with_indices_backward
            gradIn_golden = torch.ops.aten.max_pool2d_with_indices_backward(
                gradOut, x_cpu, k, s, p, d, ceil, indices_cpu.to(torch.int64)
            ).numpy()
            
            # Custom Ascend C NPU Kernel Execution
            actual = op.run(gradOut.numpy(), x_cpu.numpy(), indices_npu.numpy(), k, s, p, d, ceil)
            
            diff = np.max(np.abs(actual - gradIn_golden))
            if diff == 0.0:
                print(f"[PASS] {name:<20}: ATK Mask Packing & PyTorch ATen Golden 100% Bitwise Equal (diff=0.0)")
            else:
                print(f"[FAIL] {name:<20}: Max diff = {diff}")
                all_passed = False
        else:
            # Fallback to NumPy reference
            import max_pool2d_with_mask_backward_golden as golden
            x = np.random.uniform(-5, 5, size=shape).astype(np.float32)
            pooled, mask = golden.maxpool2d_with_argmax_expect(x, k, s, p, d, ceil)
            grad = np.random.uniform(-5, 5, size=pooled.shape).astype(np.float32)
            expected = golden.max_pool2d_with_mask_backward_golden(grad, x, mask, k, s, p, d, ceil)[0]
            actual = op.run(grad, x, mask, k, s, p, d, ceil)
            np.testing.assert_array_equal(actual, expected)
            print(f"[PASS] {name:<20}: Exact Bitwise Match with Golden (diff=0.0)")
            
    print("======================================================================")
    if all_passed:
        print("[SUCCESS] All ATK Protocol Compliance Checks Passed Successfully!")
    print("======================================================================")
else:
    print(f"[INFO] Kernel library {LIBRARY} not found (will be tested on NPU).")
