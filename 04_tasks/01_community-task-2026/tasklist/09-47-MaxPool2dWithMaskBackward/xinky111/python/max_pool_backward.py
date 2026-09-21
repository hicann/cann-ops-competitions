"""Synchronous host NumPy adapter for the compiled Ascend C standalone library.

This is not an aclnn binding. Input views are packed; optional output views are
unpacked after the actual kernel has completed. Calls are serialized because the
standalone C++ harness owns ACL initialization/finalization.
"""
import ctypes
import math
import threading
from pathlib import Path

import ml_dtypes
import numpy as np

_LOCK = threading.Lock()
_DTYPES = {np.dtype(np.float32): 0, np.dtype(np.float16): 1, np.dtype(ml_dtypes.bfloat16): 2}


class Tiling(ctypes.Structure):
    _fields_ = [(k, ctypes.c_int64) for k in (
        'planes', 'height', 'width', 'outHeight', 'outWidth',
        'kernelH', 'kernelW', 'strideH', 'strideW', 'padH', 'padW')]


def _pair(value, name, default=None):
    if not isinstance(value, (list, tuple)) or len(value) not in (0, 1, 2):
        raise ValueError(f'{name}: expected list/tuple of length 1 or 2')
    if len(value) == 0:
        if default is None:
            raise ValueError(f'{name}: empty value')
        return default
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value):
        raise ValueError(f'{name}: integer values required')
    values = (int(value[0]), int(value[-1]))
    if any(abs(v) > 2**31 - 1 for v in values):
        raise ValueError(f'{name}: exceeds supported int32 attribute range')
    return values


def _dimension(size, kernel, stride, pad, ceil):
    numerator = size + 2*pad - kernel
    result = (numerator + (stride - 1 if ceil else 0)) // stride + 1
    if ceil and (result - 1)*stride >= size + pad:
        result -= 1
    if result <= 0:
        raise ValueError('nonpositive pooled dimension')
    return result


def _check_output_view(out):
    # Fast path for normal transposes/slices, exact fallback for interleaved views.
    span = out.itemsize
    for stride, size in sorted((abs(s), n) for s, n in zip(out.strides, out.shape) if n > 1):
        if stride < span:
            offsets = np.array([0], dtype=np.int64)
            for length, step in zip(out.shape, out.strides):
                offsets = (offsets[:,None] + np.arange(length,dtype=np.int64)[None,:]*step).reshape(-1)
            offsets.sort()
            if np.any(np.diff(offsets) < out.itemsize):
                raise ValueError('output view overlaps itself')
            return
        span += (size - 1)*stride


class MaxPoolBackward:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.library.mp_last_error.restype = ctypes.c_char_p
        self.library.mp_run.argtypes = [ctypes.c_void_p]*3 + [ctypes.POINTER(Tiling), ctypes.c_int, ctypes.c_int]
        self.library.mp_run.restype = ctypes.c_int
        self.library.mp_benchmark_kernel.argtypes = [
            ctypes.c_void_p]*3 + [ctypes.POINTER(Tiling), ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        self.library.mp_benchmark_kernel.restype = ctypes.c_int

    def benchmark(self, grad_output, self_tensor, indices, kernel_size, stride,
                  padding, dilation, ceil_mode=False, *, blocks=8, warmup=20, iters=100):
        g, x, mask = map(np.asarray, (grad_output, self_tensor, indices))
        if g.ndim != 4 or x.ndim != 4 or mask.ndim != 4:
            raise ValueError('expected four-dimensional logical NCHW tensors')
        if g.dtype not in _DTYPES or x.dtype != g.dtype or mask.dtype != np.int8:
            raise ValueError('unsupported or mismatched dtype')
        n, c, h, w = x.shape
        kh, kw = _pair(kernel_size, 'kernel_size')
        sh, sw = _pair(stride, 'stride', (kh, kw))
        ph, pw = _pair(padding, 'padding')
        ho, wo = _dimension(h, kh, sh, ph, ceil_mode), _dimension(w, kw, sw, pw, ceil_mode)
        count = n*c*ho*wo
        grad = np.ascontiguousarray(g)
        raw = np.ascontiguousarray(mask).reshape(-1)
        idx = raw[:count*4].view('<i4').copy()
        result = np.empty(x.shape, dtype=g.dtype)
        t = Tiling(n*c,h,w,ho,wo,kh,kw,sh,sw,ph,pw)
        avg_us = ctypes.c_float(0.0)
        with _LOCK:
            status = self.library.mp_benchmark_kernel(
                grad.ctypes.data, idx.ctypes.data, result.ctypes.data,
                ctypes.byref(t), _DTYPES[g.dtype], blocks, warmup, iters, ctypes.byref(avg_us))
            if status:
                raise ValueError(self.library.mp_last_error().decode('utf-8'))
        return avg_us.value, result

    def __call__(self, grad_output, self_tensor, indices, kernel_size, stride,
                 padding, dilation, ceil_mode=False, *, out=None, blocks=1):
        g, x, mask = map(np.asarray, (grad_output, self_tensor, indices))
        if g.ndim != 4 or x.ndim != 4 or mask.ndim != 4:
            raise ValueError('expected four-dimensional logical NCHW tensors')
        if g.dtype not in _DTYPES or x.dtype != g.dtype or mask.dtype != np.int8:
            raise ValueError('unsupported or mismatched dtype')
        if not isinstance(ceil_mode, (bool, np.bool_)):
            raise ValueError('ceil_mode must be boolean')
        if isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= 32:
            raise ValueError('blocks must be integer in [1,32]')
        if min(x.shape) <= 0:
            raise ValueError('empty input dimensions are unsupported')
        kh, kw = _pair(kernel_size, 'kernel_size')
        sh, sw = _pair(stride, 'stride', (kh, kw))
        ph, pw = _pair(padding, 'padding')
        if min(kh, kw, sh, sw) <= 0 or ph < 0 or pw < 0 or ph > kh or pw > kw:
            raise ValueError('invalid pooling geometry')
        if _pair(dilation, 'dilation') != (1, 1):
            raise ValueError('dilation must equal 1')
        n, c, h, w = x.shape
        if h*w > 2**31-1 or math.prod(x.shape) > 2**63-257:
            raise ValueError('tensor size exceeds supported indexing range')
        ho, wo = _dimension(h, kh, sh, ph, ceil_mode), _dimension(w, kw, sw, pw, ceil_mode)
        if g.shape != (n,c,ho,wo):
            raise ValueError('gradOutput shape mismatch')
        if mask.shape != (n,c,kh*kw,((ho*wo+15)//16+1)*32):
            raise ValueError('indices container shape mismatch')
        count = n*c*ho*wo
        if mask.nbytes < count*4:
            raise ValueError('indices container cannot hold int32 prefix (including some 1x1 geometries)')
        if out is not None:
            if not isinstance(out, np.ndarray) or out.shape != x.shape or out.dtype != g.dtype or not out.flags.writeable:
                raise ValueError('invalid output shape/dtype/writeability')
            _check_output_view(out)
            if any(np.shares_memory(out, v) for v in (g, x, mask)):
                raise ValueError('output must not alias an input')
        grad = np.ascontiguousarray(g)
        if np.isnan(grad.astype(np.float32)).any() or np.isneginf(grad.astype(np.float32)).any():
            raise ValueError('NaN and negative infinity gradients are unsupported')
        # Logical flattening comes before reinterpretation, including noncontiguous masks.
        raw = np.ascontiguousarray(mask).reshape(-1)
        idx = raw[:count*4].view('<i4').copy()
        result = np.empty(x.shape, dtype=g.dtype)
        t = Tiling(n*c,h,w,ho,wo,kh,kw,sh,sw,ph,pw)
        with _LOCK:
            status = self.library.mp_run(grad.ctypes.data, idx.ctypes.data, result.ctypes.data,
                                         ctypes.byref(t), _DTYPES[g.dtype], blocks)
            if status:
                raise ValueError(self.library.mp_last_error().decode('utf-8'))
        if out is not None:
            out[...] = result
            return out
        return result
