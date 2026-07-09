"""CUDA event timing helpers for gpu_test performance runs."""
from __future__ import annotations

from typing import Callable, Dict, List

import torch


def cuda_event_benchmark(fn: Callable[[], None], warmup: int = 5, repeat: int = 20) -> Dict[str, float]:
    """Return min/avg/median GPU elapsed time (ms) for ``fn``.

    ``fn`` must enqueue GPU work (typically one ``gpu_call`` = ATK ``_gpu_call`` scope).
    Uses CUDA events; ``elapsed_time`` is in milliseconds by PyTorch API convention.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for GPU performance benchmark')

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))

    samples.sort()
    n = len(samples)
    median = samples[n // 2] if n % 2 == 1 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    return {
        'min_ms': samples[0],
        'avg_ms': sum(samples) / n,
        'median_ms': median,
        'max_ms': samples[-1],
        'repeat': float(n),
    }
