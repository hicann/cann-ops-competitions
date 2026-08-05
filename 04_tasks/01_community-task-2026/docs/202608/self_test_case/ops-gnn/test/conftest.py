import pytest
import torch


@pytest.fixture(autouse=True)
def clear_npu_cache():
    yield
    if hasattr(torch, 'npu'):
        try:
            torch.npu.synchronize()
            torch.npu.empty_cache()
        except Exception:
            pass
