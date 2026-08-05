"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""

import pytest
import torch
import ops_gnn

SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32,
                    torch.int8, torch.int16, torch.int32, torch.uint8]

TEST_SHAPES = [(128,), (1024,), (4096,), (8192,),
               (32, 512), (64, 768), (8, 16, 64), (4, 128, 256)]

GENERAL_SHAPES = [(1,), (4,), (3,), (2, 2), (1, 128),
                  (512, 768), (1024, 768), (1024, 1024)]


def gather_coo_cpu(src, index):
    sc = src.cpu(); ic = index.cpu().long()
    dim = ic.dim() - 1; sl = ic.size(dim)
    size = list(sc.shape); size[dim] = sl
    oc = sc.new_empty(size)
    sf = sc.reshape(-1, sc.size(dim))
    of = oc.reshape(-1, sl)
    idxf = ic.reshape(-1, sl)
    for b in range(sf.size(0)):
        bi = min(b, idxf.size(0) - 1)
        for j in range(sl):
            idx = int(idxf[bi, j].item())
            of[b, j] = sf[bi, idx]
    return oc


def _is_float_dtype(dtype):
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _assert_match(result, expected, dtype):
    if _is_float_dtype(dtype):
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)
    else:
        assert torch.equal(result.cpu(), expected)


class TestGatherCoo:
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            src = torch.randint(0, 128, shape).to(dtype)
        else:
            src = torch.randn(shape).to(dtype)
        seg = min(shape[-1] // 2 if len(shape) >= 2 else 32, 256)
        if len(shape) >= 2:
            idx = torch.randint(0, shape[-1], (shape[0], seg))
        else:
            idx = torch.randint(0, shape[0], (seg,))
        result = ops_gnn.gather_coo(src.npu(), idx.npu())
        expected = gather_coo_cpu(src, idx)
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            src = torch.randint(0, 128, shape).to(dtype)
        else:
            src = torch.randn(shape).to(dtype)
        seg = min(shape[-1] // 2 if len(shape) >= 2 else max(1, shape[0] // 4), 128)
        if len(shape) >= 2:
            idx = torch.randint(0, shape[-1], (max(1, shape[0] // 2), seg))
        else:
            idx = torch.randint(0, shape[0], (seg,))
        result = ops_gnn.gather_coo(src.npu(), idx.npu())
        expected = gather_coo_cpu(src, idx)
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    def test_empty_src(self):
        torch.npu.set_device(4)
        src = torch.empty(0, device='cpu')
        idx = torch.randint(0, 10, (0,))
        result = ops_gnn.gather_coo(src.npu(), idx.npu())
        assert result.numel() == 0

    def test_empty_index(self):
        torch.npu.set_device(4)
        src = torch.randn(128, device='cpu')
        idx = torch.empty(0, dtype=torch.long)
        result = ops_gnn.gather_coo(src.npu(), idx.npu())
        assert result.numel() == 0

    def test_dim_size_one(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        idx = torch.zeros(1, dtype=torch.long)
        result = ops_gnn.gather_coo(src.npu(), idx.npu())
        expected = gather_coo_cpu(src, idx)
        assert result.shape[0] == 1
        _assert_match(result, expected, torch.float32)

    def test_dim_size_large(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        idx = torch.randint(0, 128, (10000,))
        result = ops_gnn.gather_coo(src.npu(), idx.npu())
        expected = gather_coo_cpu(src, idx)
        assert result.shape[0] == 10000
        _assert_match(result, expected, torch.float32)

    def test_with_out(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(32, 512, device='cpu')
        idx = torch.randint(0, 512, (32, 64), device='cpu')
        out = torch.empty(32, 64, device='cpu')
        result = ops_gnn.gather_coo(src.npu(), idx.npu(), out=out.npu())
        expected = gather_coo_cpu(src, idx)
        assert result.device.type == 'npu'
        _assert_match(result, expected, torch.float32)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_large(self, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            src = torch.randint(0, 128, (512, 768)).to(dtype)
        else:
            src = torch.randn(512, 768).to(dtype)
        idx = torch.randint(0, 768, (512, 128), device='cpu')
        result = ops_gnn.gather_coo(src.npu(), idx.npu())
        expected = gather_coo_cpu(src, idx)
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
