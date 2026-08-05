"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""

import pytest
import torch
import ops_gnn

SUPPORTED_DTYPES = [torch.float16, torch.bfloat16, torch.float32,
                    torch.int8, torch.int16, torch.int32, torch.uint8]

REDUCES = ["sum", "add", "mul", "mean", "min", "max"]

TEST_SHAPES = [(128,), (1024,), (4096,), (8192,),
               (32, 512), (64, 768), (8, 16, 64), (4, 128, 256)]

GENERAL_SHAPES = [(1,), (2,), (4,), (2, 2), (1, 128),
                  (512, 768), (512, 1024), (1024, 768), (1024, 1024)]


def scatter_cpu(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    sc = src.cpu()
    ic = index.cpu().long()
    dim = dim % sc.dim()
    return_arg = reduce in ("min", "max")
    if dim_size is None:
        dim_size = int(ic.max().item()) + 1 if ic.numel() > 0 else 0
    out_size = list(sc.shape)
    out_size[dim] = dim_size
    if out is not None:
        oc = out.cpu().clone()
    elif reduce == "mul":
        oc = sc.new_ones(out_size)
    elif reduce == "min":
        oc = sc.new_full(out_size, float('inf'))
    elif reduce == "max":
        oc = sc.new_full(out_size, float('-inf'))
    else:
        oc = sc.new_zeros(out_size)
    cnt = sc.new_zeros(out_size) if reduce == "mean" else None
    argc = sc.new_zeros(out_size, dtype=torch.long) if return_arg else None
    sbc = sc.expand_as(ic) if sc.size(dim) == 1 else sc
    flat_before = 1
    for d in range(dim):
        flat_before *= sbc.shape[d]
    src_dim_size = sbc.size(dim)
    flat_after = 1
    for d in range(dim + 1, sbc.dim()):
        flat_after *= sbc.shape[d]
    sf = sbc.reshape(flat_before, src_dim_size, flat_after)
    idxf = ic.reshape(flat_before, src_dim_size, flat_after)
    of = oc.reshape(flat_before, dim_size, flat_after)
    cf = cnt.reshape(flat_before, dim_size, flat_after) if cnt is not None else None
    af = argc.reshape(flat_before, dim_size, flat_after) if argc is not None else None
    for b in range(flat_before):
        for j in range(flat_after):
            for i in range(src_dim_size):
                idx = int(idxf[b, i, j].item())
                val = sf[b, i, j].item()
                if reduce in ("sum", "add"):
                    of[b, idx, j] += val
                elif reduce == "mul":
                    of[b, idx, j] *= val
                elif reduce == "mean":
                    of[b, idx, j] += val
                    cf[b, idx, j] += 1
                elif reduce == "min":
                    if val < of[b, idx, j]:
                        of[b, idx, j] = val
                        if argc is not None:
                            af[b, idx, j] = i
                elif reduce == "max":
                    if val > of[b, idx, j]:
                        of[b, idx, j] = val
                        if argc is not None:
                            af[b, idx, j] = i
    if reduce == "mean" and cnt is not None:
        nz = cnt > 0
        if sc.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            oc[nz] = oc[nz].float().divide(cnt[nz].float()).floor().to(oc.dtype)
        else:
            oc[nz] = oc[nz] / cnt[nz]
    if return_arg:
        return oc, argc
    return oc


def _scatter_fn(reduce):
    return getattr(ops_gnn, f'scatter_{reduce}', None)


def _call_scatter(src, index, reduce, dim=-1):
    fn = _scatter_fn(reduce)
    if fn is None:
        return ops_gnn.scatter(src, index, dim=dim, reduce=reduce)
    return fn(src, index, dim=dim)


def _is_float_dtype(dtype):
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _assert_match(result, expected, dtype):
    if _is_float_dtype(dtype):
        assert torch.allclose(result.cpu(), expected.float(), rtol=1e-3, atol=1e-3)
    else:
        assert torch.equal(result.cpu(), expected)


class TestScatter:
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, reduce, dtype):
        if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32) and reduce in ("min", "max", "mean"):
            pytest.skip("min/max/mean not supported for integer dtypes in scatter")
        torch.npu.set_device(4)
        torch.manual_seed(42)
        dim = -1
        if len(shape) >= 2:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[dim] // 2), shape)
        else:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, 32, shape)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=dim)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=dim, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, reduce, dtype):
        if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32) and reduce in ("min", "max", "mean"):
            pytest.skip("min/max/mean not supported for integer dtypes in scatter")
        torch.npu.set_device(4)
        torch.manual_seed(42)
        if len(shape) >= 2:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[-1] // 2), shape)
        else:
            if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
                src = torch.randint(0, 128, shape).to(dtype)
            else:
                src = torch.randn(shape).to(dtype)
            index = torch.randint(0, max(1, shape[0] // 4), shape)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=-1)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=-1, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    def test_sum_add_equivalent(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 32, (128,), device='cpu')
        r1 = ops_gnn.scatter_sum(src.npu(), index.npu())
        r2 = ops_gnn.scatter_add(src.npu(), index.npu())
        assert torch.allclose(r1.cpu(), r2.cpu())

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_dim0(self, reduce):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, 32, device='cpu')
        index = torch.randint(0, 64, (128, 32), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=0)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=0, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        if isinstance(result, tuple):
            assert result[0].device.type == 'npu'
        else:
            assert result.device.type == 'npu'
            _assert_match(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_dim1(self, reduce):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, 32, device='cpu')
        index = torch.randint(0, 64, (128, 32), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=1)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=1, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        if isinstance(result, tuple):
            assert result[0].device.type == 'npu'
        else:
            assert result.device.type == 'npu'
            _assert_match(result, expected, torch.float32)

    def test_dim_size_one(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.zeros(128, dtype=torch.long)
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim_size=1)
        expected = scatter_cpu(src, index, dim_size=1, reduce="sum")
        assert result.shape[0] == 1
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_dim_size_large(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 32, (128,), device='cpu')
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim_size=10000)
        expected = scatter_cpu(src, index, dim_size=10000, reduce="sum")
        assert result.shape[0] == 10000
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    def test_empty_index(self):
        torch.npu.set_device(4)
        src = torch.randn(0, device='cpu')
        index = torch.empty(0, dtype=torch.long)
        result = ops_gnn.scatter_sum(src.npu(), index.npu())
        assert result.numel() == 0

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_with_out(self, reduce):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(32, 128, device='cpu')
        index = torch.randint(0, 256, (32, 128), device='cpu')
        out = torch.ones(32, 256, device='cpu')
        fn = _scatter_fn(reduce) or ops_gnn.scatter
        kwargs = {'dim': 1, 'out': out.npu()}
        if fn == ops_gnn.scatter:
            kwargs['reduce'] = reduce
        result = fn(src.npu(), index.npu(), **kwargs)
        if isinstance(result, tuple):
            assert result[0].device.type == 'npu'
        else:
            assert result.device.type == 'npu'

    def test_mean_int_floor(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randint(0, 10, (64,), dtype=torch.int32)
        index = torch.randint(0, 8, (64,))
        result = ops_gnn.scatter_mean(src.npu(), index.npu())
        expected = scatter_cpu(src, index, reduce="mean")
        assert torch.equal(result.cpu(), expected)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_high_conflict(self, reduce):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(1024, 64, device='cpu')
        index = torch.zeros(1024, 64, dtype=torch.long)
        result = _call_scatter(src.npu(), index.npu(), reduce, dim=1)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, dim=1, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        if isinstance(result, tuple):
            assert result[0].device.type == 'npu'
        else:
            assert result.device.type == 'npu'
            _assert_match(result, expected, torch.float32)

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_sparse_index(self, reduce):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(128, device='cpu')
        index = torch.randint(0, 10000, (128,), device='cpu')
        result = _call_scatter(src.npu(), index.npu(), reduce)
        if isinstance(result, tuple):
            result = result[0]
        expected = scatter_cpu(src, index, reduce=reduce)
        if isinstance(expected, tuple):
            expected = expected[0]
        if isinstance(result, tuple):
            assert result[0].device.type == 'npu'
        else:
            assert result.device.type == 'npu'
            _assert_match(result, expected, torch.float32)

    def test_large(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(512, 768, device='cpu')
        index = torch.randint(0, 1024, (512, 768), device='cpu')
        result = ops_gnn.scatter_sum(src.npu(), index.npu(), dim=1, dim_size=1024)
        expected = scatter_cpu(src, index, dim=1, dim_size=1024, reduce="sum")
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
