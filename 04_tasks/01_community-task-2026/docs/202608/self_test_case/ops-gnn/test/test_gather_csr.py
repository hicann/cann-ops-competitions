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

GENERAL_SHAPES = [(1,), (4,), (2, 2), (1, 128),
                  (512, 768), (1024, 768), (1024, 1024)]


def _make_indptr(N, nseg, varied=False):
    if varied:
        sizes = torch.randint(1, max(2, N // nseg), (nseg,))
        sizes = (sizes.float() / sizes.sum() * N).long()
        sizes[sizes < 1] = 1
        diff = N - sizes.sum().item()
        sizes[-1] += diff
    else:
        base = N // nseg
        sizes = torch.full((nseg,), base)
        sizes[-1] += N - base * nseg
    return torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])


def gather_csr_cpu(src, indptr):
    sc = src.cpu(); ipc = indptr.cpu().long()
    nseg = ipc.numel() - 1
    total = int(ipc[-1].item())
    out_shape = list(sc.shape)
    out_shape[-1] = total
    result = sc.new_zeros(out_shape)
    for seg in range(nseg):
        start = int(ipc[seg].item())
        end = int(ipc[seg + 1].item())
        if start >= end or seg >= sc.size(-1):
            continue
        src_val = sc.select(-1, seg)
        tgt_slice = result.narrow(-1, start, end - start)
        tgt_slice.copy_(src_val.unsqueeze(-1).expand_as(tgt_slice))
    return result


def _is_float_dtype(dtype):
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _assert_match(result, expected, dtype):
    if _is_float_dtype(dtype):
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)
    else:
        assert torch.equal(result.cpu(), expected)


class TestGatherCsr:
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        Ndense = shape[-1] if len(shape) >= 2 else shape[0]
        nseg = max(4, min(Ndense // 32, 32))
        seg_shape = list(shape)
        seg_shape[-1] = nseg
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            src = torch.randint(0, 128, seg_shape).to(dtype)
        else:
            src = torch.randn(seg_shape).to(dtype)
        indptr = _make_indptr(Ndense, nseg)
        result = ops_gnn.gather_csr(src.npu(), indptr.npu())
        expected = gather_csr_cpu(src, indptr)
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        Ndense = shape[-1] if len(shape) >= 2 else shape[0]
        nseg = max(2, min(Ndense // 4, 16))
        seg_shape = list(shape)
        seg_shape[-1] = nseg
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            src = torch.randint(0, 128, seg_shape).to(dtype)
        else:
            src = torch.randn(seg_shape).to(dtype)
        indptr = _make_indptr(Ndense, nseg)
        result = ops_gnn.gather_csr(src.npu(), indptr.npu())
        expected = gather_csr_cpu(src, indptr)
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)

    def test_empty_src(self):
        torch.npu.set_device(4)
        src = torch.empty(0, device='cpu')
        indptr = torch.tensor([0, 0, 0])
        result = ops_gnn.gather_csr(src.npu(), indptr.npu())
        assert result.numel() == 0

    def test_empty_segment(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(5, device='cpu')
        indptr = torch.tensor([0, 0, 4, 4, 8])
        result = ops_gnn.gather_csr(src.npu(), indptr.npu())
        expected = gather_csr_cpu(src, indptr)
        assert result.device.type == 'npu'
        _assert_match(result, expected, torch.float32)

    def test_single_segment(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(1, device='cpu')
        indptr = torch.tensor([0, 128])
        result = ops_gnn.gather_csr(src.npu(), indptr.npu())
        expected = gather_csr_cpu(src, indptr)
        assert result.device.type == 'npu'
        _assert_match(result, expected, torch.float32)

    def test_with_out(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        src = torch.randn(32, 4, device='cpu')
        indptr = torch.tensor([0, 8, 16, 24, 32])
        out = torch.zeros(32, 32, device='cpu')
        result = ops_gnn.gather_csr(src.npu(), indptr.npu(), out=out.npu())
        expected = gather_csr_cpu(src, indptr)
        assert torch.allclose(result.cpu(), expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_large(self, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        nseg = max(2, 768 // 48)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            src = torch.randint(0, 128, (512, nseg)).to(dtype)
        else:
            src = torch.randn(512, nseg).to(dtype)
        indptr = torch.arange(0, 769, 48).long()
        result = ops_gnn.gather_csr(src.npu(), indptr.npu())
        expected = gather_csr_cpu(src, indptr)
        assert result.device.type == 'npu'
        _assert_match(result, expected, dtype)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
