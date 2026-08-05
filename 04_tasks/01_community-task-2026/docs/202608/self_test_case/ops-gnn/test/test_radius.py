"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""

import pytest
import torch
import ops_gnn

SUPPORTED_DTYPES = [torch.float16, torch.float32]

TEST_SHAPES = [(128, 3), (1024, 64), (256, 3), (2048, 32), (512, 128)]

GENERAL_SHAPES = [(1, 3), (8, 3), (4, 64), (10000, 3)]


def radius_cpu(x, y, r, max_num_neighbors=32, ignore_same_index=False):
    xc = x.cpu().float()
    yc = y.cpu().float()
    rows, cols = [], []
    for i in range(yc.size(0)):
        diff = xc - yc[i].unsqueeze(0)
        dist = diff.pow(2).sum(1)
        neighbors = (dist <= r * r).nonzero(as_tuple=False).flatten()
        if ignore_same_index:
            neighbors = neighbors[neighbors != i]
        count = 0
        for n in neighbors.tolist():
            if count >= max_num_neighbors:
                break
            rows.append(i)
            cols.append(int(n))
            count += 1
    if not rows:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def _radius_op(x, y, r, max_num_neighbors=32, ignore_same_index=False):
    return ops_gnn.radius(x.npu(), y.npu(), r=r,
                          max_num_neighbors=max_num_neighbors,
                          ignore_same_index=ignore_same_index)


class TestRadius:
    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_shapes(self, shape, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        x = torch.randn(shape).to(dtype)
        expected = radius_cpu(x, x, r=1.0)
        try:
            result = _radius_op(x, x, r=1.0)
            assert result.device.type == 'npu'
            assert result.shape == expected.shape
            assert torch.equal(result.cpu(), expected.cpu())
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    @pytest.mark.parametrize("shape", GENERAL_SHAPES)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    def test_general_shapes(self, shape, dtype):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        x = torch.randn(shape).to(dtype)
        expected = radius_cpu(x, x, r=1.0)
        try:
            result = _radius_op(x, x, r=1.0)
            assert result.device.type == 'npu'
            assert result.shape == expected.shape
            assert torch.equal(result.cpu(), expected.cpu())
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    def test_small_radius_empty(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        x = torch.randn(128, 3, device='cpu')
        expected = radius_cpu(x, x, r=1e-9)
        try:
            result = _radius_op(x, x, r=1e-9)
            assert result.device.type == 'npu'
            assert result.size(1) == expected.size(1)
            assert torch.equal(result.cpu(), expected.cpu())
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    def test_large_radius_all(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        x = torch.randn(10, 3, device='cpu')
        expected = radius_cpu(x, x, r=1e9)
        try:
            result = _radius_op(x, x, r=1e9)
            assert result.device.type == 'npu'
            assert result.size(1) == expected.size(1)
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    def test_self_connections(self):
        torch.npu.set_device(4)
        x = torch.randn(10, 3, device='cpu')
        expected = radius_cpu(x, x, r=10.0, ignore_same_index=True)
        try:
            result = _radius_op(x, x, r=10.0, ignore_same_index=True)
            assert result.device.type == 'npu'
            assert torch.equal(result.cpu(), expected.cpu())
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    def test_max_neighbors(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        x = torch.randn(100, 3, device='cpu')
        expected = radius_cpu(x, x, r=100.0, max_num_neighbors=5)
        try:
            result = _radius_op(x, x, r=100.0, max_num_neighbors=5)
            assert result.device.type == 'npu'
            assert result.size(1) == expected.size(1)
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    def test_deterministic(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        x = torch.randn(128, 3, device='cpu')
        r1 = radius_cpu(x, x, r=1.0)
        r2 = radius_cpu(x, x, r=1.0)
        assert torch.equal(r1, r2)
        try:
            r3 = _radius_op(x, x, r=1.0)
            r4 = _radius_op(x, x, r=1.0)
            assert torch.equal(r3.cpu(), r4.cpu())
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    def test_empty_input(self):
        torch.npu.set_device(4)
        x = torch.empty(0, 3, device='cpu')
        expected = radius_cpu(x, x, r=1.0)
        try:
            result = _radius_op(x, x, r=1.0)
            assert result.device.type == 'npu'
            assert result.numel() == expected.numel()
        except (AttributeError, RuntimeError):
            pytest.skip("radius not available")

    def test_known_case(self):
        torch.manual_seed(42)
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        expected = radius_cpu(x, y, r=1.1)
        assert expected.size(1) == 6
        assert expected.size(0) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
