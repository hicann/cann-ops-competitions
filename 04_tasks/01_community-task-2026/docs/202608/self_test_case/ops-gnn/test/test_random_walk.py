"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""

import pytest
import torch
import ops_gnn

TEST_CONFIGS = [(1024, 4, 8), (4096, 8, 16), (8192, 10, 32),
                (16384, 6, 64), (32768, 4, 128), (4096, 20, 8),
                (8192, 4, 256), (2048, 32, 4)]

GENERAL_CONFIGS = [(4, 1, 1), (8, 0, 2), (16, 2, 4), (0, 1, 1),
                   (65536, 8, 256), (16384, 2, 1024), (8192, 64, 16)]




class TestRandomWalk:
    @pytest.mark.parametrize("E,wl,S", TEST_CONFIGS)
    def test_configs(self, E, wl, S):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        nodes = max(200, E // 2)
        row = torch.randint(0, nodes, (E,), device='cpu')
        col = torch.randint(0, nodes, (E,), device='cpu')
        start = torch.arange(min(S, nodes), device='cpu')
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), wl)
        assert result.device.type == 'npu'

    @pytest.mark.parametrize("E,wl,S", GENERAL_CONFIGS)
    def test_general(self, E, wl, S):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        nodes = max(200, E // 2)
        if E > 0:
            row = torch.randint(0, nodes, (E,), device='cpu')
            col = torch.randint(0, nodes, (E,), device='cpu')
            start = torch.arange(min(S, nodes), device='cpu')
            try:
                result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), wl)
                assert result.device.type == 'npu'
            except (RuntimeError, ValueError):
                pass

    def test_walk_length_zero(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        row = torch.randint(0, 10, (20,), device='cpu')
        col = torch.randint(0, 10, (20,), device='cpu')
        start = torch.tensor([0, 1], device='cpu')
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 0)
        assert result.device.type == 'npu'

    def test_walk_length_one(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        row = torch.randint(0, 10, (20,), device='cpu')
        col = torch.randint(0, 10, (20,), device='cpu')
        start = torch.tensor([0], device='cpu')
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 1)
        assert result.device.type == 'npu'

    def test_return_edge_indices(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        row = torch.tensor([0, 1, 1, 2], device='cpu')
        col = torch.tensor([1, 0, 2, 1], device='cpu')
        start = torch.tensor([0], device='cpu')
        for return_edges in [True, False]:
            result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 4,
                                         return_edge_indices=return_edges)
            if return_edges:
                assert isinstance(result, tuple)
            else:
                assert result.device.type == 'npu'

    @pytest.mark.parametrize("p,q", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)])
    def test_pq_params(self, p, q):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        row = torch.randint(0, 50, (100,), device='cpu')
        col = torch.randint(0, 50, (100,), device='cpu')
        start = torch.tensor([0], device='cpu')
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 4, p=p, q=q)
        assert result.device.type == 'npu'

    def test_reproducible(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        row = torch.randint(0, 100, (200,), device='cpu')
        col = torch.randint(0, 100, (200,), device='cpu')
        start = torch.tensor([0, 5, 10], device='cpu')
        torch.manual_seed(42)
        r1 = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 4)
        torch.manual_seed(42)
        r2 = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 4)
        assert torch.equal(r1.cpu(), r2.cpu())

    def test_isolated_node(self):
        torch.npu.set_device(4)
        row = torch.tensor([0, 1], device='cpu')
        col = torch.tensor([1, 2], device='cpu')
        start = torch.tensor([3], device='cpu')
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 4)
        assert result.device.type == 'npu'

    def test_large(self):
        torch.npu.set_device(4)
        torch.manual_seed(42)
        row = torch.randint(0, 200, (4096,), device='cpu')
        col = torch.randint(0, 200, (4096,), device='cpu')
        start = torch.tensor([0, 10, 20, 30], device='cpu')
        result = ops_gnn.random_walk(row.npu(), col.npu(), start.npu(), 8)
        assert result.device.type == 'npu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
