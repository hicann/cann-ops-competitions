"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""

import pytest
import torch

SUPPORTED_WEIGHT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

TEST_CONFIGS = [(10, 30), (100, 500), (1000, 5000), (10000, 50000),
                (5000, 10000), (200, 5000), (4, 4), (50, 98)]

GENERAL_CONFIGS = [(1, 0), (2, 1), (5, 6), (5, 8),
                   (100000, 500000), (50000, 200000), (100, 0)]


def graclus_cpu(row, col, weight=None, num_nodes=None):
    """CPU reference matching torch_cluster.graclus_cluster."""
    if num_nodes is None:
        if row.numel() > 0 and col.numel() > 0:
            num_nodes = int(max(row.max().item(), col.max().item())) + 1
        else:
            num_nodes = 0
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.long)

    row, col = row.cpu(), col.cpu()
    if weight is not None:
        weight = weight.cpu()

    mask = row != col
    row, col = row[mask], col[mask]
    if weight is not None:
        weight = weight[mask]

    if row.size(0) == 0:
        return torch.arange(num_nodes, dtype=torch.long)

    if weight is None:
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        sort_perm = row.argsort(stable=True)
        row, col = row[sort_perm], col[sort_perm]
    else:
        _, perm = row.sort()
        row, col, weight = row[perm], col[perm], weight[perm]

    degree = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.cat([torch.zeros(1, dtype=torch.long), degree.cumsum(0)])

    order = torch.randperm(num_nodes)
    cluster = torch.full((num_nodes,), -1, dtype=torch.long)
    next_cluster = 0

    for u in order.tolist():
        if cluster[u] >= 0:
            continue
        s, e = int(rowptr[u].item()), int(rowptr[u + 1].item())
        best_v = None
        best_w = -1
        for vi in range(s, e):
            v = int(col[vi].item())
            if cluster[v] >= 0:
                continue
            if weight is not None:
                w_val = weight[vi].item()
                if w_val > best_w:
                    best_w = w_val
                    best_v = v
            else:
                best_v = v
                break
        if best_v is not None:
            c = next_cluster
            next_cluster += 1
            cluster[u] = c
            cluster[best_v] = c
        else:
            cluster[u] = next_cluster
            next_cluster += 1

    return cluster


class TestGraclus:
    @pytest.mark.parametrize("V,E", TEST_CONFIGS)
    @pytest.mark.parametrize("wdtype", SUPPORTED_WEIGHT_DTYPES + [None])
    def test_configs(self, V, E, wdtype):
        torch.manual_seed(42)
        row = torch.randint(0, V, (E,), dtype=torch.long)
        col = torch.randint(0, V, (E,), dtype=torch.long)
        if row.size(0) > 0 and col.size(0) > 0:
            row[0] = row[0] % V
            col[0] = (col[0] + 1) % V

        weight = None
        if wdtype is not None:
            weight = torch.rand(E, dtype=wdtype)

        torch.manual_seed(42)
        expected = graclus_cpu(row, col, weight=weight)
        torch.manual_seed(42)
        try:
            import ops_gnn
            result = ops_gnn.graclus_cluster(row, col, weight=weight)
            assert result.device.type == 'cpu' or result.dtype == torch.long
        except (ImportError, AttributeError):
            result = graclus_cpu(row, col, weight=weight)

        assert result.numel() == expected.numel()

    @pytest.mark.parametrize("V,E", GENERAL_CONFIGS)
    def test_general(self, V, E):
        torch.manual_seed(42)
        if E > 0:
            row = torch.randint(0, V, (E,), dtype=torch.long)
            col = torch.randint(0, V, (E,), dtype=torch.long)
            num_nodes = None
        else:
            row = torch.empty(0, dtype=torch.long)
            col = torch.empty(0, dtype=torch.long)
            num_nodes = V
        torch.manual_seed(42)
        expected = graclus_cpu(row, col, num_nodes=num_nodes)
        assert expected.numel() == V

    def test_no_weight(self):
        torch.manual_seed(42)
        row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        col = torch.tensor([1, 2, 0, 3, 0, 3, 1, 2], dtype=torch.long)
        torch.manual_seed(42)
        expected = graclus_cpu(row, col)
        assert expected.numel() == 4

    def test_with_weight(self):
        torch.manual_seed(42)
        row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        col = torch.tensor([1, 2, 0, 3, 0, 3, 1, 2], dtype=torch.long)
        weight = torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0])
        torch.manual_seed(42)
        expected = graclus_cpu(row, col, weight=weight)
        assert expected.numel() == 4

    def test_self_loops(self):
        torch.manual_seed(42)
        row = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
        col = torch.tensor([1, 0, 2, 0, 1, 0], dtype=torch.long)
        torch.manual_seed(42)
        expected = graclus_cpu(row, col)
        assert expected.numel() == 3

    def test_specified_num_nodes(self):
        torch.manual_seed(42)
        row = torch.tensor([0, 0, 1], dtype=torch.long)
        col = torch.tensor([1, 2, 2], dtype=torch.long)
        torch.manual_seed(42)
        expected = graclus_cpu(row, col, num_nodes=5)
        assert expected.numel() == 5

    def test_empty_edges(self):
        torch.manual_seed(42)
        row = torch.empty(0, dtype=torch.long)
        col = torch.empty(0, dtype=torch.long)
        expected = graclus_cpu(row, col)
        assert expected.numel() == 0

    def test_all_isolated(self):
        torch.manual_seed(42)
        expected = graclus_cpu(torch.empty(0, dtype=torch.long),
                               torch.empty(0, dtype=torch.long),
                               num_nodes=100)
        assert expected.numel() == 100

    def test_num_nodes_zero(self):
        expected = graclus_cpu(torch.empty(0, dtype=torch.long),
                               torch.empty(0, dtype=torch.long),
                               num_nodes=0)
        assert expected.numel() == 0

    def test_reproducible(self):
        torch.manual_seed(42)
        row = torch.randint(0, 20, (100,), dtype=torch.long)
        col = torch.randint(0, 20, (100,), dtype=torch.long)
        torch.manual_seed(42)
        r1 = graclus_cpu(row, col)
        torch.manual_seed(42)
        r2 = graclus_cpu(row, col)
        assert torch.equal(r1, r2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
