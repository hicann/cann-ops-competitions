import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig
from atk.configs.design_config import StandardConfig


MEM_LIMIT = 20 * 1024 * 1024 * 1024

# SDDMM: C = alpha * (op(A) * op(B)) o spy(C) + beta * C
# 覆盖任务书 dtype 组合 U1 / M1 / M2 (computeType 均为 fp32=0)
DTYPE_CASES = {
    0: {'A': 'fp32', 'B': 'fp32', 'C': 'fp32', 'out': 'fp32', 'computeType': 0},   # U1
    1: {'A': 'fp16', 'B': 'fp16', 'C': 'fp32', 'out': 'fp32', 'computeType': 0},   # M1
    2: {'A': 'fp16', 'B': 'fp16', 'C': 'fp16', 'out': 'fp16', 'computeType': 0},   # M2
}

# scenario 编码 (写入 attr scenario, 便于 executor 识别特殊用例)
SCENARIO_NORMAL = 0
SCENARIO_BETA_ZERO = 1       # TC-01
SCENARIO_BETA_NONZERO = 2    # TC-02
SCENARIO_OUTER_1D = 3        # TC-03 1D 外积
SCENARIO_GNN_EDGE = 4        # TC-04 GNN (E,1)*(1,E)
SCENARIO_ZERO_NNZ = 5        # TC-06 空 nnz
SCENARIO_NONCONTIG = 6       # 非连续 stride
SCENARIO_WIDE_SHORT = 7      # S3 类宽短矩阵 m>>n
SCENARIO_FIXED_S1 = 8
SCENARIO_FIXED_S2 = 9
SCENARIO_FIXED_S3 = 10

K_CHOICES = [16, 64, 128, 256, 512]


def check_memory_limit(m, k, n, elem_bytes=4):
    items = m * k + k * n + m * n
    mem = items * elem_bytes
    if mem > MEM_LIMIT:
        i = random.randint(0, 2)
        if i == 0:
            m = max(int(m // 3 * 2), 1)
        elif i == 1:
            k = max(int(k // 3 * 2), 1)
        else:
            n = max(int(n // 3 * 2), 1)
        m, k, n = check_memory_limit(m, k, n, elem_bytes)
    return m, k, n


def sample_from_range(range_values):
    if isinstance(range_values, (list, tuple)) and len(range_values) == 2:
        lo, hi = float(range_values[0]), float(range_values[1])
        return round(random.uniform(lo, hi), 4)
    return range_values


def pick_scenario(counter):
    """按 counter 轮转覆盖任务书各类场景, 保证 ~1000 条内各类均有足够样本。"""
    slot = counter % 100
    if slot < 3:
        return SCENARIO_FIXED_S1 + slot  # S1/S2/S3 各 1 条/100
    if slot == 3:
        return SCENARIO_ZERO_NNZ
    if slot == 4:
        return SCENARIO_GNN_EDGE
    if slot == 5:
        return SCENARIO_OUTER_1D
    if slot == 6:
        return SCENARIO_NONCONTIG
    if slot == 7:
        return SCENARIO_WIDE_SHORT
    if slot < 18:
        return SCENARIO_BETA_NONZERO   # ~10% beta!=0
    if slot < 68:
        return SCENARIO_BETA_ZERO      # ~50% beta=0
    return SCENARIO_NORMAL


def apply_scenario(scenario, m, k, n, sparsity, beta):
    if scenario == SCENARIO_FIXED_S1:
        return 10000, 64, 10000, 0.999, 0.0
    if scenario == SCENARIO_FIXED_S2:
        return 100000, 128, 100000, 0.9999, 0.0
    if scenario == SCENARIO_FIXED_S3:
        return 1000000, 32, 100, 0.99, 0.0
    if scenario == SCENARIO_ZERO_NNZ:
        return max(m, 128), k, max(n, 128), 1.0, 0.0
    if scenario == SCENARIO_GNN_EDGE:
        e = random.randint(1000, 50000)
        return e, 1, e, round(random.uniform(0.95, 0.999), 4), beta
    if scenario == SCENARIO_OUTER_1D:
        if random.choice([True, False]):
            return 1, random.choice(K_CHOICES), random.randint(100, 10000), round(random.uniform(0.9, 0.999), 4), beta
        return random.randint(100, 10000), random.choice(K_CHOICES), 1, round(random.uniform(0.9, 0.999), 4), beta
    if scenario == SCENARIO_WIDE_SHORT:
        m = random.randint(10000, 100000)
        n = random.randint(10, 500)
        k = random.choice(K_CHOICES)
        sparsity = round(random.uniform(0.99, 0.9999), 4)
        return m, k, n, sparsity, beta
    if scenario == SCENARIO_BETA_ZERO:
        return m, k, n, sparsity, 0.0
    if scenario == SCENARIO_BETA_NONZERO:
        return m, k, n, sparsity, round(random.uniform(-5, 5), 4) if beta == 0 else beta
    return m, k, n, sparsity, beta


@GENERATOR_REGISTRY.register("aclSparseSddmm")
class AclSparseSddmmGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.counter = 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        dc = DTYPE_CASES[self.counter % len(DTYPE_CASES)]
        scenario = pick_scenario(self.counter)

        # 常规规模: m/n 10^3~10^5, k 从 K_CHOICES 抽样
        m = random.randint(1000, 50000)
        n = random.randint(1000, 50000)
        k = random.choice(K_CHOICES)
        sparsity = round(random.uniform(0.999 - 0.0001 * random.randint(1, 90), 0.99999), 4)
        beta = sample_from_range(case_config.inputs[5].range_values)

        m, k, n, sparsity, beta = apply_scenario(scenario, m, k, n, sparsity, beta)
        alpha = sample_from_range(case_config.inputs[4].range_values)

        opA = random.choice([0, 1])
        opB = random.choice([0, 1])
        if opB == 1 and k < n:
            k, n = n, k

        orderA = 0
        orderB = 0
        ldA = k if opA == 0 else m
        ldB = n if opB == 0 else k
        if scenario == SCENARIO_NONCONTIG:
            ldA = ldA + random.randint(1, 64)
            ldB = ldB + random.randint(1, 64)

        elem_bytes = 4 if dc['A'] == 'fp32' else 2
        m, k, n = check_memory_limit(m, k, n, elem_bytes)

        if opA == 0:
            case_config.inputs[0].shape = [m, k]
        else:
            case_config.inputs[0].shape = [k, m]
        case_config.inputs[0].dtype = dc['A']

        if opB == 0:
            case_config.inputs[1].shape = [k, n]
        else:
            case_config.inputs[1].shape = [n, k]
        case_config.inputs[1].dtype = dc['B']

        case_config.inputs[2].shape = [m, n]
        case_config.inputs[2].dtype = dc['C']

        case_config.inputs[3].range_values = sparsity
        case_config.inputs[4].range_values = alpha
        case_config.inputs[5].range_values = beta
        case_config.inputs[6].range_values = opA
        case_config.inputs[7].range_values = opB
        case_config.inputs[8].range_values = orderA
        case_config.inputs[9].range_values = orderB
        case_config.inputs[10].range_values = ldA
        case_config.inputs[11].range_values = ldB
        case_config.inputs[12].range_values = dc['computeType']
        case_config.inputs[13].range_values = 0
        case_config.inputs[14].range_values = scenario

        case_config.inputs[15].shape = []
        case_config.inputs[15].dtype = dc['out']

        self.counter += 1
        return case_config
