import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig
from atk.configs.design_config import StandardConfig


MEM_LIMIT = 20 * 1024 * 1024 * 1024

# SpGEMM: C = alpha * A * B + beta * C, 全 CSR; opA/opB 仅 NON_TRANSPOSE
# 覆盖 U1 fp32 / U2 fp16 / U3 bf16
DTYPE_CASES = {
    0: {'out': 'fp32', 'computeType': 0},
    1: {'out': 'fp16', 'computeType': 1},
    2: {'out': 'bf16', 'computeType': 2},
}

SCENARIO_NORMAL = 0
SCENARIO_ALPHA_BETA = 1       # TC-03 alpha/beta != 1
SCENARIO_ZERO_NNZ = 2         # TC-02 零 nnz / 空矩阵
SCENARIO_NNZ_INFLATE = 3      # 输出 nnz 膨胀显著
SCENARIO_HIGH_SPARSITY = 4    # TC-04 高稀疏度大图
SCENARIO_FIXED_S1 = 5
SCENARIO_FIXED_S2 = 6
SCENARIO_FIXED_S3 = 7


def check_memory_limit(m, k, n, sparsity_a, sparsity_b, elem_bytes=4):
    nnz_a = max(1, int(m * k * (1 - sparsity_a)))
    nnz_b = max(1, int(k * n * (1 - sparsity_b)))
    items = (m + 1) + nnz_a + nnz_a + (k + 1) + nnz_b + nnz_b + m * n
    mem = items * elem_bytes
    if mem > MEM_LIMIT:
        m = max(int(m * 0.7), 128)
        k = max(int(k * 0.7), 128)
        n = max(int(n * 0.7), 128)
        m, k, n, sparsity_a, sparsity_b, _ = check_memory_limit(
            m, k, n, sparsity_a, sparsity_b, elem_bytes)
    return m, k, n, sparsity_a, sparsity_b, elem_bytes


def sample_from_range(range_values):
    if isinstance(range_values, (list, tuple)) and len(range_values) == 2:
        lo, hi = float(range_values[0]), float(range_values[1])
        return round(random.uniform(lo, hi), 4)
    return range_values


def pick_scenario(counter):
    slot = counter % 100
    if slot < 3:
        return SCENARIO_FIXED_S1 + slot
    if slot == 3:
        return SCENARIO_ZERO_NNZ
    if slot == 4:
        return SCENARIO_NNZ_INFLATE
    if slot == 5:
        return SCENARIO_HIGH_SPARSITY
    if slot < 17:
        return SCENARIO_ALPHA_BETA
    return SCENARIO_NORMAL


def apply_scenario(scenario, m, k, n, sp_a, sp_b, alpha, beta):
    if scenario == SCENARIO_FIXED_S1:
        return 128, 128, 128, 0.9023, 0.9023, 1.0, 1.0
    if scenario == SCENARIO_FIXED_S2:
        return 1024, 1024, 1024, 0.9904, 0.9904, 1.0, 1.0
    if scenario == SCENARIO_FIXED_S3:
        return 10000, 10000, 10000, 0.999, 0.999, 1.0, 1.0
    if scenario == SCENARIO_ZERO_NNZ:
        return 128, 128, 128, 1.0, 1.0, 1.0, 0.0
    if scenario == SCENARIO_NNZ_INFLATE:
        return m, k, n, round(random.uniform(0.85, 0.92), 4), round(random.uniform(0.85, 0.92), 4), alpha, beta
    if scenario == SCENARIO_HIGH_SPARSITY:
        return random.randint(5000, 10000), random.randint(5000, 10000), random.randint(5000, 10000), \
            round(random.uniform(0.9995, 0.99999), 5), round(random.uniform(0.9995, 0.99999), 5), alpha, beta
    if scenario == SCENARIO_ALPHA_BETA:
        if alpha == 1.0 and beta == 1.0:
            alpha = round(random.choice([-3.0, -1.5, 0.5, 2.0, 3.0]), 4)
        if beta == 1.0:
            beta = round(random.choice([-2.0, 0.0, 0.5, 2.0]), 4)
        return m, k, n, sp_a, sp_b, alpha, beta
    return m, k, n, sp_a, sp_b, alpha, beta


@GENERATOR_REGISTRY.register("aclSparseSpgemm")
class AclSparseSpgemmGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.counter = 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        dc = DTYPE_CASES[self.counter % len(DTYPE_CASES)]
        scenario = pick_scenario(self.counter)

        m = random.randint(128, 5000)
        k = random.randint(128, 5000)
        n = random.randint(128, 5000)
        sp_a = round(random.uniform(0.99, 0.9999), 4)
        sp_b = round(random.uniform(0.99, 0.9999), 4)
        alpha = sample_from_range(case_config.inputs[2].range_values)
        beta = sample_from_range(case_config.inputs[3].range_values)

        m, k, n, sp_a, sp_b, alpha, beta = apply_scenario(
            scenario, m, k, n, sp_a, sp_b, alpha, beta)

        elem_bytes = 4 if dc['out'] == 'fp32' else 2
        m, k, n, sp_a, sp_b, _ = check_memory_limit(m, k, n, sp_a, sp_b, elem_bytes)

        alg = 0

        case_config.inputs[0].range_values = sp_a
        case_config.inputs[1].range_values = sp_b
        case_config.inputs[2].range_values = alpha
        case_config.inputs[3].range_values = beta
        case_config.inputs[4].range_values = 0  # opA NON_TRANSPOSE
        case_config.inputs[5].range_values = 0  # opB NON_TRANSPOSE
        case_config.inputs[6].range_values = m
        case_config.inputs[7].range_values = k
        case_config.inputs[8].range_values = n
        case_config.inputs[9].range_values = dc['computeType']
        case_config.inputs[10].range_values = alg
        case_config.inputs[11].range_values = scenario

        case_config.inputs[12].shape = []
        case_config.inputs[12].dtype = dc['out']

        self.counter += 1
        return case_config
