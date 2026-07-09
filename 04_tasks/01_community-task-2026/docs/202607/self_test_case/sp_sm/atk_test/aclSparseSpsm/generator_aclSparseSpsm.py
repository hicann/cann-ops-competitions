import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig
from atk.configs.design_config import StandardConfig


MEM_LIMIT = 20 * 1024 * 1024 * 1024

NRHS_CHOICES = [1, 8, 32, 128]

SCENARIO_NORMAL = 0
SCENARIO_LOWER = 1            # TC-01 下三角
SCENARIO_UPPER_TRANS = 2      # TC-02 上三角 + transpose
SCENARIO_UPDATE_MATRIX = 3    # TC-04 updateMatrix 后 re-solve
SCENARIO_INPLACE = 4          # TC-07 in-place
SCENARIO_NULL_VALUES = 5      # TC-08 bufferSize/analysis NULL values
SCENARIO_M1 = 6               # 边界 m=1
SCENARIO_NEAR_SINGULAR = 7    # 近奇异对角
SCENARIO_FIXED_S1 = 8
SCENARIO_FIXED_S2 = 9
SCENARIO_FIXED_S3 = 10


def check_memory_limit(m, nrhs, avg_degree, elem_bytes=4):
    nnz = max(1, int(m * avg_degree))
    items = (m + 1) + nnz + nnz + m * nrhs * 2
    mem = items * elem_bytes
    if mem > MEM_LIMIT:
        m = max(int(m * 0.7), 1)
        m, nrhs, avg_degree, elem_bytes = check_memory_limit(m, nrhs, avg_degree, elem_bytes)
    return m, nrhs, avg_degree, elem_bytes


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
        return SCENARIO_M1
    if slot == 4:
        return SCENARIO_UPPER_TRANS
    if slot == 5:
        return SCENARIO_INPLACE
    if slot == 6:
        return SCENARIO_NULL_VALUES
    if slot == 7:
        return SCENARIO_UPDATE_MATRIX
    if slot < 12:
        return SCENARIO_NEAR_SINGULAR
    if slot < 22:
        return SCENARIO_LOWER
    return SCENARIO_NORMAL


def apply_scenario(scenario, m, avg_degree, nrhs, alpha):
    if scenario == SCENARIO_FIXED_S1:
        return 1000, 5.0, 1, alpha
    if scenario == SCENARIO_FIXED_S2:
        return 10000, 10.0, 8, alpha
    if scenario == SCENARIO_FIXED_S3:
        return 100000, 10.0, 1, alpha
    if scenario == SCENARIO_M1:
        return 1, random.uniform(5, 10), 1, alpha
    if scenario == SCENARIO_NEAR_SINGULAR:
        return random.randint(500, 5000), random.uniform(5, 15), random.choice(NRHS_CHOICES), alpha
    return m, avg_degree, nrhs, alpha


@GENERATOR_REGISTRY.register("aclSparseSpsm")
class AclSparseSpsmGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.counter = 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        scenario = pick_scenario(self.counter)

        m = random.randint(100, 20000)
        avg_degree = round(random.uniform(5, 50), 2)
        nrhs = random.choice(NRHS_CHOICES)
        alpha = sample_from_range(case_config.inputs[3].range_values)

        m, avg_degree, nrhs, alpha = apply_scenario(scenario, m, avg_degree, nrhs, alpha)
        m, nrhs, avg_degree, _ = check_memory_limit(m, nrhs, avg_degree)

        if scenario == SCENARIO_LOWER:
            fill_mode = 0  # lower
            op_a = 0
        elif scenario == SCENARIO_UPPER_TRANS:
            fill_mode = 1  # upper
            op_a = 1       # transpose
        else:
            fill_mode = random.choice([0, 1])
            op_a = random.choice([0, 1])

        diag_type = random.choice([0, 1])  # 0=non-unit, 1=unit
        op_b = random.choice([0, 1])

        in_place = 1 if scenario == SCENARIO_INPLACE else random.choice([0, 0, 0, 1])
        null_values = 1 if scenario == SCENARIO_NULL_VALUES else random.choice([0, 0, 0, 1])
        update_matrix = 1 if scenario == SCENARIO_UPDATE_MATRIX else random.choice([0, 0, 0, 1])

        if op_a == 1:
            b_rows, c_rows = m, m
        else:
            b_rows, c_rows = m, m

        case_config.inputs[0].shape = [b_rows, nrhs]
        case_config.inputs[0].dtype = 'fp32'

        case_config.inputs[1].shape = [c_rows, nrhs]
        case_config.inputs[1].dtype = 'fp32'

        case_config.inputs[2].range_values = avg_degree
        case_config.inputs[3].range_values = alpha
        case_config.inputs[4].range_values = op_a
        case_config.inputs[5].range_values = op_b
        case_config.inputs[6].range_values = fill_mode
        case_config.inputs[7].range_values = diag_type
        case_config.inputs[8].range_values = nrhs
        case_config.inputs[9].range_values = in_place
        case_config.inputs[10].range_values = null_values
        case_config.inputs[11].range_values = update_matrix
        case_config.inputs[12].range_values = 0
        case_config.inputs[13].range_values = 0
        case_config.inputs[14].range_values = scenario

        case_config.inputs[15].shape = []
        case_config.inputs[15].dtype = 'fp32'

        case_config.standard = StandardConfig(
            acc={'cv_fused_double_benchmark': {
                'max_re_ratio': 2, 'avg_re_ratio': 1.2, 'root_mean_squared_ratio': 1.2}}
        )

        self.counter += 1
        return case_config
