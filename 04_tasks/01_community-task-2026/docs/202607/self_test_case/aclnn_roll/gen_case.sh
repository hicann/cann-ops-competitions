#!/bin/bash

# 生成基础测试用例 (不含 complex64, 50个用例)
python ../scripts/gen_case.py -i op_basic.json -o case_basic.json -n 50 -c constraint_condition.py

# 生成 complex64 测试用例 (仅含 complex64, 100个用例)
python ../scripts/gen_case.py -i op_complex.json -o case_complex64_100.json -n 100 -c constraint_condition.py

# 生成完整测试用例 (含 complex64, 50个用例)
python ../scripts/gen_case.py -i op.json -o case.json -n 50 -c constraint_condition.py
