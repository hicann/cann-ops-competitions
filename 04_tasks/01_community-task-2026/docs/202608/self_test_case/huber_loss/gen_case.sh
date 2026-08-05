#!/bin/bash

# 生成 Huber Loss 测试用例 (50个用例)
python ../scripts/gen_case.py -i op.json -o case.json -n 100 -c constraint_condition.py