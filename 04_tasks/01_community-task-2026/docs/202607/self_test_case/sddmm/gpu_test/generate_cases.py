#!/usr/bin/env python3
"""Generate gpu_test case JSON files (200 cases per operator).

Usage:
  python3 generate_cases.py              # write cases/*.json (200 each)
  python3 generate_cases.py --count 50   # override count
  python3 generate_cases.py --op sddmm   # single operator
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'lib'))

from case_generator import generate_cases  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Generate gpu_test case JSON files')
    parser.add_argument('--op', choices=['sddmm', 'spgemm', 'spsm', 'all'], default='all')
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default=os.path.join(ROOT, 'cases'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ops = ['sddmm', 'spgemm', 'spsm'] if args.op == 'all' else [args.op]

    for op in ops:
        cases = generate_cases(op, total=args.count, random_seed=args.seed)
        out_path = os.path.join(args.out_dir, f'{op}_cases.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2)
            f.write('\n')
        fixed = sum(1 for c in cases if c.get('category') == 'fixed')
        func = sum(1 for c in cases if c.get('category') == 'functional')
        rand = sum(1 for c in cases if c.get('category') == 'random')
        perf = sum(1 for c in cases if c.get('perf_only'))
        print(f'{op}: wrote {len(cases)} cases -> {out_path}')
        print(f'  fixed={fixed} functional={func} random={rand} perf_only={perf}')

    print('Done.')


if __name__ == '__main__':
    main()
