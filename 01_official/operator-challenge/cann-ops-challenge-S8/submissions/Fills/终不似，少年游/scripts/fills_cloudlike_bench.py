import argparse
import statistics
import time

import torch
import torch_npu

import custom_ops_lib
from fills_cloudlike_cases import DEFAULT_BENCH_CASES, build_golden, build_input, get_preset, list_presets


torch.npu.config.allow_internal_format = False


def verify_once(name: str, input_cpu: torch.Tensor) -> None:
    preset = get_preset(name)
    output_npu = custom_ops_lib.custom_op(input_cpu.npu(), preset.fill_value)
    torch.npu.synchronize()
    output_cpu = output_npu.cpu()
    golden = build_golden(name, input_cpu)

    if preset.dtype in (torch.float16, torch.float32, torch.bfloat16):
        if preset.dtype == torch.float32:
            atol = 1e-4
            rtol = 1e-4
        else:
            atol = 1e-3
            rtol = 1e-3
        ok = torch.isclose(output_cpu, golden, atol=atol, rtol=rtol, equal_nan=True).all().item()
    else:
        ok = torch.equal(output_cpu, golden)

    if not ok:
        diff = (output_cpu.to(torch.float32) - golden.to(torch.float32)).abs().max().item()
        raise RuntimeError(f"{name} verify failed, max_abs_diff={diff}")


def bench_case(name: str, warmup: int, iters: int, rounds: int) -> None:
    preset = get_preset(name)
    input_npu = build_input(name).npu()

    for _ in range(warmup):
        custom_ops_lib.custom_op(input_npu, preset.fill_value)
    torch.npu.synchronize()

    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iters):
            custom_ops_lib.custom_op(input_npu, preset.fill_value)
        torch.npu.synchronize()
        end = time.perf_counter()
        samples.append((end - start) * 1e6 / iters)

    print(
        "RESULT "
        f"case={name} "
        f"dtype={preset.dtype} "
        f"shape={preset.shape} "
        f"numel={preset.numel} "
        f"total_bytes={preset.total_bytes} "
        f"family={preset.family} "
        f"fill_value={preset.fill_value} "
        f"avg_us={statistics.mean(samples):.3f} "
        f"med_us={statistics.median(samples):.3f} "
        f"best_us={min(samples):.3f} "
        f"worst_us={max(samples):.3f} "
        "case_side_round=50"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom wall-time probe runner for Fills cloudlike cases.")
    parser.add_argument("--version", default="baseline")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_BENCH_CASES, choices=list_presets())
    args = parser.parse_args()

    print(
        f"version={args.version} warmup={args.warmup} iters={args.iters} "
        f"rounds={args.rounds} case_count={len(args.cases)}"
    )

    for case_name in args.cases:
        preset = get_preset(case_name)
        print(
            f"CASE case={case_name} dtype={preset.dtype} shape={preset.shape} "
            f"numel={preset.numel} total_bytes={preset.total_bytes} "
            f"family={preset.family} note={preset.note}"
        )
        if not args.skip_verify:
            verify_once(case_name, build_input(case_name))
        bench_case(case_name, args.warmup, args.iters, args.rounds)


if __name__ == "__main__":
    main()

