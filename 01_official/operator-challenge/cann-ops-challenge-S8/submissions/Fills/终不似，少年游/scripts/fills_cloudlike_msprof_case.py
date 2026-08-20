import argparse

import torch
import torch_npu

import custom_ops_lib
from fills_cloudlike_cases import build_golden, build_input, get_preset, list_presets


torch.npu.config.allow_internal_format = False


def verify_once(name: str) -> None:
    preset = get_preset(name)
    input_cpu = build_input(name)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one cloudlike Fills case for msprof app-command profiling.")
    parser.add_argument("preset", choices=list_presets())
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    preset = get_preset(args.preset)
    input_npu = build_input(args.preset).npu()

    print(
        f"preset={args.preset},dtype={preset.dtype},shape={preset.shape},"
        f"numel={preset.numel},total_bytes={preset.total_bytes},"
        f"family={preset.family},fill_value={preset.fill_value},repeat={args.repeat},case_side_round=50"
    )

    if not args.skip_verify:
        verify_once(args.preset)

    for _ in range(args.repeat):
        custom_ops_lib.custom_op(input_npu, preset.fill_value)
    torch.npu.synchronize()
    print("test pass")
    print("done")


if __name__ == "__main__":
    main()
