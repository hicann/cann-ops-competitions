import math
from dataclasses import dataclass

import torch

ELEM_THRESHOLD_8M = 8 * 1024 * 1024


@dataclass(frozen=True)
class FillsPreset:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    fill_value: float
    family: str
    note: str

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def element_bytes(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    @property
    def total_bytes(self) -> int:
        return self.numel * self.element_bytes


def _elems_for_bytes(total_bytes: int, element_bytes: int) -> int:
    if total_bytes % element_bytes != 0:
        raise ValueError(f"total_bytes={total_bytes} is not divisible by element_bytes={element_bytes}")
    return total_bytes // element_bytes


PRESETS: dict[str, FillsPreset] = {
    "public_case1_fp16": FillsPreset(
        name="public_case1_fp16",
        dtype=torch.float16,
        shape=(63,),
        fill_value=242.2,
        family="public_case",
        note="Official public case1 shape; useful as a correctness smoke test only.",
    ),
    "bf16_63_tail": FillsPreset(
        name="bf16_63_tail",
        dtype=torch.bfloat16,
        shape=(63,),
        fill_value=3.25,
        family="small_tail",
        note="Small BF16 case with a non-32B output tail.",
    ),
    "fp32_63_tail": FillsPreset(
        name="fp32_63_tail",
        dtype=torch.float32,
        shape=(63,),
        fill_value=3.25,
        family="small_tail",
        note="Small FP32 case with a non-32B output tail.",
    ),
    "i32_63_tail": FillsPreset(
        name="i32_63_tail",
        dtype=torch.int32,
        shape=(63,),
        fill_value=17.0,
        family="small_tail",
        note="Small INT32 case with a non-32B output tail.",
    ),
    "contest_case2_bf16_8m": FillsPreset(
        name="contest_case2_bf16_8m",
        dtype=torch.bfloat16,
        shape=(ELEM_THRESHOLD_8M,),
        fill_value=3.25,
        family="contest_fit",
        note="Cloud-fit candidate for hidden Case2: bf16 with totalLength >= 8M elements.",
    ),
    "contest_case3_u8_8m": FillsPreset(
        name="contest_case3_u8_8m",
        dtype=torch.uint8,
        shape=(ELEM_THRESHOLD_8M,),
        fill_value=241.0,
        family="contest_fit",
        note="Cloud-fit candidate for hidden Case3: byte path with totalLength >= 8M elements.",
    ),
    "contest_case3_i8_8m": FillsPreset(
        name="contest_case3_i8_8m",
        dtype=torch.int8,
        shape=(ELEM_THRESHOLD_8M,),
        fill_value=13.0,
        family="contest_fit_alt",
        note="Alternative byte-path candidate for hidden Case3 when the cloud case uses int8 instead of uint8.",
    ),
    "contest_case4_i32_4m": FillsPreset(
        name="contest_case4_i32_4m",
        dtype=torch.int32,
        shape=(4 * 1024 * 1024,),
        fill_value=17.0,
        family="contest_fit",
        note="Cloud-fit candidate for hidden Case4: int32 with totalLength < 8M elements.",
    ),
    "contest_case4_i32_6m": FillsPreset(
        name="contest_case4_i32_6m",
        dtype=torch.int32,
        shape=(6 * 1024 * 1024,),
        fill_value=17.0,
        family="contest_fit_alt",
        note="Larger int32 alternative for hidden Case4 while still staying below the 8M-element boundary.",
    ),
    "contest_case4_i32_128k": FillsPreset(
        name="contest_case4_i32_128k",
        dtype=torch.int32,
        shape=(128 * 1024,),
        fill_value=17.0,
        family="contest_fit_alt",
        note="Smaller int32 candidate for hidden Case4 to fit the cloud's low-teens timing band.",
    ),
    "contest_case4_i32_256k": FillsPreset(
        name="contest_case4_i32_256k",
        dtype=torch.int32,
        shape=(256 * 1024,),
        fill_value=17.0,
        family="contest_fit_alt",
        note="Mid-small int32 candidate for hidden Case4 while staying well below 8M elements.",
    ),
    "contest_case4_i32_512k": FillsPreset(
        name="contest_case4_i32_512k",
        dtype=torch.int32,
        shape=(512 * 1024,),
        fill_value=17.0,
        family="contest_fit_alt",
        note="Medium int32 candidate for hidden Case4 while staying below 8M elements.",
    ),
    "contest_case4_i32_1m": FillsPreset(
        name="contest_case4_i32_1m",
        dtype=torch.int32,
        shape=(1024 * 1024,),
        fill_value=17.0,
        family="contest_fit_alt",
        note="1M-element int32 candidate for hidden Case4 near the likely local timing knee.",
    ),
    "contest_case4_i32_2m": FillsPreset(
        name="contest_case4_i32_2m",
        dtype=torch.int32,
        shape=(2 * 1024 * 1024,),
        fill_value=17.0,
        family="contest_fit_alt",
        note="2M-element int32 candidate for hidden Case4 near the low-teens target band.",
    ),
    "contest_case4_i32_3m": FillsPreset(
        name="contest_case4_i32_3m",
        dtype=torch.int32,
        shape=(3 * 1024 * 1024,),
        fill_value=17.0,
        family="contest_fit_alt",
        note="3M-element int32 candidate for hidden Case4 before the 4M-element cliff seen locally.",
    ),
    "contest_case5_fp32_8m": FillsPreset(
        name="contest_case5_fp32_8m",
        dtype=torch.float32,
        shape=(ELEM_THRESHOLD_8M,),
        fill_value=3.25,
        family="contest_fit",
        note="Cloud-fit candidate for hidden Case5: fp32 with totalLength >= 8M elements.",
    ),
    "contest_case5_fp32_12m": FillsPreset(
        name="contest_case5_fp32_12m",
        dtype=torch.float32,
        shape=(12 * 1024 * 1024,),
        fill_value=3.25,
        family="contest_fit_alt",
        note="Larger fp32 candidate for hidden Case5 to approach the cloud's dominant heavy case timing.",
    ),
    "contest_case5_fp32_16m": FillsPreset(
        name="contest_case5_fp32_16m",
        dtype=torch.float32,
        shape=(16 * 1024 * 1024,),
        fill_value=3.25,
        family="contest_fit_alt",
        note="Even larger fp32 candidate for hidden Case5 when 8M elements is still too light locally.",
    ),
    "fp16_1mb_minus_512b": FillsPreset(
        name="fp16_1mb_minus_512b",
        dtype=torch.float16,
        shape=(_elems_for_bytes(1024 * 1024 - 512, 2),),
        fill_value=3.25,
        family="threshold_1mb",
        note="Just below the host 1MB total-byte split threshold with 512B granularity.",
    ),
    "fp16_1mb_plus_512b": FillsPreset(
        name="fp16_1mb_plus_512b",
        dtype=torch.float16,
        shape=(_elems_for_bytes(1024 * 1024 + 512, 2),),
        fill_value=3.25,
        family="threshold_1mb",
        note="Just above the host 1MB total-byte split threshold with 512B granularity.",
    ),
    "fp16_4mb_minus_512b": FillsPreset(
        name="fp16_4mb_minus_512b",
        dtype=torch.float16,
        shape=(_elems_for_bytes(4 * 1024 * 1024 - 512, 2),),
        fill_value=3.25,
        family="threshold_4mb",
        note="Just below the host 4MB full-core threshold with 512B granularity.",
    ),
    "fp16_4mb_plus_512b": FillsPreset(
        name="fp16_4mb_plus_512b",
        dtype=torch.float16,
        shape=(_elems_for_bytes(4 * 1024 * 1024 + 512, 2),),
        fill_value=3.25,
        family="threshold_4mb",
        note="Just above the host 4MB full-core threshold with 512B granularity.",
    ),
    "fp16_16mb": FillsPreset(
        name="fp16_16mb",
        dtype=torch.float16,
        shape=(_elems_for_bytes(16 * 1024 * 1024, 2),),
        fill_value=3.25,
        family="large_bandwidth",
        note="Large steady-state writeout case to stress saturated-core bandwidth behavior.",
    ),
    "fp32_1mb_minus_512b": FillsPreset(
        name="fp32_1mb_minus_512b",
        dtype=torch.float32,
        shape=(_elems_for_bytes(1024 * 1024 - 512, 4),),
        fill_value=3.25,
        family="threshold_1mb",
        note="Just below the host 1MB total-byte split threshold with 512B granularity.",
    ),
    "fp32_1mb_plus_512b": FillsPreset(
        name="fp32_1mb_plus_512b",
        dtype=torch.float32,
        shape=(_elems_for_bytes(1024 * 1024 + 512, 4),),
        fill_value=3.25,
        family="threshold_1mb",
        note="Just above the host 1MB total-byte split threshold with 512B granularity.",
    ),
    "fp32_4mb_minus_512b": FillsPreset(
        name="fp32_4mb_minus_512b",
        dtype=torch.float32,
        shape=(_elems_for_bytes(4 * 1024 * 1024 - 512, 4),),
        fill_value=3.25,
        family="threshold_4mb",
        note="Just below the host 4MB full-core threshold with 512B granularity.",
    ),
    "fp32_4mb_plus_512b": FillsPreset(
        name="fp32_4mb_plus_512b",
        dtype=torch.float32,
        shape=(_elems_for_bytes(4 * 1024 * 1024 + 512, 4),),
        fill_value=3.25,
        family="threshold_4mb",
        note="Just above the host 4MB full-core threshold with 512B granularity.",
    ),
    "fp32_16mb": FillsPreset(
        name="fp32_16mb",
        dtype=torch.float32,
        shape=(_elems_for_bytes(16 * 1024 * 1024, 4),),
        fill_value=3.25,
        family="large_bandwidth",
        note="Large steady-state writeout case to stress saturated-core bandwidth behavior.",
    ),
    "bf16_1mb_minus_512b": FillsPreset(
        name="bf16_1mb_minus_512b",
        dtype=torch.bfloat16,
        shape=(_elems_for_bytes(1024 * 1024 - 512, 2),),
        fill_value=3.25,
        family="threshold_1mb",
        note="BF16 case just below the host 1MB total-byte split threshold.",
    ),
    "bf16_1mb_plus_512b": FillsPreset(
        name="bf16_1mb_plus_512b",
        dtype=torch.bfloat16,
        shape=(_elems_for_bytes(1024 * 1024 + 512, 2),),
        fill_value=3.25,
        family="threshold_1mb",
        note="BF16 case just above the host 1MB total-byte split threshold.",
    ),
    "bf16_4mb_minus_512b": FillsPreset(
        name="bf16_4mb_minus_512b",
        dtype=torch.bfloat16,
        shape=(_elems_for_bytes(4 * 1024 * 1024 - 512, 2),),
        fill_value=3.25,
        family="threshold_4mb",
        note="BF16 case just below the host 4MB full-core threshold.",
    ),
    "bf16_4mb_plus_512b": FillsPreset(
        name="bf16_4mb_plus_512b",
        dtype=torch.bfloat16,
        shape=(_elems_for_bytes(4 * 1024 * 1024 + 512, 2),),
        fill_value=3.25,
        family="threshold_4mb",
        note="BF16 case just above the host 4MB full-core threshold.",
    ),
    "bf16_16mb": FillsPreset(
        name="bf16_16mb",
        dtype=torch.bfloat16,
        shape=(_elems_for_bytes(16 * 1024 * 1024, 2),),
        fill_value=3.25,
        family="large_bandwidth",
        note="Large BF16 writeout case to stress saturated-core bandwidth behavior.",
    ),
    "i16_63_tail": FillsPreset(
        name="i16_63_tail",
        dtype=torch.int16,
        shape=(63,),
        fill_value=17.0,
        family="small_tail",
        note="Small INT16 case with a non-32B output tail.",
    ),
    "i16_1mb_minus_512b": FillsPreset(
        name="i16_1mb_minus_512b",
        dtype=torch.int16,
        shape=(_elems_for_bytes(1024 * 1024 - 512, 2),),
        fill_value=17.0,
        family="threshold_1mb",
        note="INT16 case just below the host 1MB total-byte split threshold.",
    ),
    "i16_1mb_plus_512b": FillsPreset(
        name="i16_1mb_plus_512b",
        dtype=torch.int16,
        shape=(_elems_for_bytes(1024 * 1024 + 512, 2),),
        fill_value=17.0,
        family="threshold_1mb",
        note="INT16 case just above the host 1MB total-byte split threshold.",
    ),
    "i16_4mb_minus_512b": FillsPreset(
        name="i16_4mb_minus_512b",
        dtype=torch.int16,
        shape=(_elems_for_bytes(4 * 1024 * 1024 - 512, 2),),
        fill_value=17.0,
        family="threshold_4mb",
        note="INT16 case just below the host 4MB full-core threshold.",
    ),
    "i16_4mb_plus_512b": FillsPreset(
        name="i16_4mb_plus_512b",
        dtype=torch.int16,
        shape=(_elems_for_bytes(4 * 1024 * 1024 + 512, 2),),
        fill_value=17.0,
        family="threshold_4mb",
        note="INT16 case just above the host 4MB full-core threshold.",
    ),
    "i16_16mb": FillsPreset(
        name="i16_16mb",
        dtype=torch.int16,
        shape=(_elems_for_bytes(16 * 1024 * 1024, 2),),
        fill_value=17.0,
        family="large_bandwidth",
        note="Large INT16 writeout case to stress saturated-core bandwidth behavior.",
    ),
    "u8_1mb_minus_512b": FillsPreset(
        name="u8_1mb_minus_512b",
        dtype=torch.uint8,
        shape=(_elems_for_bytes(1024 * 1024 - 512, 1),),
        fill_value=241.0,
        family="threshold_1mb",
        note="U8 case just below the host 1MB total-byte split threshold.",
    ),
    "u8_1mb_plus_512b": FillsPreset(
        name="u8_1mb_plus_512b",
        dtype=torch.uint8,
        shape=(_elems_for_bytes(1024 * 1024 + 512, 1),),
        fill_value=241.0,
        family="threshold_1mb",
        note="U8 case just above the host 1MB total-byte split threshold.",
    ),
    "u8_4mb_minus_512b": FillsPreset(
        name="u8_4mb_minus_512b",
        dtype=torch.uint8,
        shape=(_elems_for_bytes(4 * 1024 * 1024 - 512, 1),),
        fill_value=241.0,
        family="threshold_4mb",
        note="U8 case just below the host 4MB full-core threshold.",
    ),
    "u8_4mb_plus_512b": FillsPreset(
        name="u8_4mb_plus_512b",
        dtype=torch.uint8,
        shape=(_elems_for_bytes(4 * 1024 * 1024 + 512, 1),),
        fill_value=241.0,
        family="threshold_4mb",
        note="U8 case just above the host 4MB full-core threshold.",
    ),
    "u8_16mb": FillsPreset(
        name="u8_16mb",
        dtype=torch.uint8,
        shape=(_elems_for_bytes(16 * 1024 * 1024, 1),),
        fill_value=241.0,
        family="large_bandwidth",
        note="Large U8 writeout case to stress saturated-core bandwidth behavior.",
    ),
    "u8_63_tail": FillsPreset(
        name="u8_63_tail",
        dtype=torch.uint8,
        shape=(63,),
        fill_value=241.0,
        family="small_tail",
        note="Small U8 case with a non-32B output tail.",
    ),
    "i8_63_tail": FillsPreset(
        name="i8_63_tail",
        dtype=torch.int8,
        shape=(63,),
        fill_value=13.0,
        family="small_tail",
        note="Small INT8 case with a non-32B output tail and odd half packing.",
    ),
    "i8_1mb_plus_512b": FillsPreset(
        name="i8_1mb_plus_512b",
        dtype=torch.int8,
        shape=(_elems_for_bytes(1024 * 1024 + 512, 1),),
        fill_value=13.0,
        family="threshold_1mb",
        note="INT8 case just above the host 1MB total-byte split threshold.",
    ),
    "i8_4mb_plus_512b": FillsPreset(
        name="i8_4mb_plus_512b",
        dtype=torch.int8,
        shape=(_elems_for_bytes(4 * 1024 * 1024 + 512, 1),),
        fill_value=13.0,
        family="threshold_4mb",
        note="INT8 case just above the host 4MB full-core threshold.",
    ),
    "i8_8m": FillsPreset(
        name="i8_8m",
        dtype=torch.int8,
        shape=(ELEM_THRESHOLD_8M,),
        fill_value=13.0,
        family="contest_fit_alt",
        note="INT8 candidate at the hidden Case2/3 8M-element boundary.",
    ),
}


DEFAULT_BENCH_CASES = [
    "fp16_1mb_minus_512b",
    "fp16_1mb_plus_512b",
    "fp16_4mb_plus_512b",
    "fp32_1mb_minus_512b",
    "fp32_1mb_plus_512b",
    "fp32_4mb_plus_512b",
    "bf16_1mb_plus_512b",
    "bf16_4mb_plus_512b",
    "i16_1mb_plus_512b",
    "i16_4mb_plus_512b",
]

CONTEST_FIT_PRIMARY_CASES = [
    "public_case1_fp16",
    "contest_case2_bf16_8m",
    "contest_case3_u8_8m",
    "contest_case4_i32_4m",
    "contest_case5_fp32_8m",
]


def list_presets() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> FillsPreset:
    return PRESETS[name]


def build_input(name: str) -> torch.Tensor:
    preset = get_preset(name)
    return torch.zeros(preset.shape, dtype=preset.dtype)


def build_golden(name: str, input_cpu: torch.Tensor) -> torch.Tensor:
    preset = get_preset(name)
    return torch.full_like(input_cpu, preset.fill_value)
