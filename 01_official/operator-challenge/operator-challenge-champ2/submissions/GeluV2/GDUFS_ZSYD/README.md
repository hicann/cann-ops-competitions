# GeluV2 — GDUFS_ZSYD

GDUFS_ZSYD submission for the GeluV2 operator challenge.

## Build

In a CANN environment with the toolkit initialized, run from this directory:

```bash
bash build.sh
```

Generated build directories and install packages are intentionally excluded from this submission.

## Source layout

- `op_host/`: operator registration, tiling, and shape inference.
- `op_kernel/`: Ascend C Kernel implementation.
- `cmake/`, `framework/`, and the top-level CMake files: build support.
