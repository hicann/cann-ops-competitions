# ConfusionMatrix Ascend C baseline

This directory contains a self-contained Ascend C implementation for the July 2026 CANN operator ladder `confusion_matrix` task.

## Contents

- `op_host/`: operator definition, shape inference, and tiling implementation.
- `op_kernel/`: device kernel and generated tiling headers.
- `examples/`: ACLNN invocation example.
- `tests/ut/`: host and kernel unit tests.
- `build.sh` and `CMakeLists.txt`: build entry points.

## Build and verify

Run in a Linux environment with the CANN toolkit installed and initialized (for example, after sourcing the toolkit `set_env.sh`):

```bash
bash build.sh
bash build.sh -u
```

To build and run the ACLNN example on an available NPU:

```bash
bash build.sh -e
```

The default build target is Ascend 910B. The implementation is shape-generic within the task constraints and does not branch on hidden testcase identifiers.
