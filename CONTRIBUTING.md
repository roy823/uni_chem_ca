# Contributing

Read `docs/ARCHITECTURE.zh-CN.md`, `docs/PHYSICS_SCOPE.md`, and `docs/EXTENDING.md` first.
Do not turn atomic extent into one voxel, hard-code chemical bonds into the engine,
or mutate another cell's just-updated state inside a synchronous phase.

Use atomic units internally, typed public APIs, explicit exceptions and local RNGs.
Keep immutable-by-contract tensor snapshots and shape/device/dtype checks.
A frozen dataclass is not a read-only Tensor; never rely on it for memory protection.

Run `python -m pytest -q`. Add a regression test for every numerical or scientific fix.
Use `ruff format src tests examples` and `ruff check src tests examples` when the
optional development tool is installed. Formatting/lint tools are not runtime dependencies.

A new physical model must name its reference data, approximations, charge convention,
conservation properties, and domain of validity. Do not hide failed SCF convergence.
A faster backend must be compared against the reference backend on the same inputs.

This repository contains a CPU correctness baseline. CUDA/distributed execution,
scientific accuracy, and performance must be validated separately rather than inferred
from device-compatible tensors or the presence of a protocol class.
