---
name: pytorch-inductor
description: Expert guidance for PyTorch Inductor compiler backend development and optimization. Covers FX graph lowering, decomposition→lowering pipeline, kernel fusion, Triton codegen, TritonTemplate, TritonTemplateKernel, template system, Jinja2 templates, C++ codegen, scheduling, memory planning, select_algorithm, autotuning, IR nodes (Buffer, Pointwise, Reduction), and performance optimization. Use for understanding Inductor architecture, lowering vs decomposition order, and performance optimization.
---

# PyTorch Inductor Expert

Expert guidance for working with PyTorch's Inductor compiler backend - the default compilation backend for `torch.compile`.

## Quick Start

**Working with Inductor?** Start here:
- Understanding architecture → See [ARCHITECTURE.md](ARCHITECTURE.md)
- Fusion decisions & dispatch → See [FUSION.md](FUSION.md)
- Guarding a fusion for profitability → See [FUSION-PROFITABILITY.md](FUSION-PROFITABILITY.md)
- Codegen internals → See [CODEGEN.md](CODEGEN.md)
- Memory planning → See [MEMORY-PLANNING.md](MEMORY-PLANNING.md)
- Common patterns → See [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
- Triton template system → See [TRITON-TEMPLATES.md](TRITON-TEMPLATES.md)

## What is Inductor?

Inductor is PyTorch's deep learning compiler that serves as the default backend for `torch.compile()`. It takes FX graphs from Dynamo and generates optimized machine code.

**Core pipeline**: FX Graph → Lowering → Scheduling → Fusion → Codegen (Triton/C++/CUDA)

### Design Principles

**PyTorch Native**: Uses similar abstractions to PyTorch eager mode to support nearly all PyTorch operations with a thin translation layer.

**Python First**: Pure Python compiler makes TorchInductor easy to understand and hackable by users.

**Breadth First**: Early focus on supporting wide variety of operators, hardware, and optimizations. A general purpose compiler that can scale.

### Breadth-First Capabilities

Inductor supports almost everything in a general way:
- **Aliasing/mutation/views** - Safety assured by preceding functionalization pass
- **Scatter/Gather** - Indirect writes/reads
- **Pooling/window operations** - Convolutions, pooling layers
- **Reductions** - Sum, max, mean, etc.
- **Masked execution** - Conditional operations

### Key Capabilities

- **Automatic kernel fusion** - Combines operations for better memory efficiency
- **Triton codegen** - Generates GPU kernels using Triton
- **C++ codegen** - CPU and fallback kernels with vectorization (AVX2/AVX512) and OpenMP
- **Memory planning** - Optimizes buffer allocation and reuse
- **Loop optimization** - Vectorization, unrolling, tiling
- **Layout tuning** - Channels-last, transposed layouts, padding for alignment
- **Auto-tuning** - Finds optimal kernel configurations

## When to Use This Skill

Activate when:
- Working with `torch/_inductor/` code
- Debugging Inductor compilation failures
- Optimizing generated kernel performance
- Adding support for new operators
- Working with Triton codegen
- Investigating fusion opportunities
- Memory optimization and planning
- Writing tests in `test/inductor/`

## Core Concepts (30-Second Version)

### Define-By-Run (DBR) Loop-Level IR
Inductor uses a unique IR where operations are defined as Python functions that can be analyzed and code-generated.

**Example**: `x.permute(1,0) + x[2, :]` becomes:
```python
def inner_fn(index: List[sympy.Expr]):
    i1, i0 = index
    tmp0 = ops.load("x", i1 + i0*size1)
    tmp1 = ops.load("x", 2*size1 + i0)
    tmp2 = ops.add(tmp0, tmp1)
    return tmp2

torchinductor.ir.Pointwise(
    device=torch.device(...),
    dtype=torch.float32,
    inner_fn=inner_fn,
    ranges=[size0, size1],
)
```

Override `ops` for analysis and backend codegen. This allows rapid lowering with minimal boilerplate.

### FX Graph Lowering
Inductor receives FX graphs from Dynamo and lowers them to **Node IR** — the define-by-run IR nodes (`Pointwise`, `Reduction`, etc.) with `inner_fn` and `ranges`. Heavy use of decompositions reduces the number of ops that need explicit lowering.

**File**: `torch/_inductor/graph.py`, `torch/_inductor/lowering.py`

### Dynamic Shapes & Strides
Uses **SymPy** extensively for reasoning about shapes, strides, and indexing:
- Views/slices handled by symbolic expressions
- Specializes on zero and one
- Specializes on sameness (e.g., `x + y` causes replacing y's sizes with x's)
- Strides expressed symbolically: `torch.ones(10, 10, 8)` → `shape=(s1, s1, s0), stride=(s1*s0, s0, 1)`
- Guards propagate globally

### Scheduling & Fusion
The scheduler wraps Node IR in **Schedule IR** (`SchedulerNode`, `FusedSchedulerNode`) — adding dependency tracking, `MemoryDep` access patterns, and fusion group keys. Fusion is a two-phase process — **legality** then **scoring** — that are architecturally distinct:

**Legality** (`can_fuse` in `SIMDScheduling`): A multi-branch decision tree. Not a simple "same numel" check — branches handle reduction pairs (including nested reduction dependent pairs), non-reduction pairs (with template exceptions), and reduction+pointwise epilogues.

**Scoring**: Ranks legal candidates by memory traffic saved.

**Fusion patterns**: vertical (producer-consumer), horizontal (consumer-consumer), reduction+epilogue (two-pass kernel), nested reduction (`FusedNestedReductions`).

**Files**: `scheduler.py`, `dependencies.py`, `codegen/simd.py`
**Deep dive**: [FUSION.md](FUSION.md)

### Codegen
Each `SchedulerNode`'s `inner_fn` is traced into **Loop-Level IR** (`LoopBody` — an FX graph of `ops.load`/`ops.store`/`ops.add` calls). Backend-specific ops handlers then translate Loop-Level IR into **Codegen IR** (target source strings).

**Backends**:
- **Triton** (`codegen/triton.py`) - GPU kernels, higher-level than CUDA
- **C++** (`codegen/cpp.py`) - CPU kernels with vectorization and OpenMP
- **CUDA** (`codegen/cuda/`) - Direct CUDA for specialized cases

**IR progression**: Node IR → Schedule IR → Loop-Level IR → Codegen IR → executable
**Deep dive**: [CODEGEN.md](CODEGEN.md), [ARCHITECTURE.md — IR Levels](ARCHITECTURE.md)

### Memory Planning
Determines buffer lifetimes and enables buffer reuse to minimize memory
footprint. Two modes: pool-based packing for inference, simple reuse for
training. Uses rematerialize-by-default strategy — trades compute for memory
by inlining buffers that don't need to be stored.

**Files**: `codegen/memory_planning.py`, `codegen/wrapper.py`, `memory.py`
**Deep dive**: [MEMORY-PLANNING.md](MEMORY-PLANNING.md)

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline diagram, IR hierarchy, and decomposition/lowering pipeline.

## Common Tasks

**Debug Compilation**: Enable `config.debug = True` and `config.trace.enabled = True`, check `/tmp/torchinductor_<user>/`. See the compile-trace skills.

**Add Operator**: Add lowering in `lowering.py` or decomposition in `decomposition.py`. See [COMMON-PATTERNS.md](COMMON-PATTERNS.md).

**Optimize Kernel**: Profile, check fusion opportunities, consider Triton templates..

**Custom Fusion**: Use `pattern_matcher.py` and define replacement. See [COMMON-PATTERNS.md](COMMON-PATTERNS.md).

## Codegen Backends

- **Triton** (`codegen/triton.py`) — GPU kernels via Triton. Generates `tl.load`/`tl.store`/compute. Templates for GEMMs/conv/attention.
- **C++** (`codegen/cpp.py`) — CPU kernels with AVX2/AVX512 vectorization and OpenMP.
- **CUDA** (`codegen/cuda/`) — Direct CUDA for specialized cases (CUTLASS templates).

See [CODEGEN.md](CODEGEN.md) for class hierarchies, ops handler pattern, and codegen mechanics.

## Configuration System

Inductor behavior is controlled via `torch._inductor.config`.

**Common settings**:
```python
import torch._inductor.config as config

# Enable/disable optimizations
config.triton.autotune = True        # Auto-tune Triton kernels
config.cpp.simdlen = 256             # SIMD vector width
config.fallback_random = False       # Disallow random fallbacks

# Debugging
config.debug = True                  # Enable debug output
config.trace.enabled = True          # Trace compilation
config.trace.graph_diagram = True    # Generate diagrams

# Codegen control
config.max_autotune = True           # Aggressive auto-tuning
config.coordinate_descent_tuning = True  # Advanced tuning
```

**File**: `torch/_inductor/config.py`

## Development Workflow

**Basic usage**: `torch.compile(fn, backend="inductor")` compiles and traces. Generated code in `/tmp/torchinductor_<user>/` (`.py` wrappers, `.cpp` kernels, `.cubin` binaries).

**Debugging**: Check `lowering.py` for decompositions. Enable `config.trace.enabled = True` for scheduler decisions.

## Performance Optimization

See [FUSION.md](FUSION.md) for fusion patterns (vertical, horizontal, reduction+epilogue, nested reduction).

**Auto-tuning**: `config.max_autotune = True` and `config.coordinate_descent_tuning = True` for aggressive kernel tuning.

**Memory layout**: Inductor optimizes layout (channels-last, transposed, padding) automatically.

## Testing

### Run Inductor Tests

```bash
# All inductor tests
pytest test/inductor/

# Specific test file
pytest test/inductor/test_torchinductor.py

# CPU-only tests
pytest test/inductor/test_cpu_repro.py

# GPU tests
pytest test/inductor/test_cuda_repro.py
```

### Write New Tests

Template:
```python
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA

class MyTest(TestCase):
    def test_my_op(self):
        def fn(x):
            return x.my_op()

        x = torch.randn(10, 10)
        compiled_fn = torch.compile(fn, backend="inductor")

        self.assertEqual(fn(x), compiled_fn(x))
```

## Progressive Disclosure

- **Getting started**: This file
- **Architecture deep-dive**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Fusion decisions & dispatch**: [FUSION.md](FUSION.md)
- **Guarding a fusion for profitability**: [FUSION-PROFITABILITY.md](FUSION-PROFITABILITY.md)
- **Codegen internals**: [CODEGEN.md](CODEGEN.md)
- **Memory planning**: [MEMORY-PLANNING.md](MEMORY-PLANNING.md)
- **Common patterns**: [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
- **Triton template system**: [TRITON-TEMPLATES.md](TRITON-TEMPLATES.md)

## Development Principles

1. **Preserve correctness** - Optimizations must not change semantics
2. **Respect memory constraints** - Don't bloat memory usage
3. **Fusion is key** - Most performance comes from fusion
4. **Profile before optimizing** - Measure, don't guess
5. **Test on real models** - Microbenchmarks can mislead

## Common Pitfalls

1. **Incorrect fusion assumptions** - Not all ops can be fused
2. **Memory aliasing bugs** - In-place ops require careful handling
3. **Layout mismatches** - Ensure consistent tensor layouts
4. **Precision issues** - Mixed precision can cause numerical errors
5. **Autotuning overhead** - Can be expensive for small models

Use the compile-trace skills for systematic debugging.

## Getting Help

**Compilation error?** → Use the compile-trace skills
**Performance issue?** → Profile first, check fusion and autotuning config
**Adding feature?** → [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
**Need quick command?** → See config and testing sections above
**Understanding internals?** → [ARCHITECTURE.md](ARCHITECTURE.md)

## Related Components

- **Dynamo** — Captures FX graphs fed to Inductor
- **AOTAutograd** — Handles autograd graph transformations, functionalization
- **Triton** — GPU kernel language used by Inductor
- **FX** — Graph representation format
