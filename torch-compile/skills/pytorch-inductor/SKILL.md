---
name: pytorch-inductor
description: Expert guidance for PyTorch Inductor compiler backend development and optimization. Covers FX graph lowering, decomposition→lowering pipeline, kernel fusion, Triton codegen, TritonTemplate, TritonTemplateKernel, template system, Jinja2 templates, C++ codegen, scheduling, memory planning, select_algorithm, autotuning, IR nodes (Buffer, Pointwise, Reduction), and performance optimization. Use for understanding Inductor architecture, lowering vs decomposition order, and performance optimization.
---

# PyTorch Inductor Expert

Expert guidance for working with PyTorch's Inductor compiler backend - the default compilation backend for `torch.compile`.

## Quick Start

**Working with Inductor?** Start here:
- Understanding architecture → See [ARCHITECTURE.md](ARCHITECTURE.md)
- Fusion decisions → See [FUSION.md](FUSION.md)
- Codegen internals → See [CODEGEN.md](CODEGEN.md)
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
Inductor receives FX graphs from Dynamo and lowers them to DBR IR nodes. Heavy use of decompositions reduces the number of ops that need explicit lowering.

**File**: `torch/_inductor/graph.py`, `torch/_inductor/lowering.py`

### Dynamic Shapes & Strides
Uses **SymPy** extensively for reasoning about shapes, strides, and indexing:
- Views/slices handled by symbolic expressions
- Specializes on zero and one
- Specializes on sameness (e.g., `x + y` causes replacing y's sizes with x's)
- Strides expressed symbolically: `torch.ones(10, 10, 8)` → `shape=(s1, s1, s0), stride=(s1*s0, s0, 1)`
- Guards propagate globally

### Scheduling & Fusion
Fusion is a two-phase process — **legality** then **scoring** — that are architecturally distinct:

**Legality** (`can_fuse` in `SIMDScheduling`): A multi-branch decision tree checking whether two nodes can legally share a kernel. Branches handle: reduction+reduction (including nested reduction dependent pairs), non-reduction pairs (numel/rnumel match with template exceptions), reduction+pointwise epilogues (args swapped and re-checked), and more. Not a simple "same numel" check.

**Scoring**: Ranks legal candidates by memory traffic saved (shared inputs eliminated, graph distance).

**Fusion patterns**: vertical (producer-consumer), horizontal (consumer-consumer), reduction+epilogue (two-pass kernel), nested reduction (two dependent reductions at different granularity via `FusedNestedReductions`).

All fusion analysis examines `MemoryDep` objects — sympy expressions encoding each node's access patterns and iteration domains.

**Files**: `scheduler.py`, `dependencies.py`, `codegen/simd.py`
**Deep dive**: [FUSION.md](FUSION.md)

### Codegen
IR nodes are converted to executable code via backends.

**Backends**:
- **Triton** (`codegen/triton.py`) - GPU kernels, higher-level than CUDA
- **C++** (`codegen/cpp.py`) - CPU kernels with vectorization and OpenMP
- **CUDA** (`codegen/cuda/`) - Direct CUDA for specialized cases

### Memory Planning
Determines buffer lifetimes and enables buffer reuse to minimize memory footprint. Uses rematerialize-by-default strategy.

**File**: `torch/_inductor/memory_planning.py`

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│  AOT Autograd / PrimTorch                                      │
│  - Captures forward + backwards                                │
│  - Decompose into smaller operator set                         │
│  - Backends can choose which decompositions to use             │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  Inductor Graph Lowerings (graph.py, lowering.py)              │
│  - Performs implicit broadcasting                              │
│  - Collapses dimensions                                        │
│  - Simplifies indexing                                         │
│  - Creates IR nodes (Buffer, Pointwise, Reduction, etc.)       │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  Inductor Scheduling (scheduler.py)                            │
│  - Horizontal / vertical fusion decisions                      │
│  - Reduction fusions                                           │
│  - Rematerialize vs reuse decisions                            │
│  - Tiling, layout tuning, loop order                           │
│  - Memory planning and buffer reuse                            │
│  - In-place memory buffers                                     │
│  - Autotuning / kernel selection                               │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  Wrapper Codegen (codegen/wrapper.py)                          │
│  - Outer code that calls kernels                               │
│  - Allocates memory using torch APIs                           │
│  - Replaces Python interpreter                                 │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  Backend Codegen                                               │
│  ├─ Triton (codegen/triton.py) → GPU kernels                  │
│  ├─ C++/OpenMP (codegen/cpp.py) → CPU kernels                 │
│  └─ CUDA (codegen/cuda/) → Specialized CUDA kernels           │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│              Compiled Module (ready to execute)                │
└────────────────────────────────────────────────────────────────┘
```

## Decomposition and Lowering Pipeline

Decompositions run in AOT stage, then Inductor lowerings process the post-decomposition graph.

See [ARCHITECTURE.md](ARCHITECTURE.md#decomposition-and-lowering-pipeline) for complete details on pipeline order and preprocessing patterns.

## Key Files Quick Map

**Core**: `graph.py` (lowering), `scheduler.py` (fusion), `dependencies.py` (`MemoryDep`), `memory_planning.py`, `ir.py`
**Fusion**: `scheduler.py` (`NestedReduction`, `FusedNestedReductions`), `codegen/simd.py` (`can_fuse`)
**Lowering**: `lowering.py`, `decomposition.py`, `virtualized.py`, `pattern_matcher.py`
**Templates**: `kernel/mm.py` (matmul), `kernel/conv.py`, `select_algorithm.py`
**Codegen**: `codegen/simd.py` (base), `codegen/triton.py`, `codegen/cpp.py`, `codegen/wrapper.py`, `codegen/common.py` (CSE)

## Common Tasks

**Debug Compilation**: Enable `config.debug = True` and `config.trace.enabled = True`, check `/tmp/torchinductor_<user>/`. See the compile-trace skills.

**Add Operator**: Add lowering in `lowering.py` or decomposition in `decomposition.py`. See [COMMON-PATTERNS.md](COMMON-PATTERNS.md).

**Optimize Kernel**: Profile, check fusion opportunities, consider Triton templates..

**Custom Fusion**: Use `pattern_matcher.py` and define replacement. See [COMMON-PATTERNS.md](COMMON-PATTERNS.md).

## IR Node Types

**Main types**: Buffer, Pointwise, Reduction, Scan, ComputedBuffer, TemplateBuffer (matmul/conv), ExternKernelOut, InputBuffer, View. See `torch/_inductor/ir.py`.

## Codegen Backends

### Triton Backend

Generates GPU kernels using Triton - a programming language for highly performant GPU kernels.

**What is Triton**:
- Higher level than CUDA, lower level than DSLs
- Allows non-experts to write fast custom kernels
- Users define tensors (blocks of data) in SRAM and modify them using torch-like operators
- Developed by Philippe Tillet at OpenAI: https://triton-lang.org

**Inductor's Triton Usage**:
- Generates straightforward and understandable Triton functions
- Triton compiler produces performant PTX code (then translated with ptxas)
- Compilation done in parallel with fast C++ launchers
- Compiled binaries cached on disk for fast warmup on reruns

**Triton Features**:
- Vectorized loads/stores
- Tiling for differently strided inputs
- Performant reductions using warp intrinsics or shared memory
- Kernels are templated for autotuning or use heuristics for good parameters

**Template Support**:
- Operations like GEMMs, convolutions, multi-head attention use carefully tuned Triton templates
- Templates participate in regular fusions
- Inductor can't generate these from first principles

**File**: `torch/_inductor/codegen/triton.py`

### C++ Backend

Generates CPU kernels and fallback implementations. Built in collaboration with Intel PyTorch team.

**Features**:
- Vectorization (AVX2, AVX512)
- OpenMP parallelization
- Specialized CPU optimizations
- Promising early results (outperforming IPEX on Huggingface benchmarks)
- Ensures TorchInductor is not overly specialized to GPUs

**File**: `torch/_inductor/codegen/cpp.py`

### CUDA Backend

Direct CUDA code generation for specialized cases.

**Use cases**:
- Operations not suitable for Triton
- Legacy templates
- Vendor library integration

**Directory**: `torch/_inductor/codegen/cuda/`

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

### Fusion Opportunities

**Pointwise fusion**: Chain element-wise operations
```python
# Before: 3 kernels
x.relu().add(1).mul(2)

# After fusion: 1 kernel
# fused_kernel(x): return (x.relu() + 1) * 2
```

**Reduction fusion**: Fuse reductions with pointwise ops
```python
# Before: 2 kernels
x.sum(dim=-1).add(bias)

# After fusion: 1 kernel
# fused_kernel(x, bias): return x.sum(dim=-1) + bias
```

### Memory Layout

Inductor optimizes memory layout for performance:
- **Channels last** for CNNs
- **Transposed layouts** when beneficial
- **Padding** for alignment

See `config.max_autotune` and `config.coordinate_descent_tuning`.

### Auto-tuning

Enable aggressive auto-tuning:
```python
config.max_autotune = True
config.coordinate_descent_tuning = True
```

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

## Performance Opportunities

Areas for continued optimization and development:

**Codegen Support**:
- Nested tensors / batched operations
- Multi-tensor-apply optimizers (for training)

**Memory Optimization**:
- Advanced memory planning strategies
- Better buffer reuse heuristics

**Compilation Performance**:
- Improving performance heuristics
- Reducing autotuning overhead for small models
- Faster cold-start compilation

**Layout & Padding**:
- Smart padding to improve performance
- Better layout selection heuristics

**Template Support**:
- Expanding library of optimized templates
- Better template fusion with generated kernels

**Resources**:
- Code: https://github.com/pytorch/pytorch/tree/master/torch/_inductor
- Discussion: https://dev-discuss.pytorch.org/t/747
- Results: https://github.com/pytorch/torchdynamo/issues/681

## Progressive Disclosure

- **Getting started**: This file
- **Architecture deep-dive**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Fusion decisions**: [FUSION.md](FUSION.md)
- **Codegen internals**: [CODEGEN.md](CODEGEN.md)
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

- **Dynamo** - Captures FX graphs fed to Inductor
- **AOTAutograd** - Handles autograd graph transformations
- **Triton** - GPU kernel language used by Inductor
- **FX** - Graph representation format

## Key Insights

- Inductor's power comes from **aggressive fusion** and **code specialization**
- The **scheduler** is the brain - it decides what to fuse and when
- **Memory planning** enables buffer reuse, reducing allocations
- **Triton** simplifies GPU codegen but C++ backend is still important
- **Auto-tuning** finds optimal configurations but adds compile time

---

**Line count**: <500 lines (following 500-line rule) ✅
**Progressive disclosure**: Reference files for detailed topics ✅
**YAML frontmatter**: Included ✅
