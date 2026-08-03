# TorchInductor Architecture

Architectural deep-dive into PyTorch's Inductor compiler backend.

**For practical how-to guides**: See [COMMON-PATTERNS.md](COMMON-PATTERNS.md)

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Intermediate Representation](#intermediate-representation)
5. [IR Levels](#ir-levels)
6. [Lowering System](#lowering-system)
7. [Data Flow](#data-flow)
8. [Key Files Reference](#key-files-reference)
9. [Decomposition and Lowering Pipeline](#decomposition-and-lowering-pipeline)

---

## Overview

TorchInductor is PyTorch's deep learning compiler serving as the default backend for `torch.compile()`. It transforms FX graphs into optimized machine code through a multi-stage pipeline.

**Location**: `torch/_inductor/`

**Key Design Philosophy**:
- **Define-by-Run IR**: Loop bodies as executable Python functions
- **Lazy Materialization**: Buffers only realized when scheduler decides
- **Symbolic Shapes**: Full SymPy integration for dynamic shapes
- **Fusion-First**: Aggressive kernel fusion to minimize memory traffic
- **Multi-Backend**: Pluggable backends (Triton, C++, CUDA)

---

## Directory Structure

### Core Compilation Pipeline

**Main files**: `compile_fx.py` (entry), `graph.py` (orchestrator), `lowering.py` (ATen→IR), `ir.py` (IR definitions), `scheduler.py` (fusion), `decomposition.py` (op decompositions)

### Code Generation

**Backends**: `codegen/triton.py` (GPU), `codegen/cpp.py` (CPU), `codegen/wrapper.py` (Python wrapper), `codegen/cuda/` (CUTLASS), `codegen/rocm/` (AMD CK)

### Optimization & Analysis

**Passes**: `fx_passes/pre_grad.py`, `fx_passes/joint_graph.py`, `fx_passes/post_grad.py`, `fx_passes/fuse_attention.py`, `fx_passes/split_cat.py`

**Analysis**: `pattern_matcher.py`, `select_algorithm.py`, `memory.py`, `dependencies.py`

### Runtime & Templates

**Templates**: `kernel/mm.py` (matmul), `kernel/conv.py`, `kernel/flex/` (FlexAttention)

**Runtime**: `runtime/triton_heuristics.py`, `runtime/triton_helpers.py`, `runtime/autotune_cache.py`

### Utilities

**Core utils**: `virtualized.py` (dynamic scoping), `sizevars.py` (symbolic shapes), `config.py` (configuration)

---

## Core Components

### 1. GraphLowering (graph.py)

**Main orchestrator** that interprets FX graphs and converts them to Inductor IR.

**Class Structure**:
```python
class GraphLowering(torch.fx.Interpreter):
    graph_inputs: dict         # Input TensorBoxes
    buffers: list[ir.Buffer]   # All buffers
    operations: list           # All operations
    constants: dict            # Constant tensors
    sizevars: SizeVarAllocator # Symbolic shape management
    scheduler: Scheduler       # Scheduling engine
    wrapper_code: Codegen      # Wrapper generator
```

**Responsibilities**:
- Interpret FX graph nodes via `run_node()`
- Dispatch to lowering functions via `call_function()`
- Manage graph-level state and constants
- Handle shape inference with `SizeVarAllocator`
- Coordinate layout optimization
- Generate final code via `codegen()`

**Entry Flow**:
```
compile_fx(gm) → GraphLowering(gm) → run() → compile_to_module()
```

### 2. Lowering Registry (lowering.py)

**Maps ATen/Prims operations to Inductor IR**.

**Registration**:
```python
lowerings: dict[OpOverload, Callable] = {}

@register_lowering(aten.add)
def add_lowering(a, b):
    return ops.add(a, b)  # ops handler creates IR
```

**Key Concepts**:
- **Lowering Functions**: `TensorBox → TensorBox` mappings
- **ops Handler**: Virtualized handler (define-by-run IR)
- **Layout Constraints**: Control memory layouts
- **Fallback**: Operations without lowerings call eager PyTorch

**Categories**:
- **Direct**: 1-to-1 mapping (`aten.add` → `ops.add`)
- **Template**: Specialized kernels (`aten.mm` → `Gemm`)
- **Decomposition**: Break into simpler ops
- **Fallback**: Eager execution (tracked)

### 3. Scheduler (scheduler.py)

**Brain of the compiler** - determines fusion and execution order.

**Node Types**:
```
BaseSchedulerNode
├── SchedulerNode (single operation)
├── FusedSchedulerNode (fused operations)
│   └── FusedNestedReductions (nested reduction pairs)
├── ExternKernelSchedulerNode (external call)
├── NopKernelSchedulerNode (eliminated)
└── ForeachKernelSchedulerNode (multi-tensor)
```

**Algorithm**:
1. Build dependency graph from IR
2. Topologically sort operations
3. Determine fusion legality (`can_fuse` — binary gate)
4. Score and rank legal fusions (separate from legality)
5. Compute buffer lifetimes
6. Apply memory planning
7. Generate kernel code per node
8. Generate wrapper orchestration

See [FUSION.md](FUSION.md) for the full `can_fuse` decision tree, `MemoryDep` data model, and nested reduction subsystem.

### 4. Memory Planning (memory.py)

**Buffer allocation and reuse strategy**.

**MemoryPlanningState**:
```python
buffer_pool: dict[StorageKey, list[Buffer]]

def allocate(node):
    if reusable := find_reusable(node):
        assign_reused(node, reusable)
    else:
        allocate_new(node)

def free(node):
    buffer_pool[key].append(node.buffer)
```

**Strategy**:
- Pool buffers by size/dtype/device
- Lifetime analysis determines freeing
- Rematerialize by default (recompute vs store)
- In-place mutations when safe

---

## Intermediate Representation

### IR Hierarchy

```
IRNode
├── Constant
├── TensorBox → wraps → StorageBox or View
├── StorageBox → wraps → Buffer
├── View (ReinterpretView, ExpandView, PermuteView, SqueezeView)
└── Buffer
    ├── InputBuffer, ConstantBuffer
    ├── ComputedBuffer (Pointwise, Reduction, Scan, Sort)
    ├── TemplateBuffer (TritonTemplate, CUDATemplate, CppTemplate)
    └── ExternKernelNode (ExternKernelOut, ExternKernelAlloc)
```

### TensorBox → StorageBox → Buffer Chain

**TensorBox**: User-facing tensor (like `torch.Tensor`)
- Wraps `StorageBox` (owns storage) or `View` (shares storage)
- High-level tensor operations

**StorageBox**: Storage + Layout abstraction
- Associates `Buffer` with `Layout`
- Handles layout transformations
- Manages mutations (functionalization)

**Buffer**: Memory allocation
- Concrete storage representation
- Dependency tracking
- Kernel code generation

**Example**:
```python
x_box = TensorBox.create(InputBuffer(...))
relu_buf = Pointwise.create(fn=lambda x: ops.relu(x), inputs=[x_box])
result = TensorBox.create(relu_buf)
```

### View System

**View Chain**: `TensorBox → View → StorageBox → Buffer`

**Types**:
- **ReinterpretView**: `as_strided`, `view`
- **ExpandView**: `expand`, `broadcast_to`
- **PermuteView**: `permute`, `transpose`
- **SqueezeView**: `squeeze`, `unsqueeze`

**Mutation Handling** (Functionalization):
```python
# x.t().add_(1) creates PermuteView then mutates
# Inductor "swings" StorageBox to new buffer
new_buf = Pointwise.create(fn=lambda x: x + 1, ...)
view.data = StorageBox(new_buf)  # Pointer swing
```

### Layout System

```
Layout
├── FixedLayout (concrete size/stride)
├── FlexibleLayout (optimizable)
├── NoneLayout (scalars)
├── MultiOutputLayout
└── MutationLayout
```

**Optimization Flow**:
1. Start with `FlexibleLayout`
2. Scheduler chooses optimal based on:
   - Fusion opportunities
   - Memory access patterns
   - Device capabilities
3. Materialize to `FixedLayout`

**MutationLayout**: When an in-place op (e.g., `x.copy_(y)`) is compiled,
AOT Autograd functionalizes it into a pure computation plus mutation
metadata (`InputAliasInfo` with `mutates_data=True`). Inductor assigns the
output buffer a `MutationLayout` pointing to the input buffer instead of a
`FixedLayout` that would allocate new memory. The buffer records
`mutations = ['arg0_1']`, and the wrapper passes the caller's pointer as
the output pointer — no allocation, no copy. `StarDep` (rather than
`MemoryDep`) is used for mutation dependencies because the op owns the
entire buffer, not a specific access pattern.

### ComputedBuffer (Define-by-Run IR)

**Pointwise**:
```python
Pointwise.create(
    device, dtype,
    inner_fn=lambda index: ops.add(
        ops.load(a, index),
        ops.load(b, index)
    ),
    ranges=[s0, s1, ...]
)
```

**Reduction**:
```python
Reduction.create(
    device, dtype,
    inner_fn=lambda index, rindex: ops.load(x, index + [rindex]),
    ranges=[s0, s1],          # Output dims
    reduction_ranges=[rsize], # Reduction dims
    reduction_type="sum"
)
```

**Key**: `inner_fn` is executable Python function using `ops` handler (different implementations for analysis vs codegen)

---

## IR Levels

Inductor has four distinct IR levels. Each represents the same computation
at a different stage of compilation. Understanding which level you're
looking at is critical for debugging and extending the compiler.

```
Node IR → [scheduler wraps] → Schedule IR → [trace inner_fn] → Loop-Level IR → [ops handler emits] → Codegen IR → [compile] → executable
```

### 1. Node IR (`ir.py`)

**What to compute.** The output of lowering. `Pointwise`, `Reduction`,
`Scan`, `TemplateBuffer`, `ExternKernelOut`. Each node has an `inner_fn`
(executable Python lambda), `ranges` (iteration space), and optionally
`reduction_ranges`. This is the define-by-run IR described above.

### 2. Schedule IR (`scheduler.py`)

**When and with whom.** The scheduler wraps each Node IR buffer in a
`SchedulerNode`, extracts `read_writes` (the `MemoryDep` objects used for
fusion analysis — see [FUSION.md](FUSION.md)), tracks dependencies via
`ancestors` and `unmet_dependencies`, and assigns a `group` key
`(device, (numel, rnumel))` for fusion matching. Fusion produces
`FusedSchedulerNode` (multiple Node IR nodes in one kernel) or
`FusedNestedReductions` (nested reduction pairs).

### 3. Loop-Level IR (`loop_body.py`)

**How to iterate.** Each `SchedulerNode`'s `inner_fn` is traced into a
`LoopBody` — an FX graph of `ops.load`, `ops.store`, `ops.add`,
`ops.index_expr`, etc. This is the **second FX graph** in the pipeline
(the first was Dynamo's ATen-level graph). The `LoopBody` has `var_ranges`
(iteration variables and their extents) and a `body()` method containing
the ops sequence. This level separates index computation from value
computation.

### 4. Codegen IR (phase buffers → target source)

**Target code.** The ops handler (e.g., `TritonKernelOverrides`) interprets
Loop-Level IR's `ops.*` calls and writes string fragments into phase
buffers (`loads`, `compute`, `stores`). `codegen_body()` assembles them
into Triton/C++ source. See [CODEGEN.md](CODEGEN.md) for the full
assembly mechanics.

### Level Summary

| Level | Lives in | Key types | Represents |
|---|---|---|---|
| Node IR | `ir.py` | `Pointwise`, `Reduction`, `TemplateBuffer` | What to compute |
| Schedule IR | `scheduler.py` | `SchedulerNode`, `FusedSchedulerNode`, `MemoryDep` | When and with whom (fusion) |
| Loop-Level IR | `loop_body.py` | `LoopBody`, `ops.load`, `ops.store` | How to iterate |
| Codegen IR | `codegen/` | Phase buffers, `IndentedBuffer`, target strings | Target source code |

---

## Lowering System

### Lowering Process

**FX Node → Inductor IR transformation**.

**Example Flow (aten.add)**:
1. FX: `%add = call_function[aten.add.Tensor](%a, %b)`
2. GraphLowering: `lowerings[aten.add.Tensor](a_box, b_box)`
3. Lowering fn: `return ops.add(a, b)`
4. ops.add (TritonOverrides): Creates `Pointwise` IR node
5. Pointwise → ComputedBuffer
6. Wrap in TensorBox and return

### Decomposition System

**Break complex ops into primitives**.

**Registration**:
```python
@register_decomposition([aten.gelu])
def gelu_decomposition(x, approximate="none"):
    if approximate == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(...))
    else:
        return x * 0.5 * (1.0 + torch.erf(...))
```

**Tables**:
- `core_aten_decompositions`: Always applied
- `inductor_decompositions`: Inductor-specific
- Custom tables via AOTAutograd

---

## Scheduling & Fusion

See [FUSION.md](FUSION.md) for fusion legality, scoring, the `MemoryDep` data model, and the nested reduction subsystem.

---

## Code Generation

See [CODEGEN.md](CODEGEN.md) for class hierarchies, ops handler pattern, range trees, CSE, phased assembly, and backend registration.

---

## Data Flow

### End-to-End Compilation

```
Input FX Graph (Dynamo/export)
    ↓
Pre-Grad Passes (normalize, pattern matching)
    ↓
AOT Autograd (forward/backward, functionalization)
    ↓
Joint Graph Passes (attention fusion, custom patterns)
    ↓
Post-Grad Passes (layout optimization, split/cat fusion)
    ↓
GraphLowering.run() (interpret nodes, build IR, apply constraints)
    ↓
Scheduler (dependencies, fusion, lifetimes, memory planning)
    ↓
Code Generation (Triton/C++ kernels, wrapper code)
    ↓
Compilation (Triton→PTX/HSACO, C++→.so, cache)
    ↓
Output: Compiled Function (callable, JIT-compiled kernels)
```

---

## Key Files Reference

### Compilation Pipeline
- `compile_fx.py`: Entry point
- `graph.py`: GraphLowering orchestrator
- `lowering.py`: ATen→IR lowering registry
- `decomposition.py`: Op decompositions
- `ir.py`: IR node definitions

### Fusion Decisions
- `scheduler.py`: Fusion legality, scoring, `NestedReduction`, `FusedNestedReductions`
- `dependencies.py`: `MemoryDep`, `StarDep`, dependency analysis
- `codegen/simd.py`: `SIMDScheduling.can_fuse()`, node schedule generation

### Scheduling & Optimization
- `memory.py`: Memory planning, buffer reuse
- `pattern_matcher.py`: Pattern matching and replacement
- `select_algorithm.py`: Algorithm selection and autotuning

### Code Generation
- `codegen/triton.py`: Triton kernels
- `codegen/cpp.py`: C++ kernels
- `codegen/wrapper.py`: Python wrappers
- `codegen/cuda/`: CUDA templates
- `codegen/rocm/`: AMD templates

### Utilities
- `virtualized.py`: Dynamic scoping
- `sizevars.py`: Symbolic shapes
- `config.py`: Configuration
- `utils.py`: Utilities
- `metrics.py`: Metrics

### Passes & Templates
- `fx_passes/pre_grad.py`: Pre-autograd
- `fx_passes/joint_graph.py`: Joint fwd/bwd
- `fx_passes/post_grad.py`: Post-autograd
- `kernel/mm.py`: Matmul templates
- `kernel/conv.py`: Conv templates
- `runtime/triton_heuristics.py`: Tuning

---

## Summary

TorchInductor's architecture enables:

✅ **FX graph compilation** to efficient GPU/CPU code
✅ **Aggressive optimizations** (fusion, layout, memory planning)
✅ **Dynamic shape support** via SymPy symbolic reasoning
✅ **Extensibility** for new operations and backends
✅ **Auto-tuning** for performance
✅ **Multi-backend** support (Triton, C++, CUDA)

**Key Architectural Insights**:

1. **Define-by-Run IR**: Python functions as IR enables rapid lowering with minimal boilerplate
2. **Virtualized ops**: Different handlers for analysis vs codegen using same lowering code
3. **TensorBox chain**: Clean separation (user API → storage → memory)
4. **Lazy materialization**: Scheduler decides what to compute
5. **Symbolic shapes**: Full SymPy integration from ground up
6. **Fusion-first**: Aggressive fusion drives performance
7. **Layout flexibility**: Optimizer chooses best layout
8. **Multi-backend**: Pluggable codegen for different hardware

## Decomposition and Lowering Pipeline

**Decompositions run FIRST** (in AOT Autograd), then lowerings (in Inductor):

### Pipeline architecture

```
┌────────────────────────────────────────────────────────────────┐
│  AOT Autograd (compile_fx.py)                                  │
│  - Creates Core ATen IR via functionalization                  │
│  - Applies core_aten_decompositions() (Full ATen → Core ATen)  │
│  - Applies inductor_decompositions (some ops → Prims)          │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│  Inductor Graph Lowerings (graph.py call_function)             │
│  - Processes post-decomposition graph                          │
│  - Lowerings generate IR nodes (Buffer, Pointwise, Reduction)  │
│  - Fallback to extern kernel if no lowering exists             │
└────────────────────────────────────────────────────────────────┘
```

### Pipeline order

1. **AOT Decompositions** - Applied to FX graph before Inductor sees it
2. **Inductor Lowerings** - Process the post-decomposition graph
3. **Fallback/Extern Kernel** - Last resort when no lowering exists

### When ops have both decomposition and lowering

- Decomposition runs first in AOT stage for preprocessing/normalization
- If decomposition returns `NotImplemented`, op stays in graph
- Inductor lowering then generates IR for that op

### Example: aten.full

**Decomposition (AOT stage)**: Infers dtype if missing, returns `NotImplemented` if dtype present

**Lowering (Inductor stage)**: Assumes dtype is set, generates IR via `tensor_constructor()`

**Flow**: decomposition preprocesses → lowering generates IR

### Key insight

Decompositions transform the graph structure; lowerings generate loop-level IR.

### Key files

- `torch/_inductor/compile_fx.py`: AOT applies decompositions
- `torch/_inductor/graph.py`: `call_function()` dispatches to lowerings
- `torch/_inductor/lowering.py`: Lowering registrations
- `torch/_inductor/decomposition.py`: Inductor-specific decompositions

---

**For practical patterns and examples**: See [COMMON-PATTERNS.md](COMMON-PATTERNS.md)

