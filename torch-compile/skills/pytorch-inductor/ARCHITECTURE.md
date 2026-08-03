# TorchInductor Architecture

Architectural deep-dive into PyTorch's Inductor compiler backend.

**For practical how-to guides**: See [COMMON-PATTERNS.md](COMMON-PATTERNS.md)

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Intermediate Representation](#intermediate-representation)
5. [Lowering System](#lowering-system)
6. [Scheduling & Fusion](#scheduling--fusion)
7. [Code Generation](#code-generation)
8. [Data Flow](#data-flow)
9. [Design Patterns](#design-patterns)
10. [Key Files Reference](#key-files-reference)

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
├── ExternKernelSchedulerNode (external call)
├── NopKernelSchedulerNode (eliminated)
└── ForeachKernelSchedulerNode (multi-tensor)
```

**Algorithm**:
1. Build dependency graph from IR
2. Topologically sort operations
3. Group fusible operations (score-based)
4. Compute buffer lifetimes
5. Apply memory planning
6. Generate kernel code per node
7. Generate wrapper orchestration

**Fusion Strategies**:

**Pointwise**: Element-wise ops with same iteration space
```python
can_fuse = (both_pointwise and same_device and
            same_numel and no_cycle)
```

**Reduction**: Same reduction pattern
```python
can_fuse = (same_numel and same_rnumel and
            same_reduction_type)
```

**Horizontal**: Independent ops share kernel launch
```python
can_fuse = (same_kernel_type and independent and same_device)
```

**Scoring**:
- +10 per shared input (saves memory reads)
- Penalty for distance in execution order
- Penalty for increased register pressure

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

### Fusion Legality

**Requirements**:
1. Same device
2. Compatible iteration space
3. No cyclic dependencies
4. Memory traffic reduction

**Scoring**:
```python
score = (num_shared_inputs * 10) - abs(node1.index - node2.index)
if increases_register_pressure: score -= 50
```

### Fusion Patterns

**1. Vertical (Producer-Consumer)**:
```python
producer = x.relu()
consumer = producer.add(1)
# Fused: lambda idx: add(relu(load(x, idx)), 1.0)
```

**2. Horizontal (Independent)**:
```python
y1 = x.relu(); y2 = x.sigmoid()
# Fused: x_val = load(x, idx); store(y1, relu(x_val)); store(y2, sigmoid(x_val))
```

**3. Reduction**:
```python
sum1 = x.sum(dim=-1); max1 = x.max(dim=-1)
# Fused: Single reduction loop with multiple accumulators
```

**4. Normalization (Reduction + Pointwise)**:
```python
mean = x.mean(keepdim=True); norm = x - mean
# Two-pass: reduction then pointwise using result
```

---

## Code Generation

Codegen is organized into three independently extensible pillars. For
detailed internals (range trees, CSE, phased assembly, ops handler pattern),
see [CODEGEN.md](CODEGEN.md).

### Kernel Hierarchy

```
Kernel                              -- phase buffers (loads/compute/stores), CSE, args
  └─ SIMDKernel                     -- flattened indexing, range trees, tiling
       ├─ TritonKernel              -- Triton GPU kernels
       ├─ HalideKernel / MetalKernel / PallasKernel
  └─ CppKernel                      -- Loop-nest based (not SIMD flattened)
       ├─ CppVecKernel              -- AVX2/AVX512 vectorized
       └─ CppKernelProxy            -- Dispatch to vec/tile/scalar
```

### Wrapper Hierarchy

```
PythonWrapperCodegen                -- Python wrapper (default)
  ├─ SubgraphPythonWrapperCodegen   -- Nested subgraph wrappers
  └─ CppWrapperCpu                  -- C++ wrapper for AOTInductor
       └─ CppWrapperGpu             -- + GPU kernel launch
```

### Scheduling Hierarchy

```
BaseScheduling
  └─ SIMDScheduling                 -- Fusion logic for SIMD backends
       ├─ TritonScheduling          -- Triton-specific fusion rules
       ├─ HalideScheduling / MetalScheduling / PallasScheduling
  └─ CppScheduling                  -- C++ backend fusion
```

### Backend Registration

The `DeviceCodegen` registry binds device → scheduling + wrapper.
New backends register via `register_backend_for_device()`.

### Generated Code Examples

**Triton kernel**:
```python
@triton.jit
def kernel(in_ptr0, in_ptr1, out_ptr0, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    tl.store(out_ptr0 + offsets, x + y, mask=mask)
```

**C++ kernel** (AVX2/AVX512 + OpenMP):
```cpp
#pragma omp parallel for
for (int64_t i = 0; i < numel; i += 8) {
    __m256 a = _mm256_loadu_ps(ptr_a + i);
    __m256 b = _mm256_loadu_ps(ptr_b + i);
    _mm256_storeu_ps(ptr_out + i, _mm256_add_ps(a, b));
}
```

**Python wrapper** (orchestration):
```python
def compiled_fn(arg0, arg1):
    buf0 = torch.empty([s0, s1], device='cuda', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(s0*s1, meta['BLOCK_SIZE']),)
    triton_kernel[grid](arg0, arg1, buf0, numel=s0*s1, BLOCK_SIZE=1024)
    return buf0
```

**Features**: Buffer allocation, argument preparation, grid sizing, CUDA graphs, synchronization

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

## Design Patterns

### 1. Virtualized Variables

**Thread-local dynamic scoping** for compilation context access.

```python
from torch._inductor.virtualized import V

# Graph state
V.graph.sizevars.size_hint(expr)
V.graph.wrapper_code.writeline("...")

# Current context
with V.set_current_node(node):
    # V.current_node accessible

# Operation handler (define-by-run)
ops.add(a, b)  # Different handler per context
```

**Key Variables**:
- `V.graph`: `GraphLowering` instance
- `V.fake_mode`: `FakeTensorMode` for shapes
- `V.kernel`: Current kernel (in codegen)
- `ops`: Operation handler (varies)

**ops Handlers**:
- `MockHandler`: Dependency analysis
- `TritonOverrides`: Triton codegen
- `CppOverrides`: C++ codegen
- `FallbackHandler`: Eager PyTorch

### 2. Size Variables

**Symbolic shape handling with SymPy**.

```python
class SizeVarAllocator:
    shape_env: ShapeEnv
    var_to_val: dict[sympy.Symbol, int]

    def size_hint(self, expr: sympy.Expr) -> int:
        return int(expr.subs(self.var_to_val))

    def statically_known_equals(self, a, b) -> bool:
        return self.shape_env.evaluate_expr(sympy.Eq(a, b))
```

**Dynamic Shapes**:
- Sizes: `s0, s1, s2, ...`
- Strides: `s1*s2, s2, 1`
- Guards: `s0 > 0`, `s0 == s1`
- Specialization: Generate guards, compile

### 3. Dependencies

**Dependency tracking for scheduling**.

```python
@dataclass
class MemoryDep:
    name: str            # Buffer name
    index: sympy.Expr    # Access pattern
    size: tuple          # Shape
    mode: str            # "read" or "write"

@dataclass
class StarDep:
    name: str  # Entire buffer (unknown pattern)
    mode: str

class WeakDep(Dep):
    pass  # Ordering only
```

**Scheduler Usage**:
- Build dependency graph
- Determine fusion legality
- Compute buffer lifetimes

### 4. IndentedBuffer

**Code generation utility**.

```python
code = IndentedBuffer()
code.writeline("def fn():")
with code.indent():
    code.writeline("x = 1")
code.splice(multiline_string)  # Preserves indent
```

### 5. CSE (Common Subexpression Elimination)

**Reuse computed values**.

```python
class CSE:
    cache: dict[str, CSEVariable]

    def generate(self, expr, dtype):
        key = cache_key(expr, dtype)
        if key in cache:
            return cache[key]
        var = newvar()
        emit(f"{var} = {codegen(expr)}")
        cache[key] = var
        return var
```

---

## Key Files Reference

### Compilation Pipeline
- `compile_fx.py` (2997): Entry point
- `graph.py` (2569): GraphLowering
- `lowering.py` (7660): ATen→IR
- `decomposition.py` (1259): Decompositions
- `ir.py` (9705): IR definitions

### Scheduling & Optimization
- `scheduler.py` (6588): Fusion/scheduling
- `dependencies.py` (890): Dependency analysis
- `memory.py` (1108): Memory planning
- `pattern_matcher.py` (2368): Pattern matching
- `select_algorithm.py` (4557): Algorithm selection

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

**For debugging help**: See [DEBUGGING-GUIDE.md](DEBUGGING-GUIDE.md)

**For optimization tips**: See [OPTIMIZATION-GUIDE.md](OPTIMIZATION-GUIDE.md)
