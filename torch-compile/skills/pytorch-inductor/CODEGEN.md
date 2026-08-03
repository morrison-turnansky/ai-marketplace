# Codegen Internals

How Inductor transforms **Loop-Level IR** (the `LoopBody` FX graph of
`ops.*` calls) into **Codegen IR** (Triton/C++ source). For the four IR
levels and how they relate, see [ARCHITECTURE.md](ARCHITECTURE.md#ir-levels).
For the template system (GEMM, conv, attention), see
[TRITON-TEMPLATES.md](TRITON-TEMPLATES.md).

## Three Pillars

Inductor's codegen is organized into three independently extensible pillars:
kernel codegen (compute), wrapper codegen (orchestration), and scheduling
(fusion decisions + dispatch to kernel codegen).

### Kernel Hierarchy -- generates compute code

```
CodeGen
  └─ Kernel                              -- phase buffers (loads/compute/stores), CSE, args
       └─ SIMDKernel                     -- flattened indexing, range trees, tiling
            ├─ TritonKernel              -- Triton GPU kernels
            ├─ HalideKernel              -- Halide (experimental)
            ├─ MetalKernel               -- Apple Metal shaders
            └─ PallasKernel              -- JAX Pallas for TPU
       └─ CppKernel                      -- Loop-nest based (not SIMD flattened)
            ├─ CppVecKernel              -- AVX2/AVX512 vectorized
            │    └─ CppTile2DKernel      -- 2D tiling
            └─ CppKernelProxy            -- Dispatch to vec/tile/scalar
       └─ ComboKernel                    -- Multiple sub-kernels in one launch
```

Each layer adds one capability: Kernel provides phase buffers and CSE,
SIMDKernel adds flattened indexing via range trees, TritonKernel adds
Triton-specific code emission. A new SIMD backend only needs to subclass
SIMDKernel and SIMDScheduling.

### Wrapper Hierarchy -- generates orchestration code

```
CodeGen
  └─ PythonWrapperCodegen                -- Python wrapper (default)
       ├─ SubgraphPythonWrapperCodegen   -- Nested subgraph wrappers
       └─ CppWrapperCpu                  -- C++ wrapper for AOTInductor
            └─ CppWrapperGpu             -- + GPU kernel launch
```

### Scheduling Hierarchy -- decides fusion, delegates codegen

```
BaseScheduling
  └─ SIMDScheduling                      -- Fusion logic for SIMD backends
       ├─ TritonScheduling               -- Triton-specific fusion rules
       ├─ HalideScheduling
       ├─ MetalScheduling
       └─ PallasScheduling
  └─ CppScheduling                       -- C++ backend fusion
```

### DeviceCodegen Registry

The `DeviceCodegen` registry binds a device to its scheduling + wrapper:

```python
device_codegens["cuda"] = DeviceCodegen(
    scheduling=TritonScheduling,
    wrapper_codegen=PythonWrapperCodegen,
    cpp_wrapper_codegen=CppWrapperGpu,
)
```

New backends register via `register_backend_for_device()`. This is the
single extensibility contract -- you provide a Scheduling class and a
Wrapper class; everything else is inherited or overridden. Intel XPU,
AMD ROCm, Apple MPS, and Google TPU all use this mechanism.

## The `ops` Handler Pattern

`ops.*` calls are **Loop-Level IR** operations. The active handler (set
via thread-local state in `virtualized.py`) translates them into
**Codegen IR** (target-specific strings written into phase buffers). This
avoids the backend explosion that would result from parameterizing every
lowering function — lowering code calls `ops.add(a, b)` without knowing
the backend.

### OpOverrides Hierarchy

```
OpOverrides                              -- Base: string-level code gen
  ├─ TritonOverrides                     -- Pure math string translation (stateless)
  │    └─ TritonKernelOverrides          -- + kernel-aware ops using V.kernel
  ├─ CppOverrides                        -- C++ code strings
  │    ├─ CppVecOverrides               -- Vectorized intrinsics
  │    └─ CppTile2DOverrides
  ├─ HalideOverrides
  ├─ MetalOverrides
  └─ PallasKernelOverrides
```

### TritonOverrides vs TritonKernelOverrides

**TritonOverrides** provides pure element-wise expression translation.
Stateless `@staticmethod` methods that take strings and return strings:

```python
ops.to_dtype(x, torch.float16)  ->  "x.to(tl.float16)"
ops.exp(x)                      ->  "tl_math.exp(x)"
```

No knowledge of the kernel -- no `V.kernel`, no buffers, no indexing.

**TritonKernelOverrides(TritonOverrides)** extends with kernel-aware
operations that reference `V.kernel` -- `constant()` needs the tensor ndim,
`index_expr()` calls the kernel's indexing method, `masked()` writes into
compute buffers.

The only place a bare `TritonOverrides()` instance is used independently
is in `_lift_helper`, which generates standalone `@triton.jit` combine
functions for `tl.reduce` and `tl.associative_scan`.

### Where load/store live

`load()` and `store()` are methods on **TritonKernel itself**, not on the
overrides classes. The overrides handle element-wise compute (`ops.add`,
`ops.exp`, `ops.where`), while `TritonKernel.load()` and `.store()` handle
memory access directly. Both write into the same set of IndentedBuffers.

## Range Trees

### The Problem They Solve

Triton kernels operate over flat 1D blocks, but tensors are
multi-dimensional. Range trees decompose a flat index like `xindex` into
individual dimension variables (`x0`, `x1`, `x2`) that indexing
expressions need.

### Structure

```
IterationRanges (base)           -- name, numel, prefix, divisor, length
├── IterationRangesRoot          -- one per tiled dimension (x, y, z, r0_, r1_)
│   └── nodes: dict[Expr, IterationRangesEntry]
└── IterationRangesEntry         -- one per unique (divisor, length) split
```

One root per tiled dimension. Each root owns a flat dict of entries keyed
by the SymPy expression (FloorDiv or ModularIndexing). When two fused nodes
access the same dimension with the same split, they share the entry.

### Concrete Example

A tensor `t[4, 3, 2]` accessed via a single `x` tile with flat numel=24.
When lowering encounters `t[i, j, k]`, it calls `root.lookup(divisor, length)`:

```
root.lookup(6, 4)  -> x0 = xindex // 6           (outermost, stride=6)
root.lookup(2, 3)  -> x1 = (xindex // 2) % 3     (middle)
root.lookup(1, 2)  -> x2 = xindex % 2             (innermost)
```

`construct_entries` automates this: given `lengths=[4, 3, 2]`, walks
right-to-left building up the divisor.

### Tensor Dim / Grid Dim Mapping

Each root is assigned both a tensor_dim (position in multi-dim Triton tensor)
and a grid_dim (which `tl.program_id`):

```
prefix    tensor_dim    grid_dim    role
  z           0            2       batch (outermost pointwise)
  y           1            1       middle pointwise
  x           2            0       innermost pointwise
  r0_         3           None     reduction (loop or persistent, no grid)
```

tensor_dim determines the shape of `tl.arange` reshaping. For ndim=3, x range
becomes `[:, None, None]` while r0_ becomes `[None, None, :]`. Broadcasting
makes element-wise ops work across all dimensions.

### How Range Trees Become Triton Code

`codegen_range_tree()` iterates roots and calls
`iteration_ranges_codegen_header` for each non-loop tree, producing:

```python
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = xindex < xnumel
```

`IterationRangesEntry.codegen()` is called lazily (cached on first use),
emitting decomposition lines:

```python
x0 = xindex // 6
x1 = (xindex // 2) % 3
x2 = xindex % 2
```

These go into `self.body` (pointwise dims) or `self.indexing_code`
(reduction dims inside the loop).

## Phased Code Assembly

As the ops handler interprets Loop-Level IR (`LoopBody`'s `ops.*` calls),
it writes Codegen IR fragments into separate phase buffers, then assembles
them into the right control flow structure. This separates *what to
compute* from *how to structure the control flow*.

### Buffer Layout per Kernel Layer

Each layer in the hierarchy adds buffers for its concerns:

- **Kernel** (base): `loads`, `compute`, `stores` -- the three phases
- **SIMDKernel**: `body` (assembled output), `indexing_code` (inside loops)
- **TritonKernel**: `prologue`, `post_loop_combine`, `post_loop_store`

### How Load/Store Write Into Buffers

**load()**: Resolves the buffer arg name, computes indexing, writes
`tl.load(...)` into `self.loads` via CSE, returns a CSEVariable.

**store()**: Resolves the output arg, computes indexing, writes
`tl.store(...)` into `self.stores` as a DeferredLine (can be retroactively
removed if the buffer is later marked dead via `V.graph.removed_buffers`).

**Compute** (via TritonKernelOverrides): `ops.add(a, b)` returns
`"a + b"`, which CSE may assign to `tmp0 = a + b` in `self.compute`.

### codegen_body: Assembly

`codegen_body()` splices the filled phase buffers into `self.body`:

**Pointwise** (flat splice):
```
indexing_code | loads | compute | stores
```

**Looped reduction** (loop-wrapped):
```
for roffset in tl.range(...):
    indexing_code | loads | compute | stores
post_loop_combine | post_loop_store
```

**Persistent reduction** (same as pointwise -- entire reduction dim fits
in one block).

After assembly, the per-phase buffers are cleared. This is a drain-and-clear
pattern -- `codegen_body()` can be called multiple times per kernel. For
reduction kernels it is called once to emit the reduction loop, then again
after the epilogue section to emit the pointwise stores.

### Two-Pass Schedule Codegen

`codegen_node_schedule_with_kernel()` drives the node schedule through two
passes with a critical step between them:

1. **Pass 1** -- Collect indexing and decide inplace updates. Iterates the
   schedule, calling `split_and_set_ranges()` and `indexing_from_args()`
   for each node.
2. **`finalize_indexing()`** -- Runs between passes. Finalizes index
   calculations now that all nodes' access patterns are known.
3. **Pass 2** -- Actual codegen. Applies dtype strength reduction and
   index-to-value conversion, then calls `node.codegen()` which writes
   into the phase buffers.

Both passes respect `DisableReduction`/`EnableReduction` markers in the
schedule (see [Node Schedule Structure](#node-schedule-structure)).

`codegen_kernel()` then wraps `self.body` in the `@triton.jit` function
with signature, grid heuristics, and metadata.

## CSE (Common Subexpression Elimination)

CSE prevents redundant computation. More importantly, it is the mechanism
that makes fusion actually eliminate memory traffic.

### Class Hierarchy

```
CSEVariable              -- a named variable (e.g., "tmp0") with bounds, dtype, shape
  └─ TritonCSEVariable   -- adds mask_vars: tracks which masks apply to this value

CSE                      -- the cache + code emission engine
  └─ TritonCSE           -- augments cache key with current load mask
```

TritonCSEVariable tracks `mask_vars` -- which masks must be applied when
this variable is used for indirect indexing. The `update_on_args` hook
propagates mask_vars from input args to output.

### The `generate()` Method

Given an expression string, either return the existing CSEVariable (cache
hit, no code emitted) or create a new tmp variable, emit the assignment to
the target buffer, and cache it. If the expression is already a
CSEVariable, it's returned directly with tightened bounds.

### Three Caches

**`_cache`** (expression string -> CSEVariable): Main CSE cache. Ensures
identical expression strings aren't recomputed.

**`store_cache`** (buffer name -> CSEVariable): Cross-node forwarding.
When node A stores to buffer "buf0", store_cache maps "buf0" to the
in-register value. When node B loads from "buf0", CSEProxy returns the
cached value directly -- **no memory round-trip**. This is the key
mechanism that makes producer-consumer fusion eliminate memory traffic.
Without it, fusing two nodes just puts them in the same kernel but still
emits `tl.store` + `tl.load` between them.

**`reduction_cache`** (reduction key -> CSEVariable): Caches reduction
results to avoid recomputing identical reductions.

### CSEProxy: The Glue Layer

CSEProxy sits between the ops handler and the kernel. It wraps every
op call to:

1. Compute value range bounds via ValueRangeAnalysis
2. Delegate to the actual handler (e.g., TritonKernelOverrides)
3. Pass the result through `CSE.generate()` to deduplicate and emit code

For load/store it has special logic: loads check store_cache first
(cross-node forwarding), stores update store_cache for future consumers.

### TritonCSE: Mask-Aware Caching

TritonCSE overrides `augment_key()` to include the current load mask in
the cache key. This prevents CSE from reusing a value computed under one
mask for code under a different mask.

### Cache Invalidation

`CSE.invalidate()` is called at reduction loop boundaries. It clears
store_cache and expression cache entries for variables computed inside
the loop (which may hold different values on the next iteration), while
preserving variables in `outside_loop_vars`.

`scoped_copy()` creates a child CSE with ScopedDict overlays, so
inner-scope entries are invisible to the outer scope.

### DeferredLine

Used by `store()` to emit lines that can be retroactively removed. If the
scheduler later marks a buffer as dead (`V.graph.removed_buffers`), the
DeferredLine returns None and the line is silently dropped.

## Wrapper Codegen

The wrapper uses a deferred line architecture -- appends typed WrapperLine
objects to a list instead of directly writing code. Each line type knows
how to render itself:

- **AllocateLine** -- `torch.empty(...)` calls
- **KernelCallLine** -- `kernel[grid](*args)` launches
- **FreeLine** -- buffer deallocation
- **MemoryPlanningLine** -- reuse/reinterpret decisions
- **EnterDeviceContextManagerLine** -- device guards

This allows memory planning to rewrite allocations after the full schedule
is known.

## Design Principles

These principles shape decisions throughout the codegen pipeline:

**Define-by-Run IR**: Loop bodies are executable Python functions
(`inner_fn`), not static data structures. The same function can be
interpreted in different ways by swapping the `ops` handler (analysis,
Triton codegen, C++ codegen, fallback). Lowering code is written once
and works across all backends.

**Deferred Materialization**: Decisions are deferred so later stages can
make better-informed choices:
- DeferredLines can be retroactively removed when buffers are marked dead
- FlexibleLayout is frozen to FixedLayout only when the scheduler commits
- store_cache forwarding means loads may never be emitted
- Range tree entries codegen lazily on first use
- WrapperLines let memory planning rewrite allocations after scheduling

**Separation of Kernel vs Wrapper**: Kernel codegen produces compute code
(the `@triton.jit` function). Wrapper codegen produces orchestration code
(buffer allocation, grid computation, kernel launch). They are
independently extensible via the DeviceCodegen registry.

## Full Pipeline (One Kernel)

```
1. Scheduler decides which nodes to fuse, creates SIMDKernelFeatures
2. TritonScheduling creates TritonKernel with tiling parameters
3. __init__ calls codegen_range_tree() -- emits xindex/xmask into self.body
4. For each SchedulerNode in the fused group:
   a. kernel.set_current_node(node)
   b. node's inner_fn is called -- triggers ops.load / ops.add / ops.store
   c. CSEProxy intercepts: loads check store_cache, computes go through CSE
   d. load() writes to self.loads, compute to self.compute, store() to self.stores
   e. IterationRangesEntry.codegen() lazily emits index decomposition on first use
5. codegen_body() splices phase buffers into self.body with appropriate control flow
6. codegen_kernel() wraps self.body in @triton.jit with signature and metadata
7. Source compiled by Triton -> PTX -> cubin, cached on disk
```

## Node Schedule Structure

The node schedule is not a flat list of operations — it is a structured
sequence with control flow markers from `codegen/simd_kernel_features.py`:

- **`DisableReduction`** / **`EnableReduction`** — bracket epilogue sections
  where reduction results are consumed by pointwise ops. When `DisableReduction`
  fires, `codegen_body()` flushes the reduction loop, then `inside_reduction`
  is set to `False` so subsequent nodes generate pointwise code. `EnableReduction`
  flushes the pointwise code and re-enters the reduction context.

This marker system is how reduction + epilogue fusion works in practice:
the scheduler decides the fusion is legal (see [FUSION.md](FUSION.md)),
`generate_node_schedule()` arranges nodes with markers, and
`codegen_node_schedule_with_kernel()` drives the two-pass codegen through
the marked schedule.

## Nested Reduction Codegen

When `FusedNestedReductions` reaches codegen, specialized machinery in
`codegen/simd.py` handles the multi-resolution tile layout:

### `_GroupedReductionLayout`

Maps a kernel's range trees onto a grouped reduction structure with three
axes: passthrough (rows that don't participate in reduction), group
(reduction groups), and local reduction (elements within a group). Handles
the geometric layout of tiles and provides methods for broadcasting values
between resolutions.

### `_DerivedIterationFamily`

Defines a derived iteration space for consumer stages that operate at a
different resolution than the parent reduction. Contains:

- `index_subs` — rewrite body iteration variables to derived tree symbols
- `remap_index()` — transform a sympy index expression from the consumer's
  natural space to the derived space

When active on a kernel, the derived family's range trees replace the
kernel's own for the duration of the consumer stage.

### `_PointwiseRemapHandler`

A `WrapperHandler` that intercepts `load` and `store` to transparently
remap indices via a `_DerivedIterationFamily`. This is the mechanism for
handler wrapping — a general pattern where an outer handler intercepts
specific ops, transforms their arguments, and delegates to the inner handler:

```python
class _PointwiseRemapHandler(WrapperHandler):
    def load(self, name, index):
        remapped = self._family.remap_index(index)
        return self._inner.load(name, remapped)
```

Handler wrapping is the codegen-side counterpart to the scheduler's fusion
decision. The scheduler decides "these can share a kernel," and the wrapped
handler makes that work by translating between iteration spaces at codegen
time.

## Key Files

- `codegen/common.py` — Kernel base, CSE, IndentedBuffer, OpOverrides, WrapperHandler
- `codegen/simd.py` — SIMDKernel, SIMDScheduling, range trees, `_GroupedReductionLayout`, `_DerivedIterationFamily`, `_PointwiseRemapHandler`
- `codegen/simd_kernel_features.py` — Schedule markers (`DisableReduction`, `EnableReduction`)
- `codegen/triton.py` — TritonKernel, TritonOverrides, TritonScheduling
- `codegen/cpp.py` — CppKernel, CppOverrides, CppScheduling
- `codegen/wrapper.py` — PythonWrapperCodegen, WrapperLine types
- `codegen/cuda/` — CUTLASS templates
- `virtualized.py` — `V` namespace, `ops` handler thread-local dispatch

---

**For fusion decisions** (legality, scoring, `MemoryDep`): [FUSION.md](FUSION.md)
**For template-based codegen** (GEMM, conv, attention): [TRITON-TEMPLATES.md](TRITON-TEMPLATES.md)
**For architecture overview**: [ARCHITECTURE.md](ARCHITECTURE.md)
**For practical patterns**: [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
