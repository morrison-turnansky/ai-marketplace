# Fusion Decisions

How Inductor decides which operations share a kernel. Fusion is the primary
source of performance — it eliminates memory round-trips between operations.

For how fused operations become executable code, see [CODEGEN.md](CODEGEN.md).

## Legality vs Scoring

Fusion is a two-phase process. These are architecturally distinct:

1. **Legality** (`can_fuse`) — binary gate: can these two nodes legally share
   a kernel without changing semantics?
2. **Scoring** — among legal candidates, which fusion saves the most memory
   traffic? Higher score = more shared inputs eliminated.

Debugging "why didn't these fuse?" always starts with `can_fuse`. Scoring
only matters when multiple legal fusions compete.

## `can_fuse` Decision Tree

`SIMDScheduling.can_fuse()` in `codegen/simd.py` is a multi-branch decision
tree. The common misconception is that fusion requires "same numel" — the
actual logic has at least five branches:

1. **ForeachKernel delegation** — multi-tensor ops (`torch._foreach_*`) have
   their own fusion rules and are handled first.

2. **Split-scan incompatibility** — split scans cannot fuse with reductions.

3. **Reduction + Reduction** — requires matching `(numel, rnumel)`. If that
   fails, falls through to:
   - `MixOrderReduction.can_fuse` — can the two reductions share a
     mixed-order loop?
   - `NestedReduction` dependent-pair check — are these two reductions over
     the same logical elements at different granularities (e.g., layernorm
     followed by block-scale quantization)?
   - Native-matmul tiling compatibility

4. **Non-reduction pairs** — requires `numel` and `rnumel` match, with
   exceptions for template/prologue fusion and tiling compatibility checks.

5. **Reduction + Pointwise epilogue** — when one node is a reduction and the
   other is pointwise with `rnumel=1`, args are swapped and re-checked.
   This is how reduction epilogues (e.g., `x.mean(keepdim=True)` followed
   by `x - mean`) get fused into a single kernel with two passes.

## Key Data Structure: `MemoryDep`

Fusion operates on **Schedule IR** (see [ARCHITECTURE.md](ARCHITECTURE.md#ir-levels)).
`MemoryDep` objects are extracted from Node IR's `inner_fn` during scheduler
initialization and attached to Schedule IR nodes. All fusion decisions
examine these objects from `dependencies.py`:

```python
class MemoryDep:
    name: str                            # Buffer name (e.g., "buf0")
    index: sympy.Expr                    # Access pattern (e.g., 8192*d0 + d1)
    var_names: tuple[sympy.Symbol, ...]  # Iteration variables (d0, d1, ...)
    size: tuple[sympy.Expr, ...]         # Extent per variable

    @property
    def ranges(self) -> dict:            # {var: extent} — the iteration domain
        return dict(zip(self.var_names, self.size))
```

The `index` field is a sympy expression encoding the access pattern — e.g.,
`8192*d0 + d1` for row-major access into a `[2048, 8192]` buffer. The
`var_names` and `size` fields define the iteration domain. Together they
fully describe which elements a node reads or writes and in what order.

This is the raw material for any fusion analysis. When `can_fuse` checks
"compatible iteration space," it is comparing these fields across nodes.
When extending fusion to new patterns, the data is already here — the
question is whether the analysis recognizes the compatibility.

Other dependency types:
- `StarDep` — entire buffer, unknown access pattern (prevents fusion analysis)
- `WeakDep` — ordering only, no data dependency

## Fusion Patterns

**Vertical (Producer-Consumer)**:
```python
x.relu().add(1)
# Fused: single kernel, no intermediate buffer materialized
```

**Horizontal (Consumer-Consumer)**:
```python
y1 = x.relu(); y2 = x.sigmoid()
# Fused: load x once, compute both outputs
```

**Reduction + Epilogue**:
```python
mean = x.mean(keepdim=True); norm = x - mean
# Fused: reduction pass computes mean, then pointwise pass subtracts it
# Enabled by DisableReduction/EnableReduction schedule markers (see CODEGEN.md)
```

**Nested Reduction**:
```python
# Two dependent reductions over the same elements at different granularity
# e.g., layernorm (per-row) + block amax (per-block-of-rows)
# Detected by NestedReduction, fused into FusedNestedReductions node
```

## Nested Reduction Subsystem

When a reduction's output feeds another reduction at a different
granularity, the standard `(numel, rnumel)` match is insufficient. The
nested reduction subsystem handles this:

### Scheduler side (`scheduler.py`)

- **`NestedReduction`** — detects when an outer reduction and a dependent
  grouped reduction can fuse into one kernel. Analyzes the relationship
  between the two reduction domains and determines how to map one onto the
  other. Contains the dependent-pair detection logic called from `can_fuse`.

- **`FusedNestedReductions(FusedSchedulerNode)`** — the scheduler node for
  two dependent reductions over the same logical elements. Once created by
  the scheduler, it is handed to codegen as a single unit.

### Codegen side (`codegen/simd.py`)

The codegen machinery that executes a `FusedNestedReductions` plan is
described in [CODEGEN.md](CODEGEN.md#nested-reduction-codegen). The key
classes are `_GroupedReductionLayout` (tile geometry),
`_DerivedIterationFamily` (derived iteration space), and
`_PointwiseRemapHandler` (index remapping).

## Scheduler Node Types

The scheduler wraps IR nodes in its own type hierarchy for fusion tracking:

```
BaseSchedulerNode
├── SchedulerNode              — single IR operation
├── FusedSchedulerNode         — multiple operations fused into one kernel
│   └── FusedNestedReductions  — two dependent reductions fused
├── ExternKernelSchedulerNode  — external call (matmul, conv)
├── NopKernelSchedulerNode     — eliminated operation
└── ForeachKernelSchedulerNode — multi-tensor operation
```

## Fusion to Codegen Dispatch

Once fusion decisions are made, the scheduler drives code generation. Fusion
owns both the decision (above) AND the dispatch — routing each fused node
group to the appropriate codegen handler.

### The Dispatch Loop

`Scheduler._codegen()` iterates the fused schedule in order and routes each
node to its backend handler based on type:

```
for node in nodes:
    enter_context(node)              # device guards, stream switching
    buffer_names_to_free.update(node.last_usage)

    if node.is_template():           → codegen_template()
    elif node.is_extern():           → codegen_extern_call()
    elif node.is_foreach():          → codegen_combo_kernel()
    elif FusedNestedReductions:      → codegen_nested_reduction()
    elif FusedMixOrderReductions:    → codegen_mix_order_reduction()
    elif FusedSchedulerNode/
         SchedulerNode:              → backend.codegen_node()   # main SIMD path
    else:                            → node.mark_run()          # NopKernel
```

The type-based routing pattern is stable. Specific node types may be added or
removed, but the architecture — iterate schedule order, dispatch by type to
the registered backend handler — is fundamental.

### Buffer Lifetime Handoff

Before codegen begins, `compute_last_usage()` walks the schedule in reverse
to determine when each buffer is last read. During the dispatch loop,
`buffer_names_to_free` is updated per node, and `codegen_free()` emits
`FreeIfNotReusedLine` into the wrapper. This is the scheduler's contract with
memory planning — see [MEMORY-PLANNING.md](MEMORY-PLANNING.md).

### Node Schedule Construction

For SIMD nodes (the common path), `generate_node_schedule()` creates a
structured schedule that interleaves `SchedulerNode` objects with control flow
markers. This is where the fusion decisions from Part 1 become a concrete
execution plan.

Nodes are classified into two categories:

- **Main body**: Fits inside the reduction loop
  (`node_numel == numel and node_rnumel == rnumel`, or the node's entire
  iteration space is `numel * rnumel` with `rnumel == 1`).

- **Epilogue**: Pointwise operations that consume reduction output
  (`node_numel == numel and node_rnumel == 1 and rnumel != 1`). These get
  bracketed by `DisableReduction`/`EnableReduction` markers.

The marker system is how reduction + epilogue fusion (decided by `can_fuse`
above) becomes executable: the scheduler decides the fusion is legal, then
`generate_node_schedule()` arranges the nodes so the reduction runs first,
markers flush the loop, and the epilogue runs as pointwise code consuming
the reduction result — all in one kernel.

### The Bridge to Kernel Codegen

The SIMD path flows through a chain of methods that progressively narrow
from "fused node group" to "kernel object being populated":

```
codegen_node(node)
  unwrap FusedSchedulerNode → list of SchedulerNodes
  → _codegen_nodes(nodes)
      determine (numel, rnumel) from the reduction node
      generate_node_schedule() → structured schedule with markers
      → codegen_node_schedule(features)
          compute tiling
          create kernel object (e.g. TritonKernel)
          codegen_node_schedule_with_kernel() → traces nodes into kernel
          kernel.codegen_kernel() → generates source string
          define_kernel() → assigns name, writes to wrapper
          mark_run() → allocates output buffers
          call_kernel() → emits kernel launch into wrapper
```

`codegen_node_schedule_with_kernel()` drives the two-pass codegen mechanism
described in [CODEGEN.md — Two-Pass Schedule Codegen](CODEGEN.md#two-pass-schedule-codegen).

### mark_run and Output Allocation

After kernel source is generated, `mark_run()` calls `buf.allocate()` for
each output buffer. This emits `AllocateLine` into the wrapper — the starting
point for memory planning (see
[MEMORY-PLANNING.md](MEMORY-PLANNING.md)). Outputs are allocated after source
generation but before the kernel call is emitted, ensuring buffers exist when
the kernel runs.

## Extending Fusion

The sections above (and [CODEGEN.md](CODEGEN.md)) describe how fusion *works*. This
section is how to *add* a new fusion pattern — one the existing `can_fuse` branches don't
yet recognize.

### One decision, two replays

A lowering's `inner_fn` (Loop-Level IR) is replayed under different `ops` handlers: once
under the analysis handler to extract `MemoryDep`s (what the scheduler reasons over), and
again under a backend handler (`TritonKernelOverrides`) to emit code. So any new
relationship your fusion introduces exists **twice** — as a dependency fact used for
legality/scoring, and as an emission. The two must agree; when they can drift, keep a
**single source of truth** — one pure formula, or one planner-proved fact both sides
consume — or you get silently wrong kernels.

### What to touch, by concern

Fusion is spread across the pipeline **by design** — each layer owns one concern and
constrains the next: `scheduler.py` (legality/scoring) → `codegen/simd.py` (iteration
domain / tiling) → `codegen/triton.py` (emission) → `codegen/common.py` + `memory.py`
(buffer accounting). New fusions are categorized by *which* concern they stress. Name the
concern, then the code:

- **"These two accesses are the same value" isn't recognized.** Prove the equivalence and
  hand `SIMDScheduling.can_fuse()` the set of buffer names it may treat as index-equivalent
  (a planner-proved set, e.g. `index_equivalent_dep_names`) — relax **only `MemoryDep`s**,
  never `StarDep`/`WeakDep`. Normalize both indices to a common domain
  (`MemoryDep.normalize_with_ranges`) and compare with
  `V.graph.sizevars.statically_known_equals` — match on full `ranges`, **not numel**
  (different domains share a numel; a numel-only check silently misclassifies, sometimes into
  a *numerically wrong* fusion). Keep legality pure — no cost; don't smuggle a score-time
  boolean into the shared predicate (an early `allow_index_equivalence` flag was deleted in
  favor of the proved set precisely so scoring reuses the fully-legal relation).
- **Two nodes need a new fused-node shape.** The default `FusedSchedulerNode` assumes one
  iteration group. Two *different* stacked iteration spaces (e.g. a reduction and a finer
  consumer) need a new subclass — `FusedNestedReductions` is the example — with its own
  `can_fuse_with`/`fuse_with` and group metadata, plus a branch in
  `SIMDScheduling.can_fuse()` and a dispatch arm in `Scheduler._codegen()` (see
  [Fusion to Codegen Dispatch](#fusion-to-codegen-dispatch)).
- **The iteration spaces don't match.** Resolve it in the iteration-domain layer
  (`codegen/simd.py`): reorder tree dims, reindex the flat space, or build a derived family
  (`_DerivedIterationFamily`) from the parent range tree — see
  [CODEGEN.md — Range Trees](CODEGEN.md#range-trees) and
  [Nested Reduction Codegen](CODEGEN.md#nested-reduction-codegen). Keep index-width / flops /
  bytes analysis pointed at the **combined** accesses (thread an `indexing_node_schedule`
  through the kernel-features object, e.g. `SIMDKernelFeatures`) so analysis matches what
  you'll emit. Split math that pads/derives sub-blocks typically needs a **static,
  power-of-two** parent extent (`is_power_of_2(...)`) so persistent blocks split safely.
- **The value handoff needs a remap.** Wrap the ops handler (a `WrapperHandler` such as
  `_PointwiseRemapHandler` / `_SubParentSourceLoadResolver`) so the consumer's `load`
  resolves through `store_cache` to the producer's in-register value under the remapped
  index instead of emitting `tl.load` — see
  [CODEGEN.md — CSE](CODEGEN.md#cse-common-subexpression-elimination). Emit any re-tiling
  with `tl.reshape` / `tl.split` / `tl.permute`; `tl.split` is factor-2 only, so **recurse
  for factors > 2** (factor-2 is the base case). Hoist loop-invariant shape constants used by
  derived families **once** to function scope via a conflict-detecting named-constant emitter
  (e.g. `_codegen_named_constant`) so every family references an identical definition.
- **Buffer accounting changes.** A register handoff asserts the intermediate need not be
  materialized; kernel-local buffer removal must reflect that or you leak/double-free — see
  [MEMORY-PLANNING.md](MEMORY-PLANNING.md). When one buffer is stored/forwarded across
  multiple lane-groups, track **counts, not just names** (`store_buffer_counts` alongside
  `store_buffer_names`, decremented in `remove_kernel_local_buffers`) so removal stays
  correct.
- **Gating.** Put every new path behind one `config.triton.<feature>` flag (e.g.
  `config.triton.nested_reduction`), checked through a helper (e.g. `_is_enabled_for(n1, n2)`)
  rather than the raw flag, so it applies only on the intended path and tests can flip it to
  generate a reference.

### Keep legality separate from profitability

Legality decides *whether the fusion is valid*; profitability decides *whether it's worth
it*. Never let a cheap score-time relaxation leak into legality. If your fusion is legal but
sometimes slower, add a guard — see [FUSION-PROFITABILITY.md](FUSION-PROFITABILITY.md).

### Land it incrementally

Build the most restrictive vertical slice first (one factor/layout/path, end-to-end and
correct), then generalize the quantitative axis (factor, lanes, staging), then add
qualitative variants (a second layout), then clean up. Make capability ceilings explicit
constants so raising one is a one-line, test-guarded change.

### Testing (three altitudes)

1. **Predicate/formula unit tests — mocked, no compile.** Construct
   `MemoryDep`/`StarDep`/`WeakDep` directly, call scheduler predicates with
   `Scheduler.__new__(Scheduler)` + mocked attrs, and test any shared formula in isolation.
   Assert relaxation touches only `MemoryDep`s.
2. **Fusion-decision level.** `metrics.generated_kernel_count`, a feature-specific counter,
   and explicit fallback assertions.
3. **Emitted-kernel form + numerics.** `run_and_get_code` + `FileCheck` on the Triton
   (`tl.split` count, `tl.permute`, looped-vs-persistent via `for r0_offset in tl.range`),
   plus **numeric equality against the same function compiled with the flag off**. Pick
   inputs where a *misclassification* yields a numeric mismatch, not just a lost fusion.
   Give every rejection path its own `*_rejects_*` test, including each interacting config.

## Key Files

- `scheduler.py` — `can_fuse` scoring, `NestedReduction`,
  `FusedNestedReductions`, `_codegen()` dispatch loop, `compute_last_usage()`
- `dependencies.py` — `MemoryDep`, `StarDep`, `WeakDep`, dependency analysis
- `codegen/simd.py` — `SIMDScheduling.can_fuse()`, `codegen_node()`,
  `generate_node_schedule()`, `codegen_node_schedule()`,
  `codegen_node_schedule_with_kernel()`

---

**For codegen internals** (kernel lifecycle, two-pass mechanism, CSE):
[CODEGEN.md](CODEGEN.md)
**For memory planning** (buffer reuse, allocation pools):
[MEMORY-PLANNING.md](MEMORY-PLANNING.md)
**For guarding a fusion for profitability** (cost modeling, coalescing):
[FUSION-PROFITABILITY.md](FUSION-PROFITABILITY.md)
**For architecture overview**: [ARCHITECTURE.md](ARCHITECTURE.md)
