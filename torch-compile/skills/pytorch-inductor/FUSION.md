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

## Key Files

- `scheduler.py` — `can_fuse` scoring, `NestedReduction`, `FusedNestedReductions`
- `dependencies.py` — `MemoryDep`, `StarDep`, `WeakDep`, dependency analysis
- `codegen/simd.py` — `SIMDScheduling.can_fuse()`, node schedule generation

---

**For codegen internals**: [CODEGEN.md](CODEGEN.md)
**For architecture overview**: [ARCHITECTURE.md](ARCHITECTURE.md)
