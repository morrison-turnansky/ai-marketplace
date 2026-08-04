# Memory Planning

How Inductor optimizes buffer allocation and reuse after fusion and scheduling
have determined which buffers actually need to exist.

Memory planning runs **after** the scheduler has fused nodes and computed buffer
lifetimes. It operates only on buffers that survived fusion — inline
rematerialization and dead code elimination have already removed intermediates
that never materialize. Two distinct modes handle inference vs training
workloads.

**Prerequisites**: [ARCHITECTURE.md](ARCHITECTURE.md) (IR hierarchy, buffer
types), [FUSION.md](FUSION.md) (how fusion eliminates intermediates),
[CODEGEN.md](CODEGEN.md) (wrapper codegen, WrapperLine types).

---

## Table of Contents

1. [Buffer Lifetime Analysis](#buffer-lifetime-analysis)
2. [How Fusion Reduces the Planner's Workload](#how-fusion-reduces-the-planners-workload)
3. [Rematerialize-by-Default Strategy](#rematerialize-by-default-strategy)
4. [Pool-Based Planning (Inference)](#pool-based-planning-inference)
5. [Simple Reuse Planning (Training)](#simple-reuse-planning-training)
6. [Node Reordering for Peak Memory](#node-reordering-for-peak-memory)
7. [Wrapper Integration](#wrapper-integration)

---

## Buffer Lifetime Analysis

The scheduler determines when each buffer is last read, which tells memory
planning when buffers can be freed or reused.

### compute_last_usage()

**File:** `scheduler.py`

The scheduler walks the node list in **reverse order**, tracking which
buffers are still needed by future nodes:

```python
def compute_last_usage(self):
    future_used_buffers = OrderedSet(V.graph.get_output_names())
    for node in reversed(self.nodes):
        node.set_last_usage(future_used_buffers, self.mutation_real_name)
        future_used_buffers.update(node.last_usage)
```

A buffer name appears in a node's `last_usage` when that node is the **last
one that reads it**. During the codegen dispatch loop (see
[FUSION.md — Fusion to Codegen Dispatch](FUSION.md#fusion-to-codegen-dispatch)),
`buffer_names_to_free.update(node.last_usage)` triggers `codegen_free()` calls
that emit `FreeIfNotReusedLine` into the wrapper.

### Dependency Types and Lifetime

Three dependency types from `dependencies.py` affect lifetime tracking
differently:

- **`MemoryDep`** — A specific indexed access to a named buffer. Used for both
  fusion analysis AND lifetime tracking. The buffer stays alive until the last
  `MemoryDep` referencing it is scheduled.

- **`StarDep`** — Depends on the entire buffer (unknown access pattern). Used
  for mutations and operations that access a buffer unpredictably. Prevents
  fine-grained fusion analysis but still counts for lifetime.

- **`WeakDep`** — Ordering-only dependency. When `is_fake=True`, it is
  **excluded** from lifetime tracking — prevents phantom dependencies from
  artificially extending buffer lifetimes while still ensuring correct
  execution order.

---

## How Fusion Reduces the Planner's Workload

Fusion eliminates intermediate buffers before memory planning sees them,
through three mechanisms:

### 1. Inline Rematerialization

When a `ComputedBuffer`'s `make_loader()` is called and the buffer has no
reduction, no mutation, and `num_reads() == 0` (first consumer), the
computation is **inlined** into the consumer's kernel rather than materializing
to memory. These buffers never get `AllocateLine` entries in the wrapper.

See [Rematerialize-by-Default Strategy](#rematerialize-by-default-strategy)
below.

### 2. CSE Store Cache Forwarding

When producer-consumer nodes are fused into one kernel, the CSE `store_cache`
maps the producer's output to an in-register value. The consumer loads from
the cache instead of memory — the intermediate `tl.store`/`tl.load` pair is
eliminated entirely. This is what makes fusion actually eliminate memory
traffic, not just co-location in the same kernel. See
[CODEGEN.md — CSE](CODEGEN.md#cse-common-subexpression-elimination).

### 3. Dead Code Elimination

After fusion, intermediate buffers consumed only within the fused kernel have
no external users. The scheduler's `dead_node_elimination()` adds these to
`V.graph.removed_buffers`. Memory planning's `drop_removed_buffers()` replaces
their alloc/free lines with `NullLine`.

**Net effect**: Fusion eliminates intermediate buffers → fewer buffers reach
memory planning → smaller allocation pools → lower peak memory.

---

## Rematerialize-by-Default Strategy

Inductor trades compute for memory by default. A buffer is inlined
(recomputed in each consumer) rather than materialized unless it **must** be
stored.

### The make_loader() Decision

**File:** `ir.py` — `ComputedBuffer.make_loader()`

```python
def make_loader(self):
    if (
        not self.get_reduction_type()
        and self.name not in V.graph.mutated_buffers
        and self.num_reads() == 0
        and not self._force_realize
    ):
        return self.data.make_loader()  # inline: recompute instead of loading
    return super().make_loader()        # materialize: generate ops.load()
```

A buffer is **realized** (materialized to memory) when any of these hold:
- It has a reduction (`get_reduction_type()` is not None)
- It is mutated (`name in V.graph.mutated_buffers`)
- It already has readers (`num_reads() > 0` — a second consumer forces realization)
- `_force_realize` is True
- It has exceeded `max_reads` threshold

### Impact on Memory Planning

Inlined buffers never appear as allocations in the wrapper code. Memory
planning only handles buffers that were actually realized. This means the
combination of fusion + rematerialization significantly reduces the number of
buffers memory planning must manage.

The trade-off is explicit: rematerialized buffers don't consume memory but
their computation runs once per consumer. A buffer that is realized consumes
memory but is computed only once.

---

## Pool-Based Planning (Inference)

When `config.memory_planning` is enabled (inference mode), memory planning
packs multiple buffers into large allocation pools, sub-allocating at offsets.

**File:** `codegen/memory_planning.py`

### The Planning Pipeline

`MemoryPlanner.plan()` runs a five-step pipeline:

```
1. drop_removed_buffers()    — NullLine for fused-away buffers
2. convert_to_pool_lines()   — AllocateLine → AllocFromPoolLine
3. compute_live_ranges()     — walk lines to assign timesteps
4. allocate_groups()         — pack BufferGroups into AllocationPools
5. mark_first_last_usage()   — mark pool create/destroy points
```

### AllocationPool

An `AllocationPool` represents a single large `torch.empty` call from which
multiple buffers are sub-allocated at byte offsets:

```python
class AllocationPool:
    device: torch.device
    root: TemporalSplit       # tree of time-packed allocations
    can_expand: bool          # can this pool grow to fit new buffers?
    name: str | None          # assigned during finalize(), e.g. "pool0"
```

If the pool contains exactly one buffer's worth of data, it allocates with
that buffer's dtype and shape. Otherwise it allocates as `torch.uint8` with
`(total_bytes,)` shape, and individual buffers are sub-allocated via
`alloc_from_pool(pool_name, offset, dtype, shape, stride)`.

### Allocation Tree Hierarchy

The tree uses a **spatial/temporal split** design to pack non-overlapping
buffers into the same memory:

```
AllocationTreeNode                 (abstract base)
├── Allocation                     leaf: one buffer with LiveRange + size
├── TemporalSplit                  children that DON'T overlap in time
│                                  (share the same memory region)
│                                  Size = max of children
├── SpatialSplit                   left + right placed adjacently
│                                  Size = align(left) + right
└── Empty                          unused space placeholder
```

**`TemporalSplit`**: Holds multiple allocations with non-overlapping
`LiveRange`s. Buffers that are temporally disjoint reuse the same spatial
region — buffer A can occupy the same bytes as buffer B if their lifetimes
don't overlap.

**`SpatialSplit`**: Holds a `left` and `right` `TemporalSplit` placed side by
side in memory. When a new buffer doesn't fit temporally in any existing slot,
a `SpatialSplit` partitions the space to accommodate it.

### Packing Strategy

`allocate_groups()` creates `Allocation` objects for each `BufferGroup`, then
sorts and packs them:

- **Intermediates**: sorted **largest first** (descending size, descending
  lifetime). Large buffers get placed first into the tree; smaller buffers
  fill temporal gaps left by large ones.

- **Outputs**: sorted **smallest first** (ascending size, descending lifetime).
  Under `memory_pool="outputs"` or `"combined"`, outputs are packed into their
  own pools or appended to the last pool.

### BufferGroup

Tracks buffers that share storage due to inplace reuse:

```python
class BufferGroup:
    node: BufferLike        # the original buffer
    names: list[str]        # all names sharing this storage (original + reuses)
    is_output: bool         # is any name a graph output?
    live_range: LiveRange   # spans first alloc to last dealloc of any name
```

When a `ReuseLine` is encountered during planning, the reused-as name joins
the same group, extending the group's `live_range`.

### Generated Code (Pool-Based)

```python
pool0 = empty_strided_cuda((total_bytes,), (1,), torch.uint8)
buf0 = alloc_from_pool(pool0, 0, torch.float32, (M, N), (N, 1))
buf1 = alloc_from_pool(pool0, M*N*4, torch.float32, (M,), (1,))
# ... kernels use buf0, buf1 ...
del pool0, buf0, buf1
```

---

## Simple Reuse Planning (Training)

When pool-based planning is disabled (training mode, or
`config.allow_buffer_reuse`), a simpler reuse strategy matches freed buffers
to new allocations by shape and dtype.

**File:** `codegen/wrapper.py` — `memory_plan_reuse()`

### How It Works

The planner walks wrapper lines sequentially. Each `AllocateLine.plan()` checks
if a previously freed buffer with a matching key `(device, dtype, size)` exists
in `MemoryPlanningState`. If so, the allocation is converted to a `ReuseLine`
that aliases or reinterprets the old buffer.

### Smart Reuse Gating

Not all reuse is beneficial. `should_reuse_buffer()` prevents reuse from
increasing peak memory:

```python
def should_reuse_buffer(self, free_line, size):
    if free_line.scheduler_node_index + 1 == self.scheduler_node_index:
        return True  # adjacent nodes: always reuse
    peak_memory_in_range = self.wrapper.estimate_peak.peak_between(free_line, self)
    new_peak_memory = size + peak_memory_in_range
    return new_peak_memory <= overall_peak_memory
```

Uses `EfficientPeakEstimate` backed by a segment tree for O(log n) peak
queries over any range of the schedule. If holding a buffer alive through a
high-memory region would push peak up, it's cheaper to free and re-allocate.

### Generated Code (Simple Reuse)

```python
buf3 = buf0                                   # same shape/dtype: simple alias
buf3 = reinterpret_tensor(buf0, new_size, new_stride, new_offset)  # different layout
del buf0
```

---

## Node Reordering for Peak Memory

**File:** `memory.py`

When `config.reorder_for_peak_memory` is enabled, the scheduler tries multiple
topological sort strategies **before** memory planning and picks the one with
the lowest estimated peak:

- **LPMF** (Least Peak Memory First) — greedy BFS that at each step picks the
  node that stays below current peak (preferring the one that frees the most
  memory), or if none exists, the node that increases peak the least.

- **BFS** — FIFO-based, schedules nodes whose predecessors completed earliest.
  Reduces buffer liveness duration by processing consumers soon after their
  producers.

- **DFS** — visits small-memory nodes first to keep many small buffers from
  being live simultaneously.

`reorder_for_peak_memory()` runs all three plus the baseline ordering,
evaluates peak memory for each, and selects the minimum. The specific
algorithms may evolve, but the core design — try multiple valid orderings,
pick the one with lowest peak — is stable.

---

## Wrapper Integration

Memory planning works through the wrapper's **deferred line architecture**.
Instead of emitting code directly, the wrapper appends typed `WrapperLine`
objects to a list. Each line type knows how to render itself:

| Line Type | Generated Code |
|---|---|
| `AllocateLine` | `buf0 = empty_strided_cuda(shape, stride, dtype)` |
| `FreeIfNotReusedLine` | `del buf0` (if not reused; no-op if reused) |
| `ReuseLine` | `buf1 = buf0` or `reinterpret_tensor(buf0, ...)` |
| `AllocFromPoolLine` | `buf0 = alloc_from_pool(pool0, offset, dtype, shape)` |
| `DeallocFromPoolLine` | `del pool0, buf0, buf1, ...` |

Memory planning rewrites `AllocateLine`s into pool-based or reuse-based lines
**after** the full schedule is known. This is why the wrapper uses deferred
lines — allocation decisions depend on the complete set of buffer lifetimes,
which isn't available until all nodes have been codegen'd.

`run_wrapper_ir_passes()` selects the mode:

```python
def run_wrapper_ir_passes(self, is_inference):
    if is_inference and config.memory_planning:
        self.memory_plan()          # Pool-based planning
    else:
        if config.allow_buffer_reuse:
            self.estimate_peak = EfficientPeakEstimate()
        self.memory_plan_reuse()    # Simple reuse planning
```

---

## Key Files

- `codegen/memory_planning.py` — `MemoryPlanner`, `AllocationPool`,
  `AllocationTreeNode` hierarchy, `BufferGroup`
- `codegen/wrapper.py` — `WrapperLine` types, `memory_plan_reuse()`,
  `run_wrapper_ir_passes()`
- `memory.py` — `reorder_for_peak_memory()`, topological sort strategies
  (LPMF, BFS, DFS)
- `scheduler.py` — `compute_last_usage()`, `dead_node_elimination()`
- `dependencies.py` — `MemoryDep`, `StarDep`, `WeakDep`
- `ir.py` — `ComputedBuffer.make_loader()` (rematerialization decision)

---

**For fusion decisions**: [FUSION.md](FUSION.md)
**For codegen internals**: [CODEGEN.md](CODEGEN.md)
**For architecture overview**: [ARCHITECTURE.md](ARCHITECTURE.md)
