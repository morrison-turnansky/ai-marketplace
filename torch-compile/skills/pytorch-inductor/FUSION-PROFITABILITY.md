# Guarding an Inductor Fusion for Profitability

How to keep a *legal* fusion from firing when it would make things slower — by modeling the
fused kernel's cost and rejecting the transform when it regresses, instead of reaching for a
blunt kill-switch.

> **Worked example / further reading:** the "guard loop reindexing with selected-tiling
> coalescing costs" ghstack: PRs #191349 → #191766,
> motivated by regression #189488. Loop *reindexing* reshapes a pointwise node's iteration
> space to match a reduction's split (e.g. `[1024, 8192] -> [65536, 128]`) so the two fuse.
> On some graphs this turned a *coalesced* memory access into an *uncoalesced* one, a ~6x
> regression whose only prior workaround was an env kill-switch. The stack replaces the
> kill-switch with a profitability guard. Class/method names below come from that stack;
> treat them as a naming template, not a stable API.
>
> For adding a brand-new fusion (what to touch and why when extending fusion), see
> [FUSION.md — Extending Fusion](FUSION.md#extending-fusion).

## When you need this

- A fusion is **legal and usually good, but sometimes regresses** performance
  (memory-coalescing, occupancy, register pressure).
- The current mitigation is a **kill-switch / env flag** — the smell that says "we know it's
  sometimes bad but can't tell which times."
- You can express the trade-off as an **absolute, comparable cost** you can compute for the
  fused kernel and for the un-fused kernels separately.

If the transform is sometimes *incorrect* (not merely slower), that's a legality problem —
fix it in the legality checks (see [FUSION.md](FUSION.md)), not here.

## Where this sits in the pipeline

Read [FUSION.md](FUSION.md) and [CODEGEN.md](CODEGEN.md) first. A profitability guard hooks
into the fusion decision (alongside `SIMDScheduling.can_fuse()` / the scheduler's reorder
path) but its cost number is read from the codegen tiling machinery: the guard prices a
candidate by asking the tiling selector how the fused kernel's **range trees**
([CODEGEN.md](CODEGEN.md#range-trees)) would be tiled and what memory traffic results. It
leaves *legality* and *emission* untouched — a profitability gate never changes what is
legal or what is emitted, only whether a legal transform is taken. The materialization
saving it credits comes from fused intermediates becoming kernel-local
([MEMORY-PLANNING.md](MEMORY-PLANNING.md)). So this is one concern layered onto the fusion
machinery in [FUSION.md](FUSION.md#extending-fusion), not a separate system.

## Legality vs profitability — where the guard hooks in

The guard sits **downstream of legality and does not touch it**. Legality checks
(`is_compatible`, dependency/cycle checks, non-CPU, non-foreach) decide *whether the
transform is valid*; the guard decides *whether it's worth it*.

Consequences for placement:

- **Ride the existing feature flag.** The guard doesn't add a new user flag — it makes the
  existing feature safe, so the old kill-switch can be removed. It's reached only after the
  normal fusion score already came up short (the reorder/reindex path only runs when the
  ordinary shared-data score is below threshold).
- **Guard the speculative mutation, not the boolean `can_fuse`.** The transform is applied
  the same way it always was; the guard is inserted *after* the mutation and its correctness
  checks, immediately before returning success — and rejects by rolling back.

## Model an absolute, cross-kernel-comparable cost

The whole guard rests on one number: a cost you can compute for a set of nodes wrapped as a
single kernel, such that `cost(n1) + cost(n2)` is meaningfully comparable to
`cost(fused(n1, n2))`.

The worked example models memory-access coalescing:

- **Snapshot the absolute number before any intra-kernel ranking penalty.** Tiling selection
  already scores candidate tilings, but those scores are penalized for small blocks and are
  only comparable *within* one kernel. Capture the raw coalesced bytes *before* the penalty
  loop runs, so ranking penalties don't corrupt the absolute figure.

  ```python
  coalesced_memory = sum(split_scores)   # captured BEFORE the small-block penalty mutates split_scores
  ```

- **Derive the complement from a tiling-independent total.** Total traffic doesn't depend on
  tiling, so uncoalesced = `total - coalesced`:

  ```python
  total_memory = sum(coalesce_analysis.coalesced_by_var.values()) + total_uncoalesced
  MemoryCoalescing(
      coalesced=coalesced_memory,
      uncoalesced=max(total_memory - coalesced_memory, 0),
  )
  ```

- **Weight the penalty and expose one scalar.** An uncoalesced access is modeled as ~16x a
  coalesced one (writes already weighted 2x upstream):

  ```python
  def weighted_cost(self):
      return self.coalesced + _UNCOALESCED_MEMORY_COST_WEIGHT * self.uncoalesced  # weight = 16
  ```

## Price with the real selector, not a bespoke model

Do not invent a parallel cost model — **ask the actual tiling selector what codegen would
pick, then report the cost under that tiling.** A dimension that *could* coalesce but isn't
chosen by the selected tiling counts as uncoalesced, because that's what codegen will
actually emit. This is the profitability analogue of the authoring rule "planner and codegen
share one formula": the guard and codegen share one tiling decision.

```python
def _selected_tiling_memory(self, nodes) -> Optional[MemoryCoalescing]:
    snodes = [sn for node in nodes for sn in node.get_nodes()]
    analysis = analyze_memory_coalescing_for_nodes(snodes)
    if analysis is None:
        return None
    reduction = max(snodes, key=lambda n: int(n.is_reduction()))
    _, (numel, rnumel) = reduction.group
    return SIMDScheduling.select_tiling_with_memory(snodes, numel, rnumel, analysis).memory
```

Because this reflects the real tiling, the guard is only valid when tiling selection is in
its normal mode — so it no-ops when the analysis is off or an alternate tiling path is
active (see the `None` rule next).

## The `None` rule: unknown means "don't guard"

**A profitability guard must never block a fusion it couldn't price.** Every place the model
doesn't apply returns `None`, and the caller treats `None` as "don't reject":

```python
if unfused_memory is None:            # couldn't price the baseline
    return False                      # -> do not reject the fusion
fused_memory = self._selected_tiling_memory([node1, node2])
if fused_memory is None:              # couldn't price the fused kernel
    return False
```

`_selected_tiling_memory` returns `None` when: no nodes; the coalescing analysis config is
off; an alternate tiling mode is active; any foreach node; any non-standard scheduler node;
any CPU node; or the analysis is otherwise unavailable. Erring toward "don't guard" keeps the
guard from ever being the thing that silently kills a good fusion.

## The comparison: combined baseline vs fused, with a launch credit

The accept/reject decision is literally `cost(n1) + cost(n2)` versus `cost(fused)`, made
fair to fusion in two ways:

```python
unfused_cost = sum(m.weighted_cost() for m in unfused_memory)   # cost(n1) + cost(n2)
fused_cost   = fused_memory.weighted_cost()                     # cost(fused(n1, n2))

dram_gbps = get_gpu_dram_gbps()
if math.isfinite(dram_gbps) and dram_gbps > 0:
    # GB/s is numerically bytes/ns, so cost/gbps converts byte-cost to nanoseconds,
    # letting a fixed-ns launch credit be added coherently.
    unfused_time = unfused_cost / dram_gbps + _REINDEXING_FUSION_LAUNCH_OVERHEAD_NS  # ~1000ns
    fused_time   = fused_cost   / dram_gbps
    regresses = fused_time > unfused_time
else:
    regresses = fused_cost > unfused_cost   # bandwidth unknown -> stricter byte comparison, no credit
```

- **Launch-overhead credit.** The two-kernel baseline gets `+~1000ns` because fusing saves a
  kernel launch — the fused kernel is allowed to be slightly worse on pure traffic and still
  win.
- **Bytes → time via bandwidth.** Converting to nanoseconds is what makes the fixed-time
  launch credit coherent. If the bandwidth lookup is invalid or throws, fall back to a raw
  byte comparison with no credit (stricter and safe).
- **Materialization credit is automatic.** The fused analysis omits removable intermediate
  buffers, so traffic saved by not materializing the intermediate already shows up as a lower
  `fused_cost`; a coalescing regression must overcome that saving to be rejected.

## Measure → mutate → price → rollback

The transform is destructive, so ordering matters:

1. **Measure the baseline first**, before any loop mutation — price each node separately
   while its loops are still un-reindexed.
2. **Snapshot loop state**, then apply the transform (reorder and/or reindex).
3. **Price the fused kernel** and compare.
4. **Roll back on reject** so the fusion scorer sees the transform as if it never happened.

```python
unfused_memory = tuple(self._selected_tiling_memory([n]) for n in (node1, node2))
if any(m is None for m in unfused_memory):
    unfused_memory = None
rollback = _LoopStateSnapshot.create(...)
apply_loop_reindexing([red_numel, red_rnumel])          # (+ apply_new_loop_order first, if reordering)
if self._reindexing_regresses_memory_coalescing(node1, node2, unfused_memory):
    rollback.restore()
    return False
```

Reliable speculate-then-undo requires **robust rollback infrastructure**. The worked example
hardens it: the snapshot walks fused-node children *and* the fused node's own group metadata;
the mutation tracker uses **chained listeners** so recursive `can_fuse()` calls each capture
their own decision boundary while outer scopes still observe nested mutations; and every
other loop-mutating path (e.g. dimension expansion) is routed through the same tracker so it
too is rollback-able.

## Testing (three altitudes)

1. **Cost-model arithmetic — mocked, no GPU.**
   - `weighted_cost`: assert the 16x weight and that any uncoalesced beats coalesced.
   - **Launch-credit boundary** via `@parametrize` over `(fused_cost, reject)` triples with
     `get_gpu_dram_gbps` mocked — pin the exact accept/reject cutoff
     (e.g. baseline `100M` at `1000 GB/s` → cutoff at `101,000,000`).
   - **Invalid/failed bandwidth** (`None, 0, -1, nan, inf`, and a lookup that raises) → assert
     the byte-cost fallback still rejects a costlier fusion.
   - **Tiling faithfulness**: assert
     `memory.coalesced + memory.uncoalesced == sum(coalesced_by_var) + sum(uncoalesced_addrs)`,
     and that a coalesce-able-but-not-selected var counts as uncoalesced.

2. **Decision level — patch the hook, record keyed by origin.** The reusable trick for
   asserting "this specific fusion was rejected for this specific reason": patch the guard
   (and the tiling selector) to record `(origins, reject, selection)` per call, then assert
   that for a given aten origin (e.g. `var_mean`) the guard rejected with
   `selection.memory.uncoalesced > 0`. This is far more precise than a kernel count.

3. **Observable outcome — `metrics.generated_kernel_count`.** The regression repro
   (#189488) asserts the bad fusion is now rejected; a positive case asserts a good reindex
   is *accepted* with the expected tiling and `generated_kernel_count == 1`; a forced-reject
   case asserts kernels stay split (`== 2`) and that rollback restored each node's snapshot
   exactly.

## How the stack is decomposed

**Land the safety mechanism first, then the aggressive capability on top.**

- **Bottom PR — the guard.** The whole cost model, the two scheduler methods
  (`_selected_tiling_memory`, `_reindexing_regresses_memory_coalescing`), the
  measure/mutate/price/rollback wiring, the launch/bandwidth model, and the rollback
  hardening. Self-contained: reindexing that regresses coalescing is now rejected, and the
  kill-switch can go.
- **Top PR — extend what can be reindexed.** A new class of reindex (square-block broadcast
  reorder) that would previously be rejected as incompatible — made safe *precisely because
  the guard underneath rejects the cases where it hurts.* It measures the baseline before
  reordering and reuses the same rollback snapshot to undo both the reorder and the reindex.

The general pattern: **the lower PR makes the feature safe; the upper PR grows its reach.**
Expansion can be bold when a proven guard sits beneath it.
