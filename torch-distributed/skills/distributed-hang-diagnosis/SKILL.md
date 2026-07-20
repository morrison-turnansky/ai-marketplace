---
name: distributed-hang-diagnosis
description: Reference for PyTorch distributed training hang patterns, NCCL communicator architecture, flight recorder output format, debug logging levels, and timeout configuration. Covers barrier deadlocks, p2p hangs after communicator abort, init timeouts from store, collective mismatches, and DDP unused parameter hangs. Includes ARCHITECTURE.md with source-level ProcessGroupNCCL internals (communicator lifecycle, watchdog, work queue, store-based init). Use as knowledge base for diagnosing why distributed training freezes.
---

# Distributed Hang Reference

Reference documentation for understanding and diagnosing hangs in PyTorch distributed training.

**For ProcessGroupNCCL internals**: See [ARCHITECTURE.md](ARCHITECTURE.md) — communicator lifecycle, watchdog architecture, work queue system, store-based init, and key source-level function references.

## Hang Type Classification

| Type | Description | Common Cause |
|------|-------------|--------------|
| `barrier_deadlock` | `dist.barrier()` hangs with NCCL, works with Gloo | Lazy communicator init races with rendezvous config |
| `p2p_hang_after_abort` | Ranks hang on send/recv after another rank exits | Communicator destroyed while p2p ops pending; watchdog doesn't fire |
| `init_timeout` | `init_process_group` fails with store timeout | Store timeout (5 min) < NCCL timeout (10 min); rank 0 straggler |
| `collective_mismatch` | Ranks call different collectives at same sequence point | Conditional logic varies by rank, uneven batches |
| `nccl_timeout` | NCCL operation exceeds watchdog timeout | Network failure, GPU error, slow rank |
| `unused_parameter` | DDP hang during backward pass | `find_unused_parameters=False` with conditional modules |
| `forward_order_violation` | FSDP hang during forward pass | Dynamic model architecture across iterations |

## NCCL Communicator Architecture

NCCL communicators are initialized **lazily** — the first collective operation triggers setup:

1. Rank 0 generates a unique NCCL ID (`ncclUniqueId`)
2. Rank 0 publishes the ID to the key-value store (`TCPStore` or `FileStore`)
3. Other ranks read the ID from the store via `store->get()`
4. All ranks call `ncclCommInitRank()` with the shared ID
5. NCCL creates internal communication channels (rings, trees)

`dist.barrier()` is implemented as `ncclAllReduce` — it is a collective, not a special operation. If barrier is the first collective called, it triggers the full communicator init sequence above.

### Timeout Layers

```
Application: dist.init_process_group(timeout=600s)
    ↓
Store: TCPStore timeout (default 300s / 5 min)
    ↓
NCCL watchdog: opTimeout_ (default 600s / 10 min, set via init_process_group(timeout=))
    ↓
NCCL async error handling
```

The store timeout and NCCL watchdog timeout are independent. A store timeout fires before the NCCL watchdog, producing a confusing `Socket Timeout` error rather than the expected `NCCL timeout` error.

## Debug Logging Reference

| Variable | Level | What It Shows |
|----------|-------|---------------|
| `NCCL_DEBUG` | `WARN` | NCCL errors and warnings only (default) |
| `NCCL_DEBUG` | `INFO` | Communicator init, topology detection, ring/tree selection |
| `NCCL_DEBUG` | `TRACE` | Every NCCL kernel launch and completion (very verbose) |
| `TORCH_DISTRIBUTED_DEBUG` | `OFF` | No distributed debug logging (default) |
| `TORCH_DISTRIBUTED_DEBUG` | `INFO` | Process group creation, collective calls |
| `TORCH_DISTRIBUTED_DEBUG` | `DETAIL` | Per-collective logging, tensor shapes, DDP parameter names |
| `NCCL_DEBUG_FILE` | path | Per-process NCCL log file (`%h`=hostname, `%p`=pid) |

### Timeout Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `init_process_group(timeout=timedelta(seconds=600))` | `600s` (10 min) | Sets NCCL watchdog timeout and store timeout together |
| `dist.set_timeout(timedelta(seconds=120))` | — | Change watchdog timeout at runtime without restart |
| `TORCH_NCCL_BLOCKING_WAIT` | `0` | Set `1` to make NCCL errors synchronous (better stack traces) |
| `NCCL_ASYNC_ERROR_HANDLING` | `1` | Async error detection via watchdog |

## NCCL Flight Recorder

The flight recorder is a circular buffer in `ProcessGroupNCCL` that records collective operations. Enabled by default in recent PyTorch.

### Configuration

```bash
# Control buffer size (default ~2000 entries, 0 to disable)
# TORCH_FR_BUFFER_SIZE is the current name; TORCH_NCCL_TRACE_BUFFER_SIZE still works but is deprecated
TORCH_FR_BUFFER_SIZE=2000 torchrun ...
```

### Dumping Traces

```python
# Programmatic dump
traces = torch._C._distributed_c10d._dump_nccl_trace()

# Signal-based dump (add to training script)
import signal, json
import torch.distributed as dist

def dump_traces(signum, frame):
    traces = torch._C._distributed_c10d._dump_nccl_trace()
    rank = dist.get_rank()
    with open(f"/tmp/nccl_trace_rank{rank}.json", "w") as f:
        json.dump(traces, f, indent=2)

signal.signal(signal.SIGUSR1, dump_traces)
```

**Limitation**: Signal handlers do not execute when the process is stuck inside a NCCL kernel (common with p2p hangs). Use `py-spy dump --pid <pid>` or `gdb -p <pid>` as alternatives.

### Output Format

Each flight recorder entry:
```json
{
  "profiling_name": "nccl:all_reduce",
  "state": "completed",
  "time_created_ns": 1234567890,
  "time_started_ns": 1234567900,
  "time_finished_ns": 1234568000,
  "input_sizes": [[1024]],
  "output_sizes": [[1024]],
  "process_group_name": "default",
  "process_group_ranks": [0, 1, 2, 3],
  "collective_seq_id": 42
}
```

### Interpreting Traces

- `state=completed` — operation finished normally
- `state=started` — operation in progress (or hung)
- `state=scheduled` — queued but not started
- `collective_seq_id` — monotonically increasing per process group; compare across ranks to detect mismatches

**Diagnosis patterns:**
- All ranks show `state=started` for same op → **nccl_timeout** (network/GPU issue)
- Ranks show different ops at same `collective_seq_id` → **collective_mismatch**
- Some ranks have more entries than others → **asymmetric_control_flow**

## Monitored Barrier

Unlike regular `barrier()`, `monitored_barrier` reports which ranks failed to arrive:

```python
from datetime import timedelta
import torch.distributed as dist

dist.monitored_barrier(timeout=timedelta(seconds=30))
# On timeout: "Rank 2 failed to pass barrier within 30 seconds.
#   Ranks that arrived: [0, 1, 3]  Ranks that failed: [2]"
```

Useful for binary-searching the hang location: insert at midpoint, observe which side hangs, move barrier, repeat.

## Collective Sequence Logging

Monkey-patch collectives to log the call sequence per rank:

```python
import functools
import torch.distributed as dist

_log = []

def trace(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _log.append({"rank": dist.get_rank(), "op": fn.__name__, "seq": len(_log)})
        return fn(*args, **kwargs)
    return wrapper

dist.all_reduce = trace(dist.all_reduce)
dist.broadcast = trace(dist.broadcast)
dist.all_gather = trace(dist.all_gather)
dist.reduce_scatter = trace(dist.reduce_scatter)
dist.barrier = trace(dist.barrier)
```

Compare sequences across ranks to find the divergence point that causes a collective mismatch hang.

## Agent Rationalizations

Agents frequently shortcut the diagnostic process. These are common rationalizations and why they are wrong:

| Rationalization | Why It's Wrong | What To Do Instead |
|----------------|----------------|-------------------|
| "The error message is clear enough, I don't need NCCL logs" | Hang errors are often misleading — a `Socket Timeout` from the store masks an NCCL init race. The visible error is rarely the root cause. | Always collect `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` output before diagnosing. |
| "It's probably a network issue" | Network issues cause NCCL timeouts, not deadlocks. Barrier deadlocks, collective mismatches, and p2p hangs are code bugs, not infrastructure. | Classify the hang type first. Only investigate network after ruling out code-level patterns. |
| "I'll skip the flight recorder, the stack trace is enough" | Stack traces show WHERE the hang is, not WHY. Flight recorder `collective_seq_id` comparison across ranks reveals the actual mismatch. | Dump flight recorder traces and compare across at least 2 ranks before proposing a fix. |
| "This looks like Pattern X, I'll apply the fix directly" | Multiple patterns share symptoms. Barrier deadlock and init timeout both hang at startup. P2p hang and collective mismatch both hang mid-training. | Complete the classification table. Check at least 2 distinguishing observations before matching a pattern. |
| "The fix worked on one rank, we're done" | Distributed bugs are inherently multi-rank. A fix that resolves rank 0's symptom may shift the hang to another rank or create a new mismatch. | Verify ALL ranks complete the operation. Check `NCCL_DEBUG=INFO` output shows no warnings on any rank. |
| "I'll just increase the timeout" | Increasing timeout masks the bug — the operation will still hang, just later. Timeouts are a diagnostic tool, not a fix. | Reduce the timeout (`dist.init_process_group("nccl", timeout=timedelta(seconds=120))`) for faster feedback. Fix the root cause. |
| "find_unused_parameters=True fixes DDP hangs" | It fixes unused-parameter hangs but adds overhead and masks model bugs. If parameters are unexpectedly unused, the model may be silently broken. | Investigate which parameters are unused and why. Use `find_unused_parameters=True` only if the unused parameters are intentional. |

## Known Hang Patterns

### Pattern 1: NCCL Barrier Deadlock

**Symptom**: `dist.barrier()` hangs with NCCL backend, works fine with Gloo.

**Root cause**: NCCL barrier is `allReduce`. Lazy communicator init races with certain `torchrun` rendezvous configurations (e.g., `--rdzv-endpoint=localhost:0`).

**Source-level code path** (`ProcessGroupNCCL.cpp`): The lazy init race is in `initNCCLComm()` → `broadcastUniqueNCCLID()` where `store_->get()` blocks waiting for rank 0's `store_->set()`. The fix path is `eagerConnectSingleDevice()` which calls `initNCCLComm()` at construction time when `device_id` is passed. See [ARCHITECTURE.md — Communicator Lifecycle](ARCHITECTURE.md#communicator-lifecycle).

**Fix**:
```python
# Pass device_id to init_process_group (PyTorch 2.4+)
dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))

# Or force eager communicator init before barrier
t = torch.zeros(1, device=f"cuda:{rank}")
dist.all_reduce(t)
dist.barrier()
```

**Note**: Mitigated by passing `device_id` (PyTorch 2.4+). [pytorch/pytorch#129749](https://github.com/pytorch/pytorch/issues/129749) remains open. Newer PyTorch warns: `"using GPU X as device used by this process is currently unknown"` — pass `device_id` to silence the warning and avoid the race.

**Verification evidence** — all of these must be true before the fix is confirmed:
- [ ] All ranks print past the barrier in stdout
- [ ] `NCCL_DEBUG=INFO` shows `ncclCommInitRank` completing on every rank (not just rank 0)
- [ ] No `"device used by this process is currently unknown"` warning in logs
- [ ] Re-run with `dist.init_process_group("nccl", timeout=timedelta(seconds=120))` — completes without timeout

*Reference: [pytorch/pytorch#129749](https://github.com/pytorch/pytorch/issues/129749)*

### Pattern 2: P2P Ops Hang After Communicator Abort

**Symptom**: One rank finishes and destroys its communicator. Other ranks hang on `isend`/`irecv` to that rank. NCCL watchdog does NOT fire.

**Root cause**: Pending p2p operations targeting a destroyed communicator never get scheduled on the NCCL stream, so the watchdog doesn't detect them.

**Source-level code path** (`ProcessGroupNCCL.cpp`): `pointToPoint()` creates a dedicated 2-rank communicator per send/recv pair in lazy mode. If the peer has destroyed its process group, `initNCCLComm()` → `broadcastUniqueNCCLID()` → `store_->get()` blocks forever. The watchdog (`Watchdog::runLoop()`) only checks `workMetaList_` — but `workEnqueue()` is never reached because the hang is inside `initNCCLComm()`, before the work item is created. See [ARCHITECTURE.md — Work Queue & Watchdog](ARCHITECTURE.md#work-queue--watchdog).

**Fix**: All ranks must complete p2p operations and barrier before teardown:
```python
for w in work_handles:
    w.wait()
dist.barrier()
dist.destroy_process_group()
```

For **batched isend/irecv hangs under high load** (a related but distinct issue), upgrading NCCL resolves it:

| NCCL Version | PyTorch Version | Batched P2P Status |
|-------------|----------------|-----|
| 2.29.3 | pre-2.12 or CUDA < 13 | Hang under load |
| 2.29.7 | 2.12+ | Fixed |
| 2.30.7+ | To be integrated | Fixed |

```bash
# Upgrade NCCL on CUDA 12 (if stuck on 2.29.3):
pip install nvidia-nccl-cu12==2.29.7
```

**Note**: Flight recorder signal dump does not work for this pattern — process is stuck in NCCL kernel. Use `py-spy` or `gdb` instead.

**Verification evidence** — all of these must be true before the fix is confirmed:
- [ ] All ranks complete `w.wait()` for every p2p handle
- [ ] `dist.barrier()` completes on all ranks before `destroy_process_group()`
- [ ] `NCCL_DEBUG=INFO` shows no `"abort"` or `"communicator was aborted"` messages
- [ ] `py-spy dump --native` of each rank shows no thread blocked in `ncclCommInitRank` or `store_->get()`

*Reference: [pytorch/pytorch#113281](https://github.com/pytorch/pytorch/issues/113281), [pytorch/pytorch#174288](https://github.com/pytorch/pytorch/issues/174288)*

### Pattern 3: Init Timeout Due to Store

**Symptom**: Non-zero ranks fail with `store->get('0') got error: wait timeout after Xms` while rank 0 is still starting.

**Root cause**: Store timeout (default 5 min) fires before NCCL watchdog timeout (default 10 min). Rank 0 straggler means other ranks block on `store->get()` waiting for the NCCL unique ID.

**Source-level code path** (`ProcessGroupNCCL.cpp`): `broadcastUniqueNCCLID()` is the blocking point — non-zero ranks call `store_->get(storeKey)` which blocks until rank 0 calls `store_->set()`. The `TCPStore` timeout (default 300s) is independent of the NCCL `opTimeout_` (default 600s). When the store timeout is shorter, it fires first with `"retrieving ncclUniqueId from [0] via c10d key-value store"` instead of the expected NCCL timeout. See [ARCHITECTURE.md — Timeout Architecture](ARCHITECTURE.md#timeout-architecture).

**Fix**: Align timeouts:
```python
from datetime import timedelta
dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))
```

**Verification evidence** — all of these must be true before the fix is confirmed:
- [ ] All ranks complete `init_process_group` without timeout
- [ ] Error message (if timeout still occurs) is now `"Watchdog caught collective operation timeout"` (NCCL), not `"Socket Timeout"` (store) — confirming timeouts are aligned
- [ ] `NCCL_DEBUG=INFO` shows `ncclCommInitRank` completing on all ranks

*Reference: [pytorch/pytorch#107177](https://github.com/pytorch/pytorch/issues/107177)*

### Pattern 4: Collective Mismatch

**Symptom**: Hang during training (not at init). Ranks diverge in which collective they call.

**Common causes**:
```python
# Conditional collective — not all ranks enter
if loss > threshold:
    dist.all_reduce(grad)  # WRONG

# Fix: all ranks must participate
dist.all_reduce(grad)

# Uneven batches — some ranks finish dataloader first
for batch in dataloader:  # WRONG
    train_step(batch)

# Fix: DistributedSampler with drop_last
sampler = DistributedSampler(dataset, drop_last=True)
```

**Verification evidence** — all of these must be true before the fix is confirmed:
- [ ] Flight recorder `collective_seq_id` matches across all ranks at the point of hang
- [ ] Collective sequence log (monkey-patch) shows identical op sequence on all ranks
- [ ] Training progresses past the former hang point for at least 10 steps

### Pattern 5: DDP Unused Parameters

**Symptom** (version-dependent):
- **PyTorch < 2.14**: Hang during `loss.backward()` on first iteration — DDP waits for allreduce on gradients that are never computed.
- **PyTorch >= 2.14**: `RuntimeError` on the second iteration's forward pass — DDP detects the mismatch and raises instead of hanging.

**Cause**: `find_unused_parameters=False` (default) but some parameters are not used in forward.

```python
# Hangs (< 2.14) or errors (>= 2.14) if self.optional is skipped
model = DDP(model)

# Fix
model = DDP(model, find_unused_parameters=True)
```

**Verification evidence** — all of these must be true before the fix is confirmed:
- [ ] Training completes multiple iterations without hang or `RuntimeError`
- [ ] `TORCH_DISTRIBUTED_DEBUG=DETAIL` identifies which parameters were unused (check these are intentionally unused)
- [ ] No performance regression from `find_unused_parameters=True` (compare step time before/after)

### Pattern 6: NCCL Timeout

**Symptom**: Intermittent hang during training. Eventually produces `Watchdog caught collective operation timeout` error.

**Root cause**: A collective operation takes longer than the watchdog timeout. Unlike deadlocks, the operation is actually running but blocked — typically by network failure, GPU error, or a slow rank.

**Distinguishing from deadlocks**: Flight recorder shows all ranks with `state=started` for the **same** operation at the **same** `collective_seq_id`. In a deadlock, ranks show different operations or different seq IDs.

**Fix**: Diagnose the underlying infrastructure issue:
```bash
# Check GPU health
nvidia-smi -q | grep -E "ECC|Retired|Temperature"

# Check network (multi-node)
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

# If a single rank is slow, identify it via per-rank timing
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ... 2>&1 | tee debug.log
```

**Verification evidence** — all of these must be true before the fix is confirmed:
- [ ] Flight recorder confirms all ranks were on the same operation (not a mismatch)
- [ ] `nvidia-smi` shows no GPU errors, ECC issues, or thermal throttling
- [ ] Network bandwidth test (`nccl-tests`) shows expected throughput
- [ ] Training completes without timeout after resolving the infrastructure issue

### Pattern 7: FSDP Forward Order Violation

**Symptom**: FSDP hang during forward pass. May produce `forward order violation` error.

**Root cause**: FSDP expects modules to execute in the same order every iteration. Dynamic model architectures (e.g., conditional layers, early exit) violate this assumption.

**Common causes**:
```python
# Dynamic forward — module order changes per iteration
def forward(self, x):
    if self.training and random.random() > 0.5:
        x = self.dropout_layer(x)  # WRONG: not called every iteration
    return self.output(x)

# Fix: use consistent forward order, control behavior inside the module
def forward(self, x):
    x = self.dropout_layer(x)  # always called; dropout handles train/eval internally
    return self.output(x)
```

**Verification evidence** — all of these must be true before the fix is confirmed:
- [ ] FSDP forward pass completes on all ranks for multiple iterations
- [ ] No `forward order violation` error in logs
- [ ] Module execution order is deterministic across iterations (verify with `TORCH_DISTRIBUTED_DEBUG=DETAIL`)

## Minimal Reproducer Template

```python
"""Minimal hang reproducer.
Usage: torchrun --nproc_per_node=2 repro.py
"""
import signal, json, torch
import torch.distributed as dist
from datetime import timedelta

def dump_traces(signum, frame):
    traces = torch._C._distributed_c10d._dump_nccl_trace()
    rank = dist.get_rank()
    with open(f"/tmp/nccl_trace_rank{rank}.json", "w") as f:
        json.dump(traces, f, indent=2)

signal.signal(signal.SIGUSR1, dump_traces)

dist.init_process_group(backend="nccl", timeout=timedelta(seconds=60))
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

print(f"Rank {rank} initialized")

# === Reproduce hang below ===
tensor = torch.ones(10, device=device)
dist.all_reduce(tensor)
print(f"Rank {rank}: all_reduce done, sum={tensor.sum().item()}")
# === End reproduce ===

dist.destroy_process_group()
```
