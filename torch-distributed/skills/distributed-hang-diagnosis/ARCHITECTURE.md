# ProcessGroupNCCL Architecture

Architectural deep-dive into PyTorch's NCCL process group backend — the C++ runtime behind `dist.init_process_group("nccl")`.

**For hang patterns and debugging**: See [SKILL.md](SKILL.md)

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Core Components](#core-components)
4. [Communicator Lifecycle](#communicator-lifecycle)
5. [Work Queue & Watchdog](#work-queue--watchdog)
6. [Point-to-Point Architecture](#point-to-point-architecture)
7. [Store-Based Initialization](#store-based-initialization)
8. [Timeout Architecture](#timeout-architecture)
9. [Flight Recorder](#flight-recorder)
10. [Data Flow](#data-flow)
11. [Key Files Reference](#key-files-reference)

---

## Overview

`ProcessGroupNCCL` is PyTorch's NCCL backend for distributed communication. It manages NCCL communicators, dispatches collective and point-to-point operations onto CUDA streams, and monitors them via a watchdog thread.

**Location**: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp` (~6000 lines)

**Key Design Philosophy**:
- **Lazy Initialization**: Communicators created on first collective, not at `init_process_group`
- **Async Execution**: Operations run on dedicated NCCL CUDA streams
- **Watchdog Monitoring**: Background thread polls for timeouts and errors
- **Store-Based Rendezvous**: `TCPStore`/`FileStore` exchanges NCCL unique IDs across ranks

---

## Directory Structure

### Core Files

**Main implementation**: `ProcessGroupNCCL.cpp` (runtime), `ProcessGroupNCCL.hpp` (class definitions)

**NCCL utilities**: `NCCLUtils.hpp` (communicator wrapper, error handling), `NCCLUtils.cpp`

**Store layer**: `Store.hpp`, `TCPStore.hpp`, `FileStore.hpp`, `PrefixStore.hpp`

### Python Bindings

**Process group creation**: `torch/distributed/distributed_c10d.py` — `init_process_group()`, `destroy_process_group()`

**Collective wrappers**: `torch/distributed/distributed_c10d.py` — `all_reduce()`, `broadcast()`, `barrier()`, `send()`, `recv()`

---

## Core Components

### 1. ProcessGroupNCCL (ProcessGroupNCCL.hpp)

**Main class** that manages NCCL communicators and dispatches operations.

**Key Members**:
```
ProcessGroupNCCL
├── devNCCLCommMap_          # Cache: device key → NCCLComm (communicator)
├── workMetaList_            # Queue of in-flight work for watchdog
├── completedWorkList_       # Completed work awaiting hook processing
├── ncclStreams_              # Per-device CUDA streams for NCCL ops
├── ncclEvents_              # CUDA events for stream synchronization
├── store_                   # Key-value store for cross-rank rendezvous
├── watchdog_                # Watchdog thread (timeout detection)
├── heartbeatMonitor_        # HeartbeatMonitor thread (liveness)
├── seqCollective_           # Sequence counter for collectives
├── seqP2P_                  # Sequence counter for p2p ops
├── ncclActiveGroupCounter_  # Tracks ncclGroupStart/End nesting
├── eagerInit_               # True when device_id passed to init
├── bound_device_id_         # Device constraint from device_id param
├── opTimeout_               # NCCL operation timeout
├── blockingWait_            # If true, no watchdog thread
└── terminateProcessGroup_   # Shutdown signal
```

**Construction Flow** (from `init_process_group`):
```
Python: dist.init_process_group("nccl", ...)
    ↓
C++: ProcessGroupNCCL constructor
    ↓
1. Initialize store_ (TCPStore or FileStore, wrapped in PrefixStore)
2. Create HeartbeatMonitor thread
3. Create Watchdog thread (unless blockingWait_)
4. If device_id provided → eagerConnectSingleDevice() → initNCCLComm()
5. Otherwise → communicator created lazily on first collective
```

### 2. WorkNCCL (ProcessGroupNCCL.hpp:316)

**Represents a single NCCL operation** tracked by the watchdog.

**Key Members**:
```
WorkNCCL
├── ncclComm_        # The communicator used
├── ncclStartEvent_  # CUDA event: when NCCL kernel starts
├── ncclEndEvent_    # CUDA event: when NCCL kernel ends
├── workStartTime_   # Wall clock when work was created
├── opTimeout_       # Timeout for this operation
├── seq_             # Sequence number
├── opType_          # OpType (ALLREDUCE, SEND, RECV, etc.)
└── device_          # Target CUDA device
```

**Key Methods**:
- `isStarted()` — queries `ncclStartEvent_` on GPU
- `isCompleted()` — queries `ncclEndEvent_` on GPU
- `checkTimeout()` — compares elapsed wall time against `opTimeout_`
- `wait()` — blocks CPU until GPU completion or timeout
- `abort()` — calls `ncclComm_->abort()` on the communicator

### 3. Watchdog (ProcessGroupNCCL.hpp:675)

**Background thread** that polls `workMetaList_` for errors and timeouts.

**Key Methods**:
- `run()` — thread entry point
- `runLoop()` — main polling loop (sleeps `kWatchdogThreadSleepMillis` = 100ms between iterations)
- `checkAndSetRemoteError()` — checks store for errors flagged by other ranks

### 4. HeartbeatMonitor (ProcessGroupNCCL.hpp:595)

**Monitors watchdog liveness**. If the watchdog stops incrementing its heartbeat counter, the monitor dumps debug info and optionally kills the process.

---

## Communicator Lifecycle

### Lazy Init (Default)

Communicators are NOT created at `init_process_group` time. They're created on the first collective:

```
User calls dist.all_reduce(tensor)
    ↓
ProcessGroupNCCL::allreduce()
    ↓
getNCCLComm(deviceKey) → returns nullptr (first call)
    ↓
initNCCLComm(key, device, opType)    [line ~3028]
    ↓
broadcastUniqueNCCLID(&ncclID, ...)  [line ~2852]
    ├── Rank 0: ncclGetUniqueId() → store_->set(storeKey, ncclID)
    └── Other ranks: store_->get(storeKey) → blocks until rank 0 sets it
    ↓
NCCLComm::create(numRanks, rank, ncclID)
    └── ncclCommInitRank()  ← NCCL creates internal rings/trees
    ↓
devNCCLCommMap_[key] = ncclComm  (cached for reuse)
```

**Risk**: If barrier is the first collective and rendezvous hasn't fully resolved device assignments, the store key exchange races with rendezvous completion → deadlock (Pattern 1).

### Eager Init (device_id)

When `device_id` is passed to `init_process_group`:

```
dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    ↓
ProcessGroupNCCL constructor
    ↓
eagerConnectSingleDevice(device)     [line ~1066]
    ↓
initNCCLComm(key, device, ALLREDUCE) [line ~1070]
    ↓
(same store exchange + ncclCommInitRank as above)
    ↓
eagerInit_ = true
```

**Benefit**: Communicator is ready before any user collective, eliminating the lazy init race.

### Communicator Cache

```
devNCCLCommMap_: {
    "cuda:0"    → NCCLComm (collectives)
    "0:1"       → NCCLComm (p2p between rank 0 and 1, lazy mode only)
    "1:2"       → NCCLComm (p2p between rank 1 and 2, lazy mode only)
}
```

`getNCCLComm(key)` (line ~3320) checks the cache. Returns `nullptr` on miss, triggering `initNCCLComm()`.

---

## Work Queue & Watchdog

### Work Lifecycle

```
User calls collective (e.g., all_reduce)
    ↓
Create WorkNCCL object (records workStartTime_, seq_, opType_)
    ↓
Launch NCCL kernel on ncclStream_ (records ncclStartEvent_, ncclEndEvent_)
    ↓
workEnqueue(work)                    [line ~3545]
    └── workMetaList_.emplace_back(*work)
    ↓
Return work handle to user
    ↓
Watchdog::runLoop() polls workMetaList_ every 100ms:
    ├── work.checkAndSetException()  ← check NCCL async errors
    ├── work.checkTimeout()          ← compare elapsed vs opTimeout_
    ├── If completed → move to completedWorkList_
    └── If timed out → set ErrorType::TIMEOUT, trigger desync debug
```

### Watchdog Blind Spot

The watchdog can only detect timeouts for work in `workMetaList_`. If an operation blocks **before** reaching `workEnqueue()` — for example, inside `initNCCLComm()` during store key exchange — the watchdog has nothing to check.

```
pointToPoint() → getNCCLComm() returns nullptr
    ↓
initNCCLComm() → broadcastUniqueNCCLID() → store_->get() BLOCKS FOREVER
    ↓
workEnqueue() is NEVER reached
    ↓
Watchdog sees empty workMetaList_ → no timeout detected
```

This is the root cause of Pattern 2 (p2p hang after communicator abort).

---

## Point-to-Point Architecture

### P2P Communicator Strategy

P2P operations use a different communicator strategy depending on mode:

**Lazy mode (default)**: Creates a dedicated 2-rank communicator for each send/recv pair:
```cpp
// pointToPoint() line ~4164
key = getKeySendRecv(rank_, peer);  // e.g., "0:2"
p2pRank = rank_ <= peer ? 0 : 1;   // 2-rank local rank
ncclComm = getNCCLComm(key);       // nullptr on first call
if (ncclComm == nullptr) {
    ncclComm = initNCCLComm(key, device, opType, p2pRank, ...);
}
```

**Eager mode (device_id passed)**: Reuses the parent collective communicator:
```cpp
// pointToPoint() line ~4126
if (this->eagerInit_) {
    key = getKeyFromDevice(device);  // e.g., "cuda:0"
    ncclComm = getNCCLComm(key);     // already initialized
}
```

### Batched P2P (batch_isend_irecv)

Batched operations wrap multiple send/recv in `ncclGroupStart()`/`ncclGroupEnd()`:
```cpp
ncclGroupStart()           // increment ncclActiveGroupCounter_
    ncclSend(tensor, peer) // communicator created inside group
    ncclRecv(buf, peer)
ncclGroupEnd()             // triggers actual NCCL execution
```

The `ncclActiveGroupCounter_` nesting is tracked to handle communicator creation within groups (line ~3068): if `ncclActiveGroupCounter_ > 0`, NCCL group operations are closed temporarily to ensure the communicator is initialized before any `ncclSend`/`ncclRecv` is issued.

---

## Store-Based Initialization

### NCCL Unique ID Exchange

All communicator creation requires a shared `ncclUniqueId`. The store handles cross-rank exchange:

```
broadcastUniqueNCCLID(ncclID, isSingleP2POp, p2pKey, p2pRank)  [line ~2852]

For collectives:
    storeKey = str(ncclCommCounter_++)     # "0", "1", "2", ...

For single p2p:
    storeKey = p2pKey                      # "rank0:rank1"

Rank 0 (or p2pRank 0):
    store_->set(storeKey, ncclID bytes)    # non-blocking

Other ranks:
    store_->get(storeKey)                  # BLOCKS until key exists or timeout
```

### Store Timeout

`store_->get()` blocks with the store's internal timeout. For `TCPStore`, this defaults to 300 seconds (5 minutes). This is **independent** of the NCCL watchdog timeout.

The error message on timeout (line ~2906):
```
"[rank] is setting up NCCL communicator and retrieving ncclUniqueId from [0]
 via c10d key-value store by key '<key>', but store->get('<key>') got error: "
```

### Scalable Init (Large Scale)

For large process groups (`numRanks > ranksPerRoot`), a multi-root initialization is used:
```
allgatherUniqueNCCLIDs()              [line ~2936]
    Multiple root ranks generate ncclUniqueIds
    All ranks gather IDs via TCPStore multiGet
    ncclCommInitRankScalable() uses multiple roots
```

---

## Timeout Architecture

### Three Independent Timeout Layers

```
Layer 1: Store Timeout
    Where: TCPStore/FileStore internal timeout
    Default: 300s (5 min) for TCPStore
    Fires: During store_->get() in broadcastUniqueNCCLID()
    Error: "Socket Timeout" or "wait timeout after Xms"

Layer 2: NCCL Watchdog Timeout (opTimeout_)
    Where: WorkNCCL::checkTimeout() called by Watchdog::runLoop()
    Default: 600s (10 min), set via dist.init_process_group(timeout=) or dist.set_timeout()
    Fires: For work in workMetaList_ that exceeds timeout
    Error: "Watchdog caught collective operation timeout"

Layer 3: HeartbeatMonitor
    Where: HeartbeatMonitor::runLoop() checks watchdog heartbeat
    Default: heartbeatTimeoutInSec_ (configurable)
    Fires: If watchdog thread stops incrementing heartbeat counter
    Action: Dumps debug info, optionally kills process
```

### Timeout Mismatch Problem

`init_process_group(timeout=X)` sets the NCCL `opTimeout_` but the store may have been created with a **different** timeout (the TCPStore default of 300s). If the store timeout is shorter, it fires first with a confusing error instead of the expected NCCL timeout message.

---

## Flight Recorder

### Implementation

The flight recorder is a circular buffer inside `ProcessGroupNCCL` that records collective operations. Controlled by `TORCH_FR_BUFFER_SIZE` (default ~2000 entries; `TORCH_NCCL_TRACE_BUFFER_SIZE` is the deprecated alias).

### Dump Mechanisms

**Programmatic**: `torch._C._distributed_c10d._dump_nccl_trace()` — returns JSON dict of all recorded operations.

**Signal-based**: Register `SIGUSR1` handler that calls `_dump_nccl_trace()`. Limitation: signal handlers don't execute when the process is stuck inside a NCCL kernel.

**Watchdog-triggered**: On timeout, the watchdog automatically includes flight recorder data in the error report.

---

## Data Flow

### End-to-End Collective Operation

```
Python: dist.all_reduce(tensor)
    ↓
C++: ProcessGroupNCCL::allreduce(tensors, opts)
    ↓
1. Get or create NCCLComm:
   getNCCLComm(deviceKey) → cached or initNCCLComm()
    ↓
2. Record pre-op CUDA event (ncclStartEvent_)
    ↓
3. Launch NCCL kernel on ncclStream_:
   ncclAllReduce(sendbuf, recvbuf, count, datatype, op, comm, stream)
    ↓
4. Record post-op CUDA event (ncclEndEvent_)
    ↓
5. Create WorkNCCL with events, timeout, seq number
    ↓
6. workEnqueue(work) → push to workMetaList_
    ↓
7. Return work handle to Python
    ↓
Watchdog (background):
   Polls workMetaList_ every 100ms
   Queries ncclEndEvent_ to detect completion
   Moves completed work to completedWorkList_
   Checks checkTimeout() for hung operations
```

### End-to-End P2P Operation

```
Python: dist.send(tensor, dst=peer)
    ↓
C++: ProcessGroupNCCL::send(tensors, dstRank, tag)
    ↓
ProcessGroupNCCL::pointToPoint(tensor, fn, peer, SEND)    [line ~4100]
    ↓
1. Determine communicator key:
   - Eager mode: key = deviceKey (reuse parent comm)
   - Lazy mode: key = getKeySendRecv(rank_, peer) (per-pair comm)
    ↓
2. getNCCLComm(key) → may return nullptr for new pair
    ↓
3. If nullptr: initNCCLComm() → broadcastUniqueNCCLID()
   ⚠️  This blocks on store_->get() if peer hasn't called store_->set()
    ↓
4. Launch ncclSend() on ncclStream_
    ↓
5. workEnqueue(work)
    ↓
6. Return work handle
```

---

## Key Files Reference

### Core Implementation
- `ProcessGroupNCCL.cpp` (~6084 lines): Full runtime — init, collectives, p2p, watchdog, store exchange
- `ProcessGroupNCCL.hpp`: Class definitions for ProcessGroupNCCL, WorkNCCL, Watchdog, HeartbeatMonitor

### NCCL Utilities
- `NCCLUtils.hpp`: `NCCLComm` wrapper class, NCCL error checking macros
- `NCCLUtils.cpp`: Communicator creation (`NCCLComm::create`, `NCCLComm::split`), abort logic

### Store Layer
- `Store.hpp`: Abstract `Store` interface (`get`, `set`, `wait`)
- `TCPStore.hpp` / `TCPStore.cpp`: TCP-based key-value store (default for `torchrun`)
- `FileStore.hpp` / `FileStore.cpp`: Filesystem-based store
- `PrefixStore.hpp`: Wraps store with per-PG prefix to namespace keys

### Python Layer
- `torch/distributed/distributed_c10d.py`: `init_process_group()`, collective wrappers, `destroy_process_group()`
- `torch/distributed/rendezvous.py`: Rendezvous backends for `torchrun`

### Key Functions Quick Map

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `initNCCLComm()` | ProcessGroupNCCL.cpp | ~3028 | Lazy communicator creation |
| `eagerConnectSingleDevice()` | ProcessGroupNCCL.cpp | ~1066 | Eager init (device_id) |
| `broadcastUniqueNCCLID()` | ProcessGroupNCCL.cpp | ~2852 | Store-based ncclUniqueId exchange |
| `getNCCLComm()` | ProcessGroupNCCL.cpp | ~3320 | Communicator cache lookup |
| `pointToPoint()` | ProcessGroupNCCL.cpp | ~4100 | Send/recv dispatch |
| `workEnqueue()` | ProcessGroupNCCL.cpp | ~3545 | Push work to watchdog queue |
| `Watchdog::runLoop()` | ProcessGroupNCCL.cpp | ~2247 | Timeout/error polling loop |
| `WorkNCCL::checkTimeout()` | ProcessGroupNCCL.cpp | ~707 | Timeout check per work item |
| `WorkNCCL::abort()` | ProcessGroupNCCL.cpp | ~901 | Communicator abort |
| `HeartbeatMonitor::runLoop()` | ProcessGroupNCCL.cpp | ~1808 | Watchdog liveness monitor |

---

## Design Patterns

### 1. Store-Based Rendezvous

All cross-rank coordination uses the key-value store. Rank 0 generates unique data (ncclUniqueId), publishes via `store_->set()`, other ranks retrieve via `store_->get()`. This avoids direct rank-to-rank communication during setup.

### 2. CUDA Event-Based Completion

NCCL operations are asynchronous on the GPU. Completion is detected by recording CUDA events before and after the NCCL kernel, then querying the end event from the watchdog thread (via `cudaEventQuery`). This avoids blocking the CPU.

### 3. Two-Lock Work Pipeline

Three threads coordinate via two mutexes:
- `workMetaListMutex_` guards `workMetaList_` (main thread writes, watchdog reads)
- `completedWorkListMutex_` guards `completedWorkList_` (watchdog writes, hook thread reads)
- Lock ordering: always `workMetaListMutex_` before `completedWorkListMutex_` to prevent deadlock

### 4. Per-Pair P2P Communicators (Lazy Mode)

In lazy mode, each send/recv pair gets its own 2-rank NCCL communicator. This prevents serialization (single stream per communicator) but creates the risk of orphaned communicator init if one side exits early.

### 5. Sequence Numbering

Separate counters for collectives (`seqCollective_`) and p2p (`seqP2P_`). Monotonically increasing per process group. Used by flight recorder to compare operation sequences across ranks and detect mismatches.

---

**For hang patterns and debugging**: See [SKILL.md](SKILL.md)
