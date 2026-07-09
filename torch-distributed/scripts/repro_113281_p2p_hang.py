"""Repro for pytorch/pytorch#113281: P2P ops hang after communicator abort.

Rank 1 finishes its send/recv and exits. Ranks 0 and 2 hang on
isend/irecv targeting rank 1. NCCL watchdog does not fire.

Usage: torchrun --nproc_per_node=3 scripts/repro_113281_p2p_hang.py

Expected: rank 1 exits, ranks 0 and 2 hang indefinitely.
"""
import os
import signal
import json
import torch
import torch.distributed as dist
from datetime import timedelta

def dump_traces(signum, frame):
    """Dump NCCL flight recorder on SIGUSR1."""
    try:
        traces = torch._C._distributed_c10d._dump_nccl_trace()
        rank = dist.get_rank()
        path = f"/tmp/nccl_trace_rank{rank}.json"
        with open(path, "w") as f:
            json.dump(traces, f, indent=2)
        print(f"Rank {rank}: dumped trace to {path}")
    except Exception as e:
        print(f"Flight recorder dump failed: {e}")

signal.signal(signal.SIGUSR1, dump_traces)

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))

t = torch.zeros(5, device=f"cuda:{rank}")

print(f"Rank {rank}: starting p2p ops")

if rank == 0:
    t = torch.ones_like(t)
    w1 = dist.isend(t, 1)
    w2 = dist.irecv(t, 1)
elif rank == 1:
    w1 = dist.irecv(t, 0)
    w2 = dist.isend(t, 2)
elif rank == 2:
    w1 = dist.irecv(t, 1)
    w2 = dist.isend(t, 1)

print(f"Rank {rank}: waiting on p2p ops")
w1.wait()
w2.wait()
print(f"Rank {rank}: p2p ops completed")

dist.destroy_process_group()
