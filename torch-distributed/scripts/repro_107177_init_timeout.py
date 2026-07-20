"""Repro for pytorch/pytorch#107177: Init timeout due to store.

Simulates rank 0 straggler: rank 0 sleeps before init, causing other
ranks to hit the store timeout while waiting for the NCCL unique ID.

Usage: torchrun --nproc_per_node=4 scripts/repro_107177_init_timeout.py

Expected: non-zero ranks fail with store timeout error before
rank 0 finishes sleeping.
"""
import os
import time
import torch
import torch.distributed as dist
from datetime import timedelta

rank = int(os.environ["RANK"])

if rank == 0:
    print(f"Rank 0: sleeping 40s to simulate straggler...")
    time.sleep(40)
    print(f"Rank 0: woke up, now calling init_process_group")

print(f"Rank {rank}: calling init_process_group")

try:
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=15),
    )
    print(f"Rank {rank}: init succeeded")

    torch.cuda.set_device(rank)
    t = torch.ones(10, device=f"cuda:{rank}")
    dist.all_reduce(t)
    print(f"Rank {rank}: all_reduce done")

    dist.destroy_process_group()
except Exception as e:
    print(f"Rank {rank}: FAILED — {type(e).__name__}: {e}")
