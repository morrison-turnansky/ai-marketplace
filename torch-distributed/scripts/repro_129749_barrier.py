"""Repro for pytorch/pytorch#129749: NCCL barrier deadlock.

barrier() hangs with NCCL backend but works with Gloo.
Usage: torchrun --nproc_per_node=4 scripts/repro_129749_barrier.py

Expected: hang at barrier() with NCCL. Pass with Gloo.
"""
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


def main() -> None:
    dist.init_process_group("nccl", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.tensor([1]).to(rank)

    print(f"Rank {rank}: START")
    dist.barrier()
    print(f"Rank {rank}: GOOD — barrier passed")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
