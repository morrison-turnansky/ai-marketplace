# References & Resources

Links to RFCs, tracking documents, review processes, and example PRs for the PyTorch test refactoring initiative.

## RFCs & Issues

| Document | Link | Description |
|----------|------|-------------|
| RFC: Overview of Test Case Refactoring | [pytorch/pytorch#174469](https://github.com/pytorch/pytorch/issues/174469) | Original proposal — motivation, benefits, roadmap |
| RFC: Test Class Classification | [pytorch/pytorch#185142](https://github.com/pytorch/pytorch/issues/185142) | albanD's proposal for 5-category classification with frequency metadata |
| Tracking Issue | [pytorch/pytorch#185590](https://github.com/pytorch/pytorch/issues/185590) | Main tracking issue consolidating all prior discussions |
| hw_classification PR | [pytorch/pytorch#186918](https://github.com/pytorch/pytorch/pull/186918) | Implementation of `HardwareClassification` enum and `--hw-classification` filter |

## Tracking

| Resource | Link |
|----------|------|
| Test Class Refactoring Tracker (spreadsheet) | [Google Sheets](https://docs.google.com/spreadsheets/d/1cDNiLW4KvPcGYPlA3KCDm0zV5PLPUWubno1OyCznKBw/edit?gid=2107261205#gid=2107261205) |
| PyTorch Test Refactoring Project Board | [GitHub Project #154](https://github.com/orgs/pytorch/projects/154) |
| Slack Channel | [#C0AUTDQJ743](https://pytorch.slack.com/archives/C0AUTDQJ743) |

## Spreadsheet Sheets

The tracking spreadsheet has one sheet per module area:

| Sheet | Files | Focus |
|-------|-------|-------|
| Core | 53 | NN, pooling, multihead attention, RNN, modules |
| Tensor | 70 | Core ops, binary ufuncs, indexing, reductions, views |
| Distributed | 285 | c10d, DDP, FSDP, pipeline parallel, RPC |
| Graph | 588 | JIT, export, dynamo, inductor, functorch |
| Math | 16 | linalg, FFT, functorch transforms |
| Quantization | 51 | AO sparsity, quantization, PT2E |
| Utils | 146 | Hardware, profiler, autograd, serialization, utils |

## Example Merged PRs

Study these PRs to understand the accepted refactoring patterns:

| PR | File | What it demonstrates |
|----|------|---------------------|
| [#185211](https://github.com/pytorch/pytorch/pull/185211) | `test/test_accelerator.py` | Converting hardcoded device iteration to `instantiate_device_type_tests` + `self.device_type` |
| [#185699](https://github.com/pytorch/pytorch/pull/185699) | `test/test_binary_ufuncs.py` | Refactoring a large ops test file |
| [#183586](https://github.com/pytorch/pytorch/pull/183586) | `test/test_linalg.py` | Extracting CUDA-only tests (TunableOp) into `TestLinalgCUDA` |
| [#187922](https://github.com/pytorch/pytorch/pull/187922) | `test/nn/test_embedding.py` | Renaming classes with `hw_classification` following PR #186918 |
| [#190830](https://github.com/pytorch/pytorch/pull/190830) | `test/distributed/tensor/test_dtensor_ops.py` | Classifying distributed tests using fake/multi-threaded PGs as GENERIC |

## Key Source Files

These files in the PyTorch repo implement the testing infrastructure:

| File | Purpose |
|------|---------|
| `torch/testing/_internal/common_utils.py` | `HardwareClassification` enum, `run_tests()`, `HardwareClassificationTestLoader` |
| `torch/testing/_internal/common_device_type.py` | `DeviceTypeTestBase`, `instantiate_device_type_tests()`, device decorators |
| `test/conftest.py` | pytest integration for `--hw-classification` filter |
| `test/run_test.py` | Test runner with `--hw-classification` CLI forwarding |

## Review Process

### Phase 1: Initial Review
- **Reviewer:** @fffrog (can increase as needed)
- Tag PR for the PyTorch Test Refactoring Project Board
- Mark as "Ready for Review (Definitive)" when qualified

### Phase 2: Definitive Review
- **Reviewers:** @albanD, @jbschlosser (final gatekeepers)
- Architectural review and Approve & Merge

## PR Conventions

- **Title format:** `[TEST] Refactor <filename> with hw_classification`
- **Body:** Include test plan with commands and output showing tests pass
- **Link PR** in the tracking spreadsheet's "PRs (Ready for Review)" column
- **Update status** in the spreadsheet when PR is merged

## Classification Quick Reference

| Category | Enum Value | When to Use |
|----------|-----------|-------------|
| GENERIC | `HardwareClassification.GENERIC` | CPU-only logic, no device dependency. Device-agnostic tests (dispatcher, autograd, serialization, JIT/FX, FakePG distributed). Runs once on CPU. |
| ACCELERATOR | `HardwareClassification.ACCELERATOR` | Tests expected to pass on every accelerator (op numerics, tensor ops, distributed collectives across backends). Uses `instantiate_device_type_tests`. |
| Device-specific | `HardwareClassification.CUDA` / `.XPU` / `.MPS` / `.CPU` | Locked to one device — functionality has no equivalent elsewhere (CUDA memory/graphs, cuDNN, NCCL-specific, TunableOp). Use sparingly. |
