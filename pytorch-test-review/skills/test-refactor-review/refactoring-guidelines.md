# Refactoring Guidelines

Domain-specific rules for reviewing PyTorch test refactoring PRs. These supplement the review checklist with detailed reference material, conversion tables, and common mistake catalogs.

## Hardware Classification Reference

### Enum Definition (PR #186918)

```python
from torch.testing._internal.common_utils import HardwareClassification

class HardwareClassification(Enum):
    GENERIC      = "generic"      # Device-irrelevant, runs once on CPU
    ACCELERATOR  = "accelerator"  # Runs across all accelerators via instantiate_device_type_tests
    CPU          = "cpu"          # CPU-locked (use sparingly)
    CUDA         = "cuda"         # CUDA-locked (use sparingly)
    MPS          = "mps"          # MPS-locked (use sparingly)
    XPU          = "xpu"          # XPU-locked (use sparingly)
```

### Decision Tree

```
Is the test class parameterized via instantiate_device_type_tests?
├── YES → HardwareClassification.ACCELERATOR
└── NO
    ├── Does it test CPU-only / device-irrelevant logic?
    │   └── YES → HardwareClassification.GENERIC
    ├── Is it locked to a specific device API?
    │   ├── CUDA profiler, CUDA runtime → HardwareClassification.CUDA
    │   ├── MPS graph API → HardwareClassification.MPS
    │   └── XPU-specific kernels → HardwareClassification.XPU
    └── CPU-only but tests device-aware ops?
        └── Evaluate: if logic works on accelerators → ACCELERATOR, otherwise → CPU
```

### Usage Pattern

```python
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)

class TestMyFeature(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_something(self):
        ...

class TestMyFeatureDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_something(self, device):
        ...

instantiate_device_type_tests(TestMyFeatureDevice, globals())
```

### CLI Verification

```bash
# Run only GENERIC tests
python test/<file>.py --hw-classification GENERIC -v

# Run only ACCELERATOR tests
python test/<file>.py --hw-classification ACCELERATOR -v

# Verify zero unclassified
python test/<file>.py --hw-classification unclassified -v 2>&1 | grep "Ran 0 tests"
```

## Profiler Test Organization

Rules from @fffrog's reviews (#182432, #182434) for profiler test files:

| Class | Purpose | Classification |
|-------|---------|----------------|
| `TestProfiler` | General profiler mechanisms, no accelerator logic | `HardwareClassification.GENERIC` |
| `TestProfilerDevice` | Multi-backend profiler tests via `instantiate_device_type_tests` | `HardwareClassification.ACCELERATOR` |
| `TestProfilerCUDA` | CUDA-specific: multigpu, oom, sync_events | `HardwareClassification.CUDA` |
| `TestProfilerITT` | ITT backend-specific | Device-specific |

### Profiler-Specific Rules

- Move CUDA-specific tests (multigpu, oom, sync_events) to `TestProfilerCUDA`
- Move multi-backend tests (kineto, flops, memory_profiler — 19 tests) to `TestProfilerDevice`
- **Refactor while moving**: complete hardcoded value removal simultaneously — don't just move code without converting device references
- Use `get_profiler_activities(device_type)` — never manual `ProfilerActivity.CUDA` mapping
- Use `self_device_time_total` — `self_cuda_time_total` is deprecated

## Dynamo-Specific Rules

### What NOT to Device-Parameterize

Guidance from Andrew James and @mikaylagawarecki:

- **Pure tracing/compilation tests**: Dynamo tracing is device-independent. Adding `device` param creates N identical tests with no additional coverage. Keep as `GENERIC`.
- **Inductor-reliant tests** (`backend="inductor"`): Hold off per Mikayla's guidance. Leave as-is.
- **Triton codegen tests**: Hardware-specific by nature. Keep as `CUDA`.
- **Tests with `backend="eager"`**: These test tracing, not execution. Usually `GENERIC`.

### Device API Conversion Table

When converting a test from CUDA-specific to device-agnostic (`ACCELERATOR`), apply these conversions:

| From (CUDA-specific) | To (device-agnostic) |
|----------------------|---------------------|
| `torch.cuda.Stream()` | `torch.Stream(device)` |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` |
| `with torch.cuda.stream(s):` | `with s:` |
| `torch.cuda.Event()` | `torch.Event(device)` |
| `torch.cuda.device(idx)` | `torch.get_device_module(device).device(idx)` |
| `torch.cuda._exchange_device(idx)` | `device_module._exchange_device(idx)` |
| `torch.cuda._maybe_exchange_device(idx)` | `device_module._maybe_exchange_device(idx)` |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_idx()` |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` |
| `torch.autocast(device_type="cuda")` | `torch.autocast(device_type=device)` |
| `device="cuda"` | `device=device` |
| `x.cuda()` | `x.to(device)` |
| `x.to("cuda")` | `x.to(device)` |
| `torch.cuda.is_available()` | Remove — device-type infra handles availability |
| `@unittest.skipIf(not torch.cuda.is_available())` | Remove — framework handles it |
| `self_cuda_time_total` | `self_device_time_total` |

## Common Mistakes

### Classification Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| `GENERIC` on a class with `instantiate_device_type_tests` | GENERIC means no device parameterization | Use `ACCELERATOR` |
| `ACCELERATOR` on pure tracing tests | Creates N identical tests, no coverage gain | Use `GENERIC` |
| Using old names (`DEVICE_GENERIC`, `DEVICE_SPECIFIC`) | Don't exist in merged enum (#186918) | Use `ACCELERATOR` or device-specific value |
| `hw_classification = "generic"` (string) | Must use enum, not string | `HardwareClassification.GENERIC` |
| `@onlyCPU` class auto-classified as `GENERIC` | `@onlyCPU` implies device awareness | Check if test is truly device-irrelevant |
| `world_size=1` test classified as multi-accelerator | Single-process distributed may be `GENERIC` | Evaluate if device matters |
| Missing `HardwareClassification` import | Code will fail at runtime | Add import from `common_utils` |

### Structural Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Tests deleted during split | Reduces test coverage — albanD's top concern | Migrate every `test_*` to appropriate class |
| `setUp`/`tearDown` not duplicated | New class missing setup logic | Copy to every class that needs it |
| Helpers orphaned in old class | Not available in new class | Move to first use |
| `instantiate_device_type_tests` with wrong `only_for` | Changes test scope silently | Match original device restrictions |
| Class renamed unnecessarily | Gratuitous churn | Only rename when splitting |
| Device name in instantiated class name | `TestFooCUDA` → `TestFooCUDACUDA` | Use neutral name (`TestFoo`) |
| Manual mixin splitting | Against PyTorch convention | Use `instantiate_device_type_tests` + decorators |

### Code Pattern Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| `torch.cuda.*` in `ACCELERATOR` class | Not device-agnostic | Use `torch.accelerator.*` or `device` param |
| Hardcoded `"cuda"` in string comparisons | Fails on XPU/HPU/PrivateUse1 | Use `device` or `self.device_type` |
| Unnecessary intermediate variable | `use_device = device_type; fn(use_device=use_device)` | `fn(use_device=device_type)` directly |
| Manual `ProfilerActivity.CUDA` mapping | Breaks on other devices | Use `get_profiler_activities(device_type)` |
| Missing PrivateUse1 fallback | Breaks custom backends | Add `_get_privateuse1_backend_name()` check |

### PR Format Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Multiple files refactored in one PR | Harder to review, higher revert risk | One file per PR |
| Mixing refactoring with cleanup | Violates single-concern rule | Separate PRs |
| Missing test plan | Reviewer can't verify | Add exact commands + output |
| Missing `--hw-classification` verification | Can't confirm zero unclassified | Add filter run to test plan |
| Missing AI disclosure | Violates AI_POLICY.md | Add "Authored with AI assistant" |
| `[skip ci]` in commit | Tests must validate | Never skip CI |

## Anti-Pattern / Good-Pattern Quick Reference

### Anti-Patterns (flag these)

| Pattern | Severity |
|---------|----------|
| Missing `hw_classification` on any class | **Critical** |
| `hw_classification = "generic"` (string instead of enum) | **Critical** |
| `DEVICE_GENERIC` / `DEVICE_SPECIFIC` / `MULTI_DEVICE_GENERIC` names | **Critical** |
| `[skip ci]` in commit message | **Critical** |
| Tests deleted during refactoring | **Critical** |
| `device="cuda"` / `device="cuda:0"` in ACCELERATOR class | **Warning** |
| `.cuda()` / `.to("cuda")` / `.to("mps")` in ACCELERATOR class | **Warning** |
| `torch.cuda.is_available()` in test logic | **Warning** |
| `torch.cuda.current_device()` in ACCELERATOR class | **Warning** |
| `torch.cuda.device_count()` in ACCELERATOR class | **Warning** |
| `@unittest.skipIf(not TEST_CUDA)` | **Warning** |
| `@skipIf(IS_MACOS)` for dtype constraints | **Warning** |
| Manual config save/restore in ACCELERATOR class | **Warning** |
| `self_cuda_time_total` (deprecated) | **Warning** |
| `NotImplementedError` without issue number | **Suggestion** |

### Good Patterns (expected in well-refactored PRs)

| Pattern | Purpose |
|---------|---------|
| `hw_classification = HardwareClassification.GENERIC` | Correct classification |
| `hw_classification = HardwareClassification.ACCELERATOR` | Correct classification |
| `instantiate_device_type_tests(TestFoo, globals())` | Device-generic test infrastructure |
| `@dtypes(torch.float32, torch.float64)` | Multi-dtype coverage |
| `@onlyAccelerators` | Clean CPU exclusion (replaces `@onlyCUDA` when test works on any accelerator) |
| `@deviceCountAtLeast(2)` + `@onlyAccelerators` | Multi-device tests |
| `torch.accelerator.*` APIs | Device-agnostic accelerator access |
| `torch.Stream(device)` / `torch.Event(device)` | Device-agnostic stream/event |
| `get_profiler_activities(device_type)` | Correct profiler API |
| `self_device_time_total` | Non-deprecated profiler API |
| `make_tensor(shape, device=device, dtype=dtype)` | Proper tensor creation |

## `instantiate_device_type_tests` Reference

```python
instantiate_device_type_tests(
    generic_test_class,     # Your test class
    scope,                  # globals()
    except_for=None,        # ["cpu", "cuda"] — exclude these devices
    only_for=None,          # ["cuda", "xpu"] — only these devices
    include_lazy=False,     # Include lazy tensor tests
    allow_mps=False,        # Include MPS tests
    allow_xpu=False,        # Include XPU tests
)
```

Creates device-specific subclasses automatically:
- `TestFooCPU` (extends CPUTestBase + TestFoo)
- `TestFooCUDA` (extends CUDATestBase + TestFoo)
- `TestFooXPU`, `TestFooMPS`, `TestFooHPU`, etc.

**Key rule**: Don't put device names in the class name passed to `instantiate_device_type_tests`. The framework appends the device suffix automatically.

## Upstream Reviewer Reference

| Reviewer | Focus Area | Key PRs |
|----------|-----------|---------|
| @fffrog | OpenReg, PrivateUse1, profiler organization, decorators, single-concern PRs | #180328, #180368, #181487, #181522, #181888, #181889, #182432, #182434, #185881 |
| @albanD | Classification naming ("union and documentation"), enum design, no device names in instantiated classes | #185802, #186918 |
| @jbschlosser | Classification simplicity, mechanical migration rules, no manual mixin splitting | #186918, #188331 |
| @mikaylagawarecki | Inductor hold-off, dynamo-specific guidance | dynamo test files |
| @Lucaskabela | Never delete tests — always migrate | #180374 |
| @mansiag05 | `@dtypesIfMPS` over `@skipIf(IS_MACOS)`, correct multi-device classification | #176593, #190082 |

---

*Based on merged PR [#186918](https://github.com/pytorch/pytorch/pull/186918) and upstream review feedback from the test refactoring initiative. Last updated: August 2026.*
