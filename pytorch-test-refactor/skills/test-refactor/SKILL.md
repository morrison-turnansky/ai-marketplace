---
name: test-refactor
description: Refactor a PyTorch test file to be device-agnostic. Walks through analyzing test classes, classifying them (GENERIC/ACCELERATOR/CPU/CUDA/XPU/MPS), splitting mixed classes, adding hw_classification attributes, converting hardcoded device references, and verifying the refactoring. Use when refactoring any test file in the PyTorch repo for the device-agnostic testing initiative.
---

# PyTorch Test Refactor

Refactor a PyTorch test file to be device-agnostic, following the community test refactoring initiative.

## When to Use

- Refactoring a PyTorch test file for device-agnostic testing
- Adding `hw_classification` attributes to existing test classes
- Splitting a test class that mixes accelerator-unrelated and accelerator-related tests
- Converting hardcoded `cuda`/`xpu`/`mps` references to `self.device_type`

## Background

PyTorch has 1,205+ test files and 560,000+ test cases. The test refactoring initiative decouples tests from specific hardware so new backends can reuse the test suite. Every test class gets a `hw_classification` attribute and must follow the single-responsibility principle — no mixing categories within a class.

See [CLASSIFICATION-GUIDE.md](CLASSIFICATION-GUIDE.md) for the full decision tree, [PATTERNS.md](PATTERNS.md) for before/after code examples, and [PRECEDENTS.md](PRECEDENTS.md) for reviewer corrections from merged PRs.

**Before classifying, check [PRECEDENTS.md](PRECEDENTS.md) for corrections in your file's domain.** PRECEDENTS.md is the sole source of reviewer guidance — apply any corrections that match your current task. Key references: RFC overview [#174469](https://github.com/pytorch/pytorch/issues/174469), classification RFC [#185142](https://github.com/pytorch/pytorch/issues/185142), hw_classification implementation [#186918](https://github.com/pytorch/pytorch/pull/186918), tracking issue [#185590](https://github.com/pytorch/pytorch/issues/185590).

## Core Principles

These three rules from the project lead govern all refactoring decisions:

1. **Class-level grouping**: All test cases must be grouped at the class level, inheriting from `unittest.TestCase` (or a subclass like `TestCase` from `torch.testing._internal.common_utils`).
2. **Single responsibility**: Each test class must contain only ONE category of tests. No mixing GENERIC with ACCELERATOR, no mixing device-specific with ACCELERATOR, etc.
3. **hw_classification attribute**: Every test class must have a `hw_classification` class attribute set to the appropriate `HardwareClassification` enum value.

## Workflow

### Phase 1: Analyze

Read the target test file and build an inventory.

**Step 1 — Read the file and identify all test classes:**
```
For each class in the file:
  - Class name
  - Parent class (TestCase, DeviceTypeTestBase, OpDTypeTestBase, etc.)
  - Number of test methods (methods starting with test_)
  - setUp/tearDown methods present?
  - Class-level fixtures or decorators
```

**Step 2 — Scan for device patterns in each class:**

Look for these signals in every test method and class-level code:

| Signal | What it means |
|--------|--------------|
| `@onlyCUDA`, `@onlyXPU`, `@onlyMPS` | Device-specific test |
| `@onlyNativeDeviceTypes` | ACCELERATOR (CUDA + CPU) |
| `torch.cuda.*`, `torch.xpu.*`, `torch.mps.*` | Hardcoded device reference |
| `"cuda"`, `"xpu"`, `"mps"` as string literals | Hardcoded device reference |
| `self.device_type` or `device` parameter | Already ACCELERATOR |
| `instantiate_device_type_tests()` | Already using device-type infrastructure |
| `instantiate_parametrized_tests()` | Parametrized but not device-typed |
| `DeviceTypeTestBase` as parent | Already ACCELERATOR base |
| `torch.distributed.*`, `dist.init_process_group` | ACCELERATOR (distributed) |
| `dist.get_default_backend_for_device` | ACCELERATOR (distributed) |
| `@with_comms` (DTensor) | ACCELERATOR (distributed) |
| `gloo` (CPU/CUDA only, no XPU/MPS) | ACCELERATOR (limited) |
| `nccl` | CUDA |
| `ProcessGroup` (backend-neutral use) | ACCELERATOR (distributed) |
| `@require_distributed` | ACCELERATOR (distributed) |
| No device references at all | Likely GENERIC (CPU-only logic) |

> **Distributed backend rule:** Distributed tests are ACCELERATOR if the backend is selected dynamically via `dist.get_default_backend_for_device(device_type)` or `@with_comms`. They are CUDA if `"nccl"` is hard-coded.

**Step 3 — Identify mixed classes:**

A class is "mixed" if it contains test methods from more than one category. Flag these — they need splitting in Phase 3.

**Step 4 — Check if `instantiate_device_type_tests` is already in use:**

If it is, the file may be partially refactored. Note which classes already use it and which don't.

### Phase 2: Classify

Apply the classification decision tree to each test class. See [CLASSIFICATION-GUIDE.md](CLASSIFICATION-GUIDE.md) for the full decision tree.

**The three categories:**

| Classification | Description | Infrastructure |
|---------------|-------------|---------------|
| `GENERIC` | Device-agnostic tests: dispatcher, autograd, serialization, JIT/FX, FakePG distributed (mocked PG on CPU). No `torch.cuda/xpu/mps` calls or device-specific decorators. Runs once on CPU, saving accelerator CI capacity since behavior is identical everywhere. | `TestCase` + `instantiate_parametrized_tests()` or plain class |
| `ACCELERATOR` | Tests expected to pass on every accelerator: op numerics, tensor creation/ops, distributed collectives, multi-device communication. Uses `self.device_type`. Litmus test: would swapping CUDA → XPU/MPS/PrivateUse1 still make sense and pass? | Keep existing parent if it provides `self.device_type`; add `instantiate_device_type_tests()` only when no other instantiation mechanism is in use (see Phase 3 Step 4). |
| `CPU` / `CUDA` / `XPU` / `MPS` | Locked to one device because the functionality has no equivalent elsewhere: CUDA memory/graphs, cuDNN/cuBLAS, TunableOp, XPU/MPS-specific kernels. Replaces old `@onlyCPU`/`@onlyCUDA`-style decorators. Use sparingly — only when the test genuinely can't be generalized to ACCELERATOR. | `TestCase` with device guard, use `HardwareClassification.CUDA` / `.XPU` / `.MPS` / `.CPU` |

> **GENERIC vs CPU-specific**: GENERIC tests framework logic that never touches hardware; CPU is just where it happens to run. CPU tests functionality that only exists on the CPU backend (AVX/vectorization, MKL-DNN, thread-pool internals). That's why CPU is rare in practice — most CPU-only tests are GENERIC.

**For each class, produce a classification verdict:**
```
ClassName: ACCELERATOR
  Reason: 15/18 methods test tensor operations with device parameter,
          3 methods use @onlyCUDA (should be split out)
  Action: Split into TestFooAccelerator (ACCELERATOR) + TestFooCUDA (CUDA)
```

**When uncertain about domain-specific tests:** If the test file is in a specialized module (e.g., `test/inductor/`, `test/distributed/`, `test/functorch/`), and you're unsure whether a test is truly device-generic or domain-specific, delegate to the appropriate specialist agent if one is available. The specialist can clarify whether the tested behavior is inherently tied to a specific backend.

### Phase 3: Refactor

Apply the refactoring changes. Work class by class.

**Step 1 — Ensure imports:**

```python
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    # Only if needed:
    # DeviceTypeTestBase, dtypes, onlyCUDA, ops, etc.
)
```

**Step 2 — Add `hw_classification` to existing classes that don't need splitting:**

```python
class TestFoo(TestCase):
    hw_classification = HardwareClassification.GENERIC
    # ... all methods are CPU-only
```

**Step 3 — Split mixed classes:**

When a class has methods from multiple categories:

1. Create new classes with clear names following the pattern: `Test<Feature><Category>`
   - `TestConvGeneric` (GENERIC)
   - `TestConvAccelerator` (ACCELERATOR)
   - `TestConvCUDA` (CUDA)
   - `TestConvXPU` (XPU)
   - `TestConvMPS` (MPS)
   - `TestConvCPU` (CPU)

2. Move methods to the appropriate class. Move shared helpers (setUp, utility methods) to whichever class uses them. If shared across classes, duplicate or extract to a mixin/base.

3. Add `hw_classification` to each new class.

**Step 4 — Convert ACCELERATOR classes:**

For classes classified as ACCELERATOR:

1. Keep the existing parent class if it already provides `self.device_type` (e.g., `DeviceTypeTestBase`, `DTensorTestBase`, `DTensorContinuousTestBase`, `DistributedTestBase`). Only change the parent to `DeviceTypeTestBase` if the class currently inherits from plain `TestCase` and needs device-type support. `instantiate_device_type_tests()` can be used with any of these base classes, not just `DeviceTypeTestBase` — unless the class already uses `instantiate_parametrized_tests` (see Common Pitfall #8).
2. Ensure every test method accepts `self` only — the device is `self.device_type`.
3. Replace hardcoded device strings:
   - `"cuda"` → `self.device_type`
   - `torch.cuda.is_available()` → check removed (device availability is handled by the framework)
   - `torch.device("cuda")` → `torch.device(self.device_type)`
   - `x.cuda()` → `x.to(self.device_type)`
   - `@onlyCUDA` → remove (or move method to device-specific class)
4. Add `instantiate_device_type_tests()` call at module level, at the end of the file (after all class definitions and after any `create_local_tensor_test_class` / `instantiate_parametrized_tests` declarations), just before `if __name__ == "__main__":`. This is required because `instantiate_device_type_tests` removes the original class from globals, so all references to the class must come before it:
   ```python
   # At end of file, after all classes and LocalTensor declarations:
   instantiate_device_type_tests(TestConvAccelerator, globals())

   if __name__ == "__main__":
       run_tests()
   ```
5. Guard setUp/tearDown if they reference specific devices:
   ```python
   def setUp(self):
       super().setUp()
       # If tf32 guard was CUDA-specific, make it conditional
       if self.device_type == "cuda":
           self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
           torch.backends.cuda.matmul.allow_tf32 = False
   ```

**Step 5 — Handle distributed tests:**

For distributed test files:

1. Classify distributed tests appropriately:
   - Generic distributed tests (collectives that work across backends) → `ACCELERATOR`
   - NCCL-specific, CUDA-specific multi-GPU tests → `CUDA`

2. Add `instantiate_device_type_tests()` for ACCELERATOR-classified distributed classes — it works with `DTensorTestBase`, `DTensorContinuousTestBase`, `LocalDTensorTestBase`, and other multi-process bases. Do NOT skip it just because the class uses distributed infrastructure. Do NOT add `only_for=` unless explicitly required.

**Step 6 — Preserve test decorators:**

Keep existing decorators that are still relevant:
- `@skipIfNoLapack`, `@skipIfRocm`, `@skipCUDAIfNoCudnn` — keep as-is
- `@onlyCUDA` — remove if method moved to ACCELERATOR; keep if in CUDA-specific class
- `@dtypes(...)`, `@ops(...)` — keep as-is
- `@parametrize(...)` — keep as-is
- `@skipIfTorchDynamo(...)` — keep as-is

### Phase 4: Verify

**Step 1 — Syntax check:**

```bash
python -c "import ast; ast.parse(open('test/<file>.py').read()); print('OK')"
```

**Step 2 — Run the tests:**

```bash
# Runs on all available devices (CPU + any available accelerator)
python test/<file>.py -v
```

**Step 3 — Verify test count is preserved:**

Compare the total number of test methods before and after refactoring. The count should be the same (methods were moved, not deleted).

**Step 4 — Verify hw_classification coverage:**

```bash
# Every test class should have hw_classification
python -c "
import ast, sys
tree = ast.parse(open('test/<file>.py').read())
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        has_hw = any(
            isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'hw_classification'
                for t in (n.targets if isinstance(n, ast.Assign) else [])
            )
            for n in node.body
        )
        prefix = '✓' if has_hw else '✗'
        print(f'{prefix} {node.name}')
"
```

**Step 5 — Verify `instantiate_device_type_tests` for ACCELERATOR classes:**

Every class with `hw_classification = HardwareClassification.ACCELERATOR` must have a corresponding `instantiate_device_type_tests()` call at module level:

```bash
# Check that all ACCELERATOR classes have instantiate_device_type_tests
python -c "
import ast
tree = ast.parse(open('test/<file>.py').read())
accel_classes = set()
instantiated = set()
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        for n in node.body:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == 'hw_classification':
                        if 'ACCELERATOR' in ast.dump(n.value):
                            accel_classes.add(node.name)
    if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == 'instantiate_device_type_tests':
        if node.args:
            instantiated.add(node.args[0].id if isinstance(node.args[0], ast.Name) else '')
for cls in sorted(accel_classes):
    prefix = '✓' if cls in instantiated else '✗'
    print(f'{prefix} {cls}')
"
```

**Step 6 — Run tests and capture output for PR description:**

Run all three test commands, capture the summary line from each, and use them to populate the PR description. Only include classifications that are actually present in the file.

```bash
# 1. Run all tests
python -m pytest test/<file>.py -v
# Capture the summary line, e.g.: "95 passed, 3 skipped in 819.56s"

# 2. Run one command per classification present in the file
# Only include the ones actually used — check which hw_classification values
# appear in the refactored file and run one command for each.
python -m pytest test/<file>.py -v --hw-classification <CLASSIFICATION>
# e.g.: GENERIC, ACCELERATOR, CPU, CUDA, XPU, MPS
```

If any test fails, investigate and fix before proceeding. Common failure causes:
- `instantiate_device_type_tests` conflicting with `instantiate_parametrized_tests` on the same class (see Common Pitfalls #8)
- Missing helper methods after a class split
- Device reference not fully converted

**Step 7 — PR checklist:**

Before submitting the PR:
- [ ] All test classes have `hw_classification` attribute
- [ ] No class mixes test categories (single responsibility)
- [ ] Test count matches pre-refactoring count
- [ ] Tests pass on CPU
- [ ] Tests pass on CUDA (if applicable)
- [ ] PR title follows format: `[TEST] Refactor <filename> with hw_classification`
- [ ] PR body includes test plan with command and output

**Step 8 — Generate PR description:**

Produce a ready-to-use PR description. Populate the Test Plan section with the actual summary lines captured in Step 6.

```markdown
## Summary
Apply hardware classification structure following [#186918](https://github.com/pytorch/pytorch/pull/186918) guidelines.

Changes:
- <list each class rename/split/classification change, e.g.:>
- Rename TestFoo → TestFooGeneric (hw_classification = GENERIC) — N CPU-only tests
- Split TestBar into TestBarAccelerator (ACCELERATOR, M tests) + TestBarCUDA (CUDA, K tests)
- Add hw_classification = ACCELERATOR to TestBaz (no structural changes)

## Test Plan
<populate with actual captured output from Step 6>
<one entry per classification present in the file>

```bash
python -m pytest test/<file>.py -v
X passed, Y skipped in Z.ZZs

python -m pytest test/<file>.py -v --hw-classification <CLASSIFICATION>
X passed, Y skipped in Z.ZZs
```

## Dependencies
<only if this PR depends on another unmerged PR, add: Depends on #<number>>
```

Check if the refactoring depends on any unmerged PRs (e.g., if `HardwareClassification` enum changes are pending, or if a prior N/N series PR must land first). Only add `Depends on #<number>` lines when there is an actual dependency. Omit the section entirely if there are none.

## Common Pitfalls

1. **Don't delete tests** — This is pure refactoring. Every test method must survive.
2. **Don't change test logic** — Only restructure classes, change device references, add attributes.
3. **setUp/tearDown duplication** — When splitting a class, each new class needs its own setUp/tearDown if the original had one. Don't forget `super().setUp()`.
4. **Helper method ownership** — Helper methods (non-test methods) must move with the tests that call them. Check call sites before moving.
5. **Module-level code** — `instantiate_device_type_tests()` and `instantiate_parametrized_tests()` calls must be at module level, after the class definition.
6. **Import ordering** — Follow the existing file's import style. Don't reorganize unrelated imports.
7. **Class naming** — Follow existing naming conventions in the file. If the file uses `TestFooDeviceType`, keep that pattern for ACCELERATOR classes.
8. **`instantiate_device_type_tests` and `instantiate_parametrized_tests` are mutually exclusive** — Never apply both to the same class. Both mechanisms parametrize test methods; combining them causes `TypeError` from double-parametrization (e.g., `got an unexpected keyword argument 'dtype'`). If a class already uses `instantiate_parametrized_tests`, add `hw_classification` but do NOT add `instantiate_device_type_tests`. See [PATTERNS.md](PATTERNS.md) Pattern 8 for a concrete example.
9. **Don't skip `instantiate_device_type_tests` for distributed tests** — It works with any base class that provides `self.device_type`, including `DTensorContinuousTestBase`, `DTensorTestBase`, and `LocalDTensorTestBase`. Don't reason yourself out of adding it because the class uses multi-process infrastructure.
10. **Don't add `only_for=` to `instantiate_device_type_tests`** — The call should be plain `instantiate_device_type_tests(ClassName, globals())` unless there is an explicit, documented reason to restrict device types. Do not copy patterns from neighboring files without checking this skill first.
