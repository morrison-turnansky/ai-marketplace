---
name: test-refactor
description: Refactor a PyTorch test file to be device-agnostic. Walks through analyzing test classes, classifying them (GENERIC/DEVICE_GENERIC/DEVICE_SPECIFIC/MULTI_DEVICE_GENERIC/MULTI_DEVICE_SPECIFIC), splitting mixed classes, adding hw_classification attributes, converting hardcoded device references, and verifying the refactoring. Use when refactoring any test file in the PyTorch repo for the device-agnostic testing initiative.
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
2. **Single responsibility**: Each test class must contain only ONE category of tests. No mixing GENERIC with DEVICE_GENERIC, no mixing DEVICE_SPECIFIC with DEVICE_GENERIC, etc.
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
| `@onlyNativeDeviceTypes` | Device-generic (CUDA + CPU) |
| `torch.cuda.*`, `torch.xpu.*`, `torch.mps.*` | Hardcoded device reference |
| `"cuda"`, `"xpu"`, `"mps"` as string literals | Hardcoded device reference |
| `self.device_type` or `device` parameter | Already device-generic |
| `instantiate_device_type_tests()` | Already using device-type infrastructure |
| `instantiate_parametrized_tests()` | Parametrized but not device-typed |
| `DeviceTypeTestBase` as parent | Already device-generic base |
| `torch.distributed.*`, `dist.init_process_group` | Multi-device test |
| `nccl`, `gloo`, `ProcessGroup` | Multi-device / distributed test |
| `@require_distributed` | Multi-device test |
| No device references at all | Likely GENERIC (CPU-only logic) |

**Step 3 — Identify mixed classes:**

A class is "mixed" if it contains test methods from more than one category. Flag these — they need splitting in Phase 3.

**Step 4 — Check if `instantiate_device_type_tests` is already in use:**

If it is, the file may be partially refactored. Note which classes already use it and which don't.

### Phase 2: Classify

Apply the classification decision tree to each test class. See [CLASSIFICATION-GUIDE.md](CLASSIFICATION-GUIDE.md) for the full decision tree.

**The five categories:**

| Classification | Description | Infrastructure |
|---------------|-------------|---------------|
| `GENERIC` | CPU-only, tests shared logic (dispatcher, autograd mechanics, serialization, fakePG distributed). No device dependency. | `TestCase` + `instantiate_parametrized_tests()` or plain class |
| `DEVICE_GENERIC` | Tests on-device behavior that should work on ANY accelerator (aten op numerics, backend integration). CPU is included. | `DeviceTypeTestBase` + `instantiate_device_type_tests()` |
| `DEVICE_SPECIFIC` | Tests locked to one specific accelerator (CUDA memory management, XPU-specific kernels, MPS graph API). | `TestCase` with device guard, use `HardwareClassification.CUDA` / `.XPU` / `.MPS` |
| `MULTI_DEVICE_GENERIC` | Tests multi-device behavior (distributed collectives, multi-GPU communication). Should work across accelerator types. | `TestCase` with multi-device setup |
| `MULTI_DEVICE_SPECIFIC` | Tests requiring multiple devices of a specific type (NCCL-specific, multi-GPU CUDA). | `TestCase` with specific multi-device guard |

**For each class, produce a classification verdict:**
```
ClassName: DEVICE_GENERIC
  Reason: 15/18 methods test tensor operations with device parameter, 
          3 methods use @onlyCUDA (should be split out)
  Action: Split into TestFooDeviceGeneric + TestFooCUDA
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
   - `TestConvDeviceGeneric` (DEVICE_GENERIC — replaces the old `TestConvDeviceType` naming)
   - `TestConvCUDA` (DEVICE_SPECIFIC)

2. Move methods to the appropriate class. Move shared helpers (setUp, utility methods) to whichever class uses them. If shared across classes, duplicate or extract to a mixin/base.

3. Add `hw_classification` to each new class.

**Step 4 — Convert DEVICE_GENERIC classes:**

For classes classified as DEVICE_GENERIC:

1. Change parent to inherit from `DeviceTypeTestBase` (if not already).
2. Ensure every test method accepts `self` only — the device is `self.device_type`.
3. Replace hardcoded device strings:
   - `"cuda"` → `self.device_type`
   - `torch.cuda.is_available()` → check removed (device availability is handled by the framework)
   - `torch.device("cuda")` → `torch.device(self.device_type)`
   - `x.cuda()` → `x.to(self.device_type)`
   - `@onlyCUDA` → remove (or move method to DEVICE_SPECIFIC class)
4. Add `instantiate_device_type_tests()` call at module level:
   ```python
   instantiate_device_type_tests(TestConvDeviceGeneric, globals())
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

**Step 5 — Handle MULTI_DEVICE tests (distributed):**

For distributed test files:

1. Check if `HardwareClassification.MULTI_DEVICE_GENERIC` and `MULTI_DEVICE_SPECIFIC` exist in the enum. If not, add them:
   ```python
   # In torch/testing/_internal/common_utils.py, HardwareClassification enum:
   MULTI_DEVICE_GENERIC = "multi_device_generic"
   MULTI_DEVICE_SPECIFIC = "multi_device_specific"
   ```
2. Classify distributed tests appropriately:
   - Generic distributed tests (collectives that work across backends) → `MULTI_DEVICE_GENERIC`
   - NCCL-specific, CUDA-specific multi-GPU tests → `MULTI_DEVICE_SPECIFIC`

**Step 6 — Preserve test decorators:**

Keep existing decorators that are still relevant:
- `@skipIfNoLapack`, `@skipIfRocm`, `@skipCUDAIfNoCudnn` — keep as-is
- `@onlyCUDA` — remove if method moved to DEVICE_GENERIC; keep if in DEVICE_SPECIFIC class
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
# Run with CUDA (if available)
TEST_CONFIG=cuda python test/<file>.py -v

# Run with CPU only
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

**Step 5 — Verify with hw-classification filter:**

```bash
# Test that classification filtering works
python test/<file>.py --hw-classification GENERIC -v
python test/<file>.py --hw-classification DEVICE_GENERIC -v
```

**Step 6 — PR checklist:**

Before submitting the PR:
- [ ] All test classes have `hw_classification` attribute
- [ ] No class mixes test categories (single responsibility)
- [ ] Test count matches pre-refactoring count
- [ ] Tests pass on CPU
- [ ] Tests pass on CUDA (if applicable)
- [ ] PR title follows format: `[TEST] Refactor <filename> with hw_classification`
- [ ] PR body includes test plan with command and output

**Step 7 — Generate PR description:**

Produce a ready-to-use PR description following this template:

```markdown
## Summary
Apply hardware classification structure following [#186918](https://github.com/pytorch/pytorch/pull/186918) guidelines.

Changes:
- <list each class rename/split/classification change, e.g.:>
- Rename TestFoo → TestFooGeneric (hw_classification = GENERIC) — N CPU-only tests
- Split TestBar into TestBarDeviceGeneric (DEVICE_GENERIC, M tests) + TestBarCUDA (CUDA, K tests)
- Add hw_classification = DEVICE_GENERIC to TestBaz (no structural changes)

## Test Plan
**<TEST_CONFIG=cuda python test/<file>.py>**
Ran X tests in Y.YYYs — OK (skipped=Z)

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
7. **Class naming** — Follow existing naming conventions in the file. If the file uses `TestFooDeviceType`, keep that pattern for DEVICE_GENERIC classes.
