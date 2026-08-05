# Test Refactoring Review Checklist

Checklist for reviewing PyTorch device-agnostic test refactoring PRs. Every item is a must-check — skip none.

## Classification

- [ ] `hw_classification` attribute set on **every** test class in the file (including unchanged classes)
- [ ] Uses `HardwareClassification` enum values, not strings (`HardwareClassification.GENERIC`, not `"generic"`)
- [ ] `HardwareClassification` imported from `torch.testing._internal.common_utils`
- [ ] No old/invalid enum names: `DEVICE_GENERIC`, `DEVICE_SPECIFIC`, `MULTI_DEVICE_GENERIC`, `MULTI_DEVICE_SPECIFIC` do not exist
- [ ] `ACCELERATOR` classes use `instantiate_device_type_tests` — if ACCELERATOR, the instantiation call must exist
- [ ] `GENERIC` classes do NOT use `instantiate_device_type_tests` — GENERIC means no device parameterization
- [ ] `ACCELERATOR` on class NOT using `instantiate_device_type_tests` is flagged (likely should be GENERIC)
- [ ] Device-specific classes (`CUDA`, `MPS`, `XPU`) are used only for truly device-locked APIs (CUDA memory management, MPS graph API, XPU-specific kernels)
- [ ] No mixed categories in a single class — all test methods in a class share one classification
- [ ] `@onlyCPU` classes are NOT automatically classified as GENERIC — examine actual logic first (PRECEDENTS: fffrog #185881)
- [ ] `world_size=1` or `@skip_if_lt_x_gpu(1)` classes are NOT classified as multi-accelerator — match classification to runtime requirements (PRECEDENTS: mansiag05 #190082)

## Structure

- [ ] **Test count preserved** — every `test_*` method from the original file exists in the refactored version (no deletions)
- [ ] **setUp/tearDown duplicated** — when splitting a class, each new class has its own setUp/tearDown if the original had one, with `super().setUp()` call
- [ ] **Class-level decorators preserved** — when moving test methods, class decorators are either applied to the new class or moved to individual methods
- [ ] **`instantiate_device_type_tests` correct** — called with proper `only_for`, `except_for`, `allow_xpu`, `allow_mps` parameters
- [ ] **`only_for` matches original** — if the original class had device restrictions, they are preserved in the refactored version (PRECEDENTS: fffrog #181889)
- [ ] **No gratuitous renames** — classes that aren't being split should NOT be renamed, just add `hw_classification` (PRECEDENTS: fffrog #185881)
- [ ] **No device names in instantiated classes** — don't name a class `TestFooCUDA` and pass it to `instantiate_device_type_tests` (creates `TestFooCUDACUDA`) (PRECEDENTS: albanD #185802)
- [ ] **No manual mixin splitting** — use `instantiate_device_type_tests()` + decorators, not `TestFooMixin` → `TestFooCUDA(TestFooMixin)` (PRECEDENTS: jbschlosser #188331)
- [ ] **`instantiate_device_type_tests` not used for device-specific classes** — use real device names directly (PRECEDENTS: fffrog + albanD #185881, #185802)
- [ ] **Helpers at first use** — no helper methods defined where they aren't invoked (PRECEDENTS: fffrog #182432)
- [ ] **No unused intermediate variables** — use `device_type` directly, don't assign to `use_device` first (PRECEDENTS: fffrog #182434)
- [ ] **No separate classes for tests that just need an accelerator** — use `@onlyAccelerator` decorator within device-type class instead (PRECEDENTS: fffrog #185881)

## Code Patterns

- [ ] **No hardcoded device strings** — no `device="cuda"`, `device="cuda:0"`, `"mps"`, `"xpu"` in ACCELERATOR classes
- [ ] **No `.cuda()` / `.to("cuda")` / `.to("mps")`** — use `.to(device)` or `.to(self.device_type)`
- [ ] **No `torch.cuda.*` in ACCELERATOR classes** — use `torch.accelerator.*` or device parameter
- [ ] **Correct stream conversion** — `torch.cuda.Stream()` → `torch.Stream(device)`
- [ ] **Correct event conversion** — `torch.cuda.Event()` → `torch.Event(device)`
- [ ] **Correct current_stream** — `torch.cuda.current_stream()` → `torch.accelerator.current_stream()`
- [ ] **Correct device context** — `torch.cuda.device(idx)` → `torch.get_device_module(device).device(idx)`
- [ ] **Correct autocast** — `torch.autocast(device_type="cuda")` → `torch.autocast(device_type=device)`
- [ ] **Correct current_device** — `torch.cuda.current_device()` → `torch.accelerator.current_device_idx()`
- [ ] **Correct profiler API** — use `get_profiler_activities(device_type)`, not manual `ProfilerActivity.CUDA` mapping (PRECEDENTS: fffrog #182434)
- [ ] **No deprecated profiler API** — `self_cuda_time_total` → `self_device_time_total`
- [ ] **PrivateUse1 fallback** — `getattr(DeviceType, device.upper())` needs `_get_privateuse1_backend_name()` check (PRECEDENTS: fffrog #182432)
- [ ] **Platform check extended** — capability checks return True for PrivateUse1 (PRECEDENTS: fffrog #180328)
- [ ] **No `torch.cuda.is_available()` in test logic** — device availability is handled by the framework
- [ ] **No `torch.cuda.device_count()` in ACCELERATOR classes** — use `torch.accelerator.device_count()`

## Decorators

### Skip Decorators
- [ ] Uses correct skip decorators from `torch.testing._internal.common_device_type`:

| Decorator | Purpose |
|-----------|---------|
| `@skipCPUIf(cond, reason)` | Skip on CPU if condition |
| `@skipCUDAIf(cond, reason)` | Skip on CUDA if condition |
| `@skipXPUIf(cond, reason)` | Skip on XPU if condition |
| `@skipGPUIf(cond, reason)` | Skip on all GPU types if condition |
| `@skipMPSIf(cond, reason)` | Skip on MPS if condition |
| `@skipHPUIf(cond, reason)` | Skip on HPU if condition |
| `@skipPRIVATEUSE1If(cond, reason)` | Skip on PrivateUse1 if condition |

### Only Decorators
- [ ] Uses correct only decorators:

| Decorator | Purpose |
|-----------|---------|
| `@onlyCPU` | CPU only |
| `@onlyCUDA` | CUDA only |
| `@onlyXPU` | XPU only |
| `@onlyMPS` | MPS only |
| `@onlyHPU` | HPU only |
| `@onlyPRIVATEUSE1` | PrivateUse1 only |
| `@onlyAccelerators` | All accelerators, skip CPU |
| `@onlyNativeDeviceTypes` | CPU + CUDA only |
| `@onlyNativeDeviceTypesAnd(["xpu"])` | CPU + CUDA + specified |
| `@onlyOn("cuda")` | Run only on specified device |

### Dtype Decorators
- [ ] Uses correct dtype decorators:

| Decorator | Purpose |
|-----------|---------|
| `@dtypes(torch.float32, torch.float64)` | All devices |
| `@dtypesIfCPU(torch.float32)` | CPU dtype override |
| `@dtypesIfCUDA(torch.float16, torch.bfloat16)` | CUDA dtype override |
| `@dtypesIfXPU(torch.float16)` | XPU dtype override |
| `@dtypesIfMPS(torch.float32)` | MPS dtype override |
| `@dtypesIfHPU(torch.float32, torch.bfloat16)` | HPU dtype override |
| `@dtypesIfPRIVATEUSE1(torch.float32)` | PrivateUse1 dtype override |

### Decorator Patterns
- [ ] **Multi-device tests** use `@deviceCountAtLeast(2)` + `@onlyAccelerators` (PRECEDENTS: fffrog)
- [ ] **`@onlyCUDA` → `@onlyAccelerator` migration** — tests that work on any accelerator should use `@onlyAccelerators`, not `@onlyCUDA`
- [ ] **No `@skipIf(IS_MACOS)` for dtype constraints** — use `@dtypesIfMPS` instead (PRECEDENTS: mansiag05 #176593)
- [ ] **No `@unittest.skipIf(not TEST_CUDA)`** — use `@onlyCUDA` or device-generic skip decorators
- [ ] **No manual `@unittest.skipIf(not torch.cuda.is_available())`** — framework handles this

## CI Lint Compliance

- [ ] **`# Owner(s):` comment** at top of file (TESTOWNERS linter)
- [ ] **`if __name__ == "__main__":` block** present with `run_tests()` (TEST_HAS_MAIN linter)
- [ ] **PR under 2000 LOC** — split larger PRs into multiple
- [ ] **No `[skip ci]`** in commit messages — CI must run
- [ ] **`lintrunner -a` clean** — no lint failures in modified files

## PR Format

- [ ] **Title format**: `[TEST] Refactor <file>.py with hw_classification` or `[Test] Make <TestClass> device-agnostic in <file>`
- [ ] **Single concern** — one file or one logical refactoring per PR (PRECEDENTS: fffrog #181487, #180368)
- [ ] **No unrelated changes** — no formatting, import sorting, or cleanup mixed in
- [ ] **Summary section** — lists each class with its classification and structural changes
- [ ] **Test plan section** — exact commands run and their output, including `--hw-classification` filter runs
- [ ] **`unclassified=0` verified** — test plan output shows zero unclassified test classes
- [ ] **AI disclosure** — "Authored with AI assistant" per PyTorch AI_POLICY.md
- [ ] **No formatting-only changes** — whitespace, import reordering, etc. should not be mixed with refactoring

## Testing Patterns

- [ ] Test classes inherit from `TestCase` (from `torch.testing._internal.common_utils`) or a recognized subclass
- [ ] Uses `run_tests()` at module level
- [ ] Device-generic tests use `instantiate_device_type_tests()` — not manual device loops
- [ ] Uses `@dtypes()` decorator for dtype coverage — not manual dtype iteration
- [ ] Uses `make_tensor()` for test tensor creation — not `torch.rand()` / `torch.randn()` directly
- [ ] Uses `@parametrize()` for non-device parameterization — not `subTest()` loops
