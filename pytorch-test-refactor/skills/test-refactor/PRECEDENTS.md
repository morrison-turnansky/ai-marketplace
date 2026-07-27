---
last_updated: 2026-07-18
update_cadence: weekly
---

# Precedents — Reviewer Guidance

Corrections from upstream reviewers on openreg/hw_classification PRs. These override generic guidance in SKILL.md/PATTERNS.md. Grouped by workflow phase — check the section matching your current phase.

This file is the sole source of reviewer guidance. It is refreshed at a regular cadence (currently weekly) by the maintainer.

## Classify

Do not classify `@onlyCPU` tests as GENERIC automatically.
Many are historical — the test logic may be device-agnostic.
Examine the actual logic first. If it works on accelerators, keep it in DEVICE_GENERIC and remove the decorator. Defer to follow-up PRs if uncertain.
Why: @fffrog rejected auto-classification in [#185881](https://github.com/pytorch/pytorch/pull/185881).

Do not classify a test as MULTI_ACCELERATOR based on class name alone.
A class with `world_size = 1` or `@skip_if_lt_x_gpu(1)` that runs on a single GPU should be CUDA, not MULTI_ACCELERATOR_CUDA. Match classification to actual runtime requirements.
Why: @mansiag05 caught misclassification in [#190082](https://github.com/pytorch/pytorch/pull/190082).

## Refactor

Do not include device names (CUDA/XPU/MPS) in class names passed to `instantiate_device_type_tests`.
Use neutral names — the framework auto-appends the device suffix.
  ✗ `TestFooCUDA` → `instantiate_device_type_tests` → `TestFooCUDACUDA`
  ✓ `TestFoo` → `instantiate_device_type_tests` → `TestFooCUDA`
Why: @albanD caught this in [#185802](https://github.com/pytorch/pytorch/pull/185802).

Do not rename classes that aren't being split.
Just add `hw_classification`. Only rename when splitting into multiple classes.
  ✗ `TestReductions` → `TestReductionsDevice` (unnecessary churn)
  ✓ `TestReductions` + `hw_classification = HardwareClassification.DEVICE_GENERIC`
Why: @fffrog rejected gratuitous renames in [#185881](https://github.com/pytorch/pytorch/pull/185881).

Do not use `instantiate_device_type_tests` for device-specific classes.
Use real device names directly in the class.
  ✗ `TestReductionOnlyCPU` → `instantiate_device_type_tests(..., only_for="cpu")`
  ✓ `TestReductionCPU` with `hw_classification = HardwareClassification.CPU`
Why: @fffrog + @albanD aligned on this in [#185881](https://github.com/pytorch/pytorch/pull/185881), [#185802](https://github.com/pytorch/pytorch/pull/185802).

Do not manually split classes into mixin + device-specific subclasses.
Use `instantiate_device_type_tests()` + decorators (`@onlyCUDA`, `@skipMPS`, `@onlyAccelerator`) for device variation.
  ✗ `TestFooMixin` → `TestFooCUDA(TestFooMixin)` + `TestFooXPU(TestFooMixin)`
  ✓ `TestFoo` → `instantiate_device_type_tests(TestFoo, globals())`
Why: @jbschlosser rejected manual splitting in [#188331](https://github.com/pytorch/pytorch/pull/188331).

Do not create separate classes for tests that just need an accelerator.
Use `@onlyAccelerator` decorator within the existing device-type class instead of `@onlyCUDA`.
  ✗ Extract `test_foo` into new `TestBarCUDA` class
  ✓ Keep `test_foo` in device-type class with `@onlyAccelerator`
Why: @fffrog guided this approach in [#185881](https://github.com/pytorch/pytorch/pull/185881).