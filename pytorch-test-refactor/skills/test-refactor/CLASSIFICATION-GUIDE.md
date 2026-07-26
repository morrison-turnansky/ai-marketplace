# Hardware Classification Decision Tree

Use this decision tree to classify each test class in a PyTorch test file. Walk through the questions top-down — the first match wins.

## Decision Tree

```
Does the test reference any specific accelerator?
├── NO (pure CPU logic, dispatcher, autograd, serialization) → GENERIC
│
└── YES → Can the test work on ANY accelerator (not just CUDA)?
    ├── YES (aten op numerics, device tensor ops, backend integration) → ACCELERATOR
    │   └── Includes distributed tests that work across accelerator types
    │       (gloo, any-backend collectives, DDP/FSDP at the API level)
    │
    └── NO (CUDA memory, XPU kernel, MPS graph, TunableOp) → CPU / CUDA / XPU / MPS
        └── Includes device-specific distributed tests
            (NCCL-only, multi-GPU CUDA, peer-to-peer)
```

## Category Details

### GENERIC

Device-agnostic tests: dispatcher, autograd, serialization, JIT/FX, FakePG distributed (mocked PG on CPU). No `torch.cuda/xpu/mps` calls or device-specific decorators. Runs once on CPU, saving accelerator CI capacity since behavior is identical everywhere.

Tests that verify shared CPU-side logic with no ties to any accelerator.

**Indicators:**
- No `torch.cuda.*`, `torch.xpu.*`, `torch.mps.*` calls
- No device string literals (`"cuda"`, `"xpu"`, `"mps"`)
- No device-specific decorators (`@onlyCUDA`, `@onlyXPU`)
- Tests dispatcher, autograd mechanics, serialization, JIT, Python API behavior
- FakePG distributed tests (they mock the process group and run on CPU)

**Examples:**
- Testing that `torch.save` / `torch.load` roundtrips correctly
- Testing autograd graph construction
- Testing op schema validation
- Testing Python API argument parsing

**Infrastructure:**
```python
class TestFooGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC
```

### ACCELERATOR

Tests expected to pass on every accelerator: op numerics, tensor creation/ops, conv/embedding correctness. Uses `self.device_type`, instantiated via `instantiate_device_type_tests`. Litmus test: would swapping CUDA → XPU/MPS/PrivateUse1 still make sense and pass?

Tests that check on-device behavior and should run across ALL accelerators. CPU is included as a device.

**Indicators:**
- Tests aten operator numerics (correctness, dtype handling)
- Tests tensor operations that should work identically on any device
- Uses or should use `self.device_type` / `device` parameter
- Currently uses `@onlyCUDA` but the logic isn't actually CUDA-specific — it just needs *some* accelerator
- Uses `DeviceTypeTestBase` or `instantiate_device_type_tests`
- Tests distributed collectives (all_reduce, broadcast, etc.) using backend-agnostic APIs
- Tests `torch.distributed` with gloo or any-backend configuration
- Tests multi-device tensor movement that isn't backend-specific
- Tests distributed training patterns (DDP, FSDP) at the API level

**Examples:**
- Testing `torch.add` numerics across dtypes
- Testing convolution forward/backward correctness
- Testing embedding lookup on device tensors
- Testing that tensor creation on device works (`torch.randn(3, device=...)`)
- Testing all_reduce correctness with gloo backend
- Testing DDP wrapper behavior
- Testing distributed checkpoint save/load

**Key question:** "If I replace CUDA with XPU/MPS/PrivateUse1, would this test still make sense and be expected to pass?" If yes → ACCELERATOR.

**Infrastructure:**
```python
class TestFooAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_something(self):
        x = torch.randn(3, 3, device=self.device_type)
        # ...

instantiate_device_type_tests(TestFooAccelerator, globals())
```

### CPU / CUDA / XPU / MPS (device-specific)

Locked to one device because the functionality has no equivalent elsewhere: CUDA memory/graphs, cuDNN/cuBLAS, TunableOp, XPU/MPS-specific kernels. Replaces old `@onlyCPU`/`@onlyCUDA`-style decorators. Use sparingly — only when the test genuinely can't be generalized to ACCELERATOR.

Tests locked to a single accelerator because they test hardware-specific functionality.

**Indicators:**
- Tests CUDA-specific APIs: `torch.cuda.memory_allocated()`, CUDA graphs, `cudnn` settings
- Tests XPU-specific kernels or APIs
- Tests MPS graph API or Metal-specific behavior
- Tests TunableOp (CUDA/ROCm-specific tuning)
- Tests backend-specific library integration (cuBLAS, cuDNN, cuSOLVER)
- The test would NOT make sense on another accelerator
- Tests NCCL-specific collectives or optimizations
- Tests multi-GPU CUDA functionality (peer-to-peer, NVLink)
- Tests that require `torch.cuda.device_count() > 1`
- Tests XPU-specific multi-device features

**Examples:**
- Testing CUDA memory management (`torch.cuda.empty_cache()`)
- Testing CUDA graph capture and replay
- Testing cuDNN benchmark mode behavior
- Testing TunableOp tunable parameters
- Testing NCCL all_reduce performance characteristics
- Testing CUDA peer-to-peer memory access
- Testing multi-GPU CUDA graph capture

**Key question:** "Does this test exercise functionality that only exists on one specific device?" If yes → use the appropriate device-specific classification.

> **GENERIC vs CPU-specific**: GENERIC tests framework logic that never touches hardware; CPU is just where it happens to run. CPU tests functionality that only exists on the CPU backend (AVX/vectorization, MKL-DNN, thread-pool internals). That's why CPU is rare in practice — most CPU-only tests are GENERIC.

**Infrastructure:**
```python
class TestFooCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cuda_memory(self):
        # CUDA-specific test
        torch.cuda.empty_cache()
        # ...
```

Use the appropriate enum value: `HardwareClassification.CUDA`, `.XPU`, `.MPS`, or `.CPU`.

## Edge Cases

### Tests with `@onlyCUDA` that are actually ACCELERATOR

Many tests use `@onlyCUDA` simply because CUDA was the only accelerator when the test was written. If the test logic works on any device:
- Remove `@onlyCUDA`
- Move to an ACCELERATOR class
- Replace `"cuda"` with `self.device_type`

### Tests that skip on certain devices

A test can be ACCELERATOR even if it skips on some devices. The classification is about intent, not runtime behavior:
```python
class TestOpsAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @skipCUDAIfNoCudnn  # Skips if cudnn not available, but still ACCELERATOR
    def test_conv_backward(self):
        # This test is ACCELERATOR — it runs on any device that supports conv
        ...
```

### setUp/tearDown with device-specific code

If setUp references a specific device (e.g., saving tf32 settings), guard it:
```python
def setUp(self):
    super().setUp()
    if self.device_type == "cuda":
        self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
```

### Helper classes and mixins

Non-test classes (helper classes, mixins, base classes without test methods) do NOT need `hw_classification`. Only classes that contain `test_*` methods need it.

### Parametrized tests

Tests using `@parametrize` can be any category. The parametrization is orthogonal to hardware classification. A parametrized test that varies dtypes across devices is ACCELERATOR. A parametrized test that varies CPU-only parameters is GENERIC.
