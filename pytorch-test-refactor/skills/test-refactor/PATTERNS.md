# Refactoring Patterns — Before/After Examples

Concrete code transformations for the most common refactoring scenarios.

## Pattern 1: Adding hw_classification to an Existing Class

The simplest case — the class is already well-structured, just needs the attribute.

### Before

```python
from torch.testing._internal.common_utils import TestCase, run_tests

class TestSerialization(TestCase):
    def test_save_load(self):
        x = torch.randn(3, 3)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            y = torch.load(f)
        self.assertEqual(x, y)

if __name__ == "__main__":
    run_tests()
```

### After

```python
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)

class TestSerialization(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_save_load(self):
        x = torch.randn(3, 3)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            y = torch.load(f)
        self.assertEqual(x, y)

if __name__ == "__main__":
    run_tests()
```

**What changed:** Added `HardwareClassification` import and `hw_classification` attribute. Nothing else.

---

## Pattern 2: Converting a Hardcoded CUDA Test to ACCELERATOR

Tests that hardcode `"cuda"` but test behavior that works on any device.

### Before

```python
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_cuda import TEST_CUDA

class TestBinaryOps(TestCase):
    @unittest.skipIf(not TEST_CUDA, "no CUDA")
    def test_add_on_device(self):
        x = torch.randn(3, 3, device="cuda")
        y = torch.randn(3, 3, device="cuda")
        z = x + y
        self.assertEqual(z.device.type, "cuda")
        self.assertEqual(z.shape, (3, 3))

    @unittest.skipIf(not TEST_CUDA, "no CUDA")
    def test_mul_on_device(self):
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x * y
        self.assertEqual(z.shape, (3, 3))

    def test_add_cpu(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        self.assertEqual(z.shape, (3, 3))

if __name__ == "__main__":
    run_tests()
```

### After

```python
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)
from torch.testing._internal.common_device_type import (
    DeviceTypeTestBase,
    instantiate_device_type_tests,
)


class TestBinaryOpsGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_add_cpu(self):
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        self.assertEqual(z.shape, (3, 3))


class TestBinaryOps(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_add_on_device(self):
        x = torch.randn(3, 3, device=self.device_type)
        y = torch.randn(3, 3, device=self.device_type)
        z = x + y
        self.assertEqual(z.device.type, self.device_type)
        self.assertEqual(z.shape, (3, 3))

    def test_mul_on_device(self):
        x = torch.randn(3, 3, device=self.device_type)
        y = torch.randn(3, 3, device=self.device_type)
        z = x * y
        self.assertEqual(z.shape, (3, 3))


instantiate_device_type_tests(TestBinaryOps, globals())

if __name__ == "__main__":
    run_tests()
```

**What changed:**
- Split into GENERIC (CPU-only `test_add_cpu`) and ACCELERATOR (device tests)
- `"cuda"` → `self.device_type`
- `.cuda()` → `.to(self.device_type)` (or use `device=self.device_type` in creation)
- Removed `@unittest.skipIf(not TEST_CUDA, ...)` — the framework handles device availability
- Changed parent to `DeviceTypeTestBase`
- Added `instantiate_device_type_tests()` call

---

## Pattern 3: Splitting a Mixed Class

A class with tests spanning multiple categories.

### Before

```python
class TestConv(TestCase):
    def test_conv1d_shape(self):
        # Pure CPU logic test
        conv = torch.nn.Conv1d(3, 6, 3)
        x = torch.randn(1, 3, 10)
        self.assertEqual(conv(x).shape, (1, 6, 8))

    @onlyCUDA
    def test_conv2d_forward(self):
        # Device test — should work on any accelerator
        conv = torch.nn.Conv2d(3, 6, 3).to("cuda")
        x = torch.randn(1, 3, 10, 10, device="cuda")
        y = conv(x)
        self.assertEqual(y.shape, (1, 6, 8, 8))

    @onlyCUDA
    def test_cudnn_benchmark(self):
        # CUDA-specific — cudnn is CUDA-only
        with torch.backends.cudnn.flags(benchmark=True):
            conv = torch.nn.Conv2d(3, 6, 3).cuda()
            x = torch.randn(1, 3, 10, 10, device="cuda")
            conv(x)  # Should use cudnn benchmark
```

### After

```python
class TestConvGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_conv1d_shape(self):
        conv = torch.nn.Conv1d(3, 6, 3)
        x = torch.randn(1, 3, 10)
        self.assertEqual(conv(x).shape, (1, 6, 8))


class TestConv(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_conv2d_forward(self):
        conv = torch.nn.Conv2d(3, 6, 3).to(self.device_type)
        x = torch.randn(1, 3, 10, 10, device=self.device_type)
        y = conv(x)
        self.assertEqual(y.shape, (1, 6, 8, 8))


class TestConvCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cudnn_benchmark(self):
        with torch.backends.cudnn.flags(benchmark=True):
            conv = torch.nn.Conv2d(3, 6, 3).cuda()
            x = torch.randn(1, 3, 10, 10, device="cuda")
            conv(x)


instantiate_device_type_tests(TestConv, globals())
```

**What changed:**
- One class became three, each with single responsibility
- `@onlyCUDA` removed from the ACCELERATOR test (framework handles it)
- `@onlyCUDA` removed from CUDA-specific test (class-level `hw_classification` handles it)
- Device references converted in the ACCELERATOR class

---

## Pattern 4: Handling setUp/tearDown During Split

When the original class has setUp/tearDown that references specific devices.

### Before

```python
class TestLinalg(TestCase):
    def setUp(self):
        super().setUp()
        self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.prev_tf32
        super().tearDown()

    def test_matmul_cpu(self):
        # CPU-only test
        ...

    def test_matmul_device(self):
        # Device test
        x = torch.randn(3, 3, device="cuda")
        ...
```

### After

```python
class TestLinalgGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_matmul_cpu(self):
        ...


class TestLinalg(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        if self.device_type == "cuda":
            self.prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        if self.device_type == "cuda" and hasattr(self, "prev_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = self.prev_tf32
        super().tearDown()

    def test_matmul_device(self):
        x = torch.randn(3, 3, device=self.device_type)
        ...


instantiate_device_type_tests(TestLinalg, globals())
```

**What changed:**
- setUp/tearDown guarded with `if self.device_type == "cuda"` since tf32 is CUDA-specific
- GENERIC class doesn't need setUp/tearDown at all (no CUDA references)

---

## Pattern 5: Converting `instantiate_parametrized_tests` to `instantiate_device_type_tests`

When a file already uses parametrized tests but not device-type tests.

### Before

```python
class TestOps(TestCase):
    @parametrize("dtype", [torch.float32, torch.float64])
    def test_unary_op(self, dtype):
        x = torch.randn(3, 3, dtype=dtype, device="cuda")
        y = torch.sin(x)
        self.assertEqual(y.dtype, dtype)

instantiate_parametrized_tests(TestOps)
```

### After

```python
class TestOps(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @dtypes(torch.float32, torch.float64)
    def test_unary_op(self, dtype):
        x = torch.randn(3, 3, dtype=dtype, device=self.device_type)
        y = torch.sin(x)
        self.assertEqual(y.dtype, dtype)

instantiate_device_type_tests(TestOps, globals())
```

**What changed:**
- `TestCase` → `DeviceTypeTestBase`
- `@parametrize("dtype", [...])` → `@dtypes(...)` (use the device-type framework's dtype parametrization)
- `"cuda"` → `self.device_type`
- `instantiate_parametrized_tests` → `instantiate_device_type_tests`

---

## Pattern 6: Real-World Example — test_accelerator.py (PR #185211)

This is an actual merged PR showing the refactoring pattern.

### Before

```python
class TestAccelerator(TestCase):
    def test_current_accelerator(self):
        for device_type in ["cuda", "xpu", "mps"]:
            if is_device_available(device_type):
                accelerator = torch.accelerator.current_accelerator()
                self.assertEqual(accelerator.type, device_type)
```

### After

```python
class TestAccelerator(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_current_accelerator(self):
        accelerator = torch.accelerator.current_accelerator()
        self.assertEqual(accelerator.type, self.device_type)

instantiate_device_type_tests(TestAccelerator, globals())
```

**What changed:**
- Removed manual iteration over `["cuda", "xpu", "mps"]`
- Used `self.device_type` — the framework provides the device
- `instantiate_device_type_tests` generates per-device test classes automatically

---

## Pattern 7: Real-World Example — test_embedding.py (PR #187922)

Renaming classes to match the classification convention.

### Before

```python
class TestEmbeddingNN(TestCase):
    # 18 CPU-only tests

class TestEmbeddingNNDeviceType(TestCase):
    # 29 device-agnostic tests
```

### After

```python
class TestEmbeddingGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC
    # 18 CPU-only tests

class TestEmbedding(DeviceTypeTestBase):
    hw_classification = HardwareClassification.ACCELERATOR
    # 29 device-agnostic tests

instantiate_device_type_tests(TestEmbedding, globals())
```

**What changed:**
- `TestEmbeddingNN` → `TestEmbeddingGeneric` (clarifies classification)
- `TestEmbeddingNNDeviceType` → `TestEmbedding` (clarifies classification)
- Added `hw_classification` to both
- Ensured `instantiate_device_type_tests` is called

---

## Pattern 8: ACCELERATOR Class Already Using `instantiate_parametrized_tests`

When a class is classified as ACCELERATOR but already uses `instantiate_parametrized_tests`, do NOT also add `instantiate_device_type_tests`. They are mutually exclusive — both parametrize test methods, and combining them causes `TypeError` from double-parametrization.

### Before

```python
class RedistributeTest(DTensorContinuousTestBase):
    @parametrize("dtype", [torch.float32, torch.complex64])
    @with_comms
    def test_partial_to_shard(self, dtype):
        ...

    @with_comms
    def test_redistribute_negative_shard_dim(self):
        ...

instantiate_parametrized_tests(RedistributeTest)
```

### After

```python
class RedistributeTest(DTensorContinuousTestBase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("dtype", [torch.float32, torch.complex64])
    @with_comms
    def test_partial_to_shard(self, dtype):
        ...

    @with_comms
    def test_redistribute_negative_shard_dim(self):
        ...

instantiate_parametrized_tests(RedistributeTest)
# NO instantiate_device_type_tests here — would conflict with instantiate_parametrized_tests
```

**What changed:** Only added `hw_classification`. The existing `instantiate_parametrized_tests` call is preserved as-is.

**Why this matters:** Adding `instantiate_device_type_tests` to a class that already uses `instantiate_parametrized_tests` causes failures like:
```
TypeError: RedistributeTest.test_partial_to_shard() got an unexpected keyword argument 'dtype'
```
The `@parametrize` decorator and `instantiate_device_type_tests` both inject parameters into the test method signature, leading to double-parametrization conflicts.

**How to verify:** After adding `hw_classification`, run the tests. If you see `TypeError` about unexpected keyword arguments, check whether both instantiation mechanisms are applied to the same class and remove the conflicting one.

---

## Common Device Reference Conversions

| Before | After |
|--------|-------|
| `"cuda"` | `self.device_type` |
| `"cuda:0"` | `f"{self.device_type}:0"` |
| `torch.device("cuda")` | `torch.device(self.device_type)` |
| `x.cuda()` | `x.to(self.device_type)` |
| `x.cpu()` then `x.cuda()` | `x.to("cpu")` then `x.to(self.device_type)` |
| `torch.cuda.is_available()` | Remove check (framework handles availability) |
| `@unittest.skipIf(not TEST_CUDA, ...)` | Remove (framework handles) |
| `@onlyCUDA` | Remove if ACCELERATOR; keep if in CUDA-specific class |
| `torch.cuda.device_count()` | Keep if in CUDA-specific class |
| `torch.cuda.memory_allocated()` | Keep — this is CUDA-specific, belongs in CUDA class |
