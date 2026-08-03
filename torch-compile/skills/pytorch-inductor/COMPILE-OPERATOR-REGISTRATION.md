# torch.compile Operator Registration

Guide for making custom operators work with `torch.compile()`.

**For Inductor-specific patterns**: See [COMMON-PATTERNS.md](COMMON-PATTERNS.md)

## Table of Contents

1. [Overview](#overview)
2. [Registration Approaches](#registration-approaches)
3. [FakeTensor Kernel](#faketensor-kernel)
4. [Decomposition Approach](#decomposition-approach)
5. [Lowering Approach](#lowering-approach)
6. [Testing](#testing)
7. [Best Practices](#best-practices)

---

## Overview

To make custom operations compilable with `torch.compile()`, you need:

1. **FakeTensor kernel** - Computes output metadata without execution
2. **Decomposition OR Lowering** - How to generate code for the op

**Choose based on your use case**:
- **Decomposition**: Operation can be expressed using existing PyTorch ops → Easier
- **Lowering**: Need specialized code generation or fusion → More control

---

## Registration Approaches

### When to Use Each Approach

**Use Decomposition when**:
- Operation is composition of existing ops
- Standard fusion is sufficient
- Easier to maintain
- Example: GELU, LayerNorm

**Use Lowering when**:
- Need custom kernel code
- Special fusion patterns required
- Performance-critical path
- Example: Custom CUDA kernels, specialized templates

---

## FakeTensor Kernel

**Required for all custom ops** - enables torch.compile to trace through your operation.

### Basic FakeTensor Registration

```python
import torch
from torch import Tensor
from torch.library import custom_op

# 1. Define custom op
@custom_op("mylib::muladd", mutates_args=())
def muladd(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Custom multiply-add: (a * b) + c"""
    return (a * b) + c

# 2. Register FakeTensor kernel
@torch.library.register_fake("mylib::muladd")
def _(a, b, c):
    # Validate constraints
    torch._check(
        a.shape == b.shape == c.shape,
        lambda: f"All tensors must have same shape, got {a.shape}, {b.shape}, {c.shape}"
    )
    torch._check(
        a.dtype == b.dtype == c.dtype,
        lambda: f"All tensors must have same dtype, got {a.dtype}, {b.dtype}, {c.dtype}"
    )
    torch._check(
        a.device == b.device == c.device,
        lambda: f"All tensors must be on same device, got {a.device}, {b.device}, {c.device}"
    )

    # Return tensor with correct metadata (no computation)
    return torch.empty_like(a)
```

### FakeTensor with Shape Computation

```python
@custom_op("mylib::matmul_add", mutates_args=())
def matmul_add(a: Tensor, b: Tensor, bias: Tensor) -> Tensor:
    """Matrix multiply with bias: (a @ b) + bias"""
    return (a @ b) + bias

@torch.library.register_fake("mylib::matmul_add")
def _(a, b, bias):
    # Validate dimensions
    torch._check(a.dim() == 2, lambda: "a must be 2D")
    torch._check(b.dim() == 2, lambda: "b must be 2D")
    torch._check(bias.dim() == 1, lambda: "bias must be 1D")

    m, k1 = a.shape
    k2, n = b.shape

    torch._check(k1 == k2, lambda: f"Inner dimensions must match: {k1} vs {k2}")
    torch._check(bias.shape[0] == n, lambda: f"Bias shape {bias.shape} must match output dim {n}")

    # Compute output shape
    output_shape = (m, n)
    return torch.empty(output_shape, dtype=a.dtype, device=a.device)
```

### FakeTensor with Dynamic Shapes

```python
@custom_op("mylib::adaptive_pool", mutates_args=())
def adaptive_pool(input: Tensor, output_size: int) -> Tensor:
    """Adaptive pooling to fixed output size"""
    # Implementation...
    pass

@torch.library.register_fake("mylib::adaptive_pool")
def _(input, output_size):
    torch._check(input.dim() >= 2, lambda: "Input must be at least 2D")

    # Output shape: same batch dims + output_size
    batch_dims = input.shape[:-1]
    output_shape = (*batch_dims, output_size)

    return torch.empty(output_shape, dtype=input.dtype, device=input.device)
```

### FakeTensor Best Practices

✅ **DO**:
- Use `torch._check()` for runtime assertions with lambda messages
- Return `torch.empty()` or `torch.empty_like()` for same-shape outputs
- Compute correct output shape/dtype/device
- Test with dynamic shapes (use symbolic sizes)
- Validate all shape constraints

❌ **DON'T**:
- Perform actual computation (defeats lazy execution)
- Allocate real memory beyond metadata tensor
- Access tensor data (`.item()`, indexing, `.numpy()`)
- Use control flow based on tensor values
- Assume static shapes

---

## Decomposition Approach

**Recommended for most cases** - decompose into existing PyTorch operations.

### Simple Decomposition

```python
from torch._inductor.decomposition import register_decomposition

@register_decomposition([torch.ops.mylib.muladd])
def muladd_decomposition(a, b, c):
    """Decompose into mul + add"""
    return torch.add(torch.mul(a, b), c)
```

### Complex Decomposition with Conditionals

```python
@register_decomposition([torch.ops.mylib.gelu])
def gelu_decomposition(x, approximate="none"):
    """GELU with optional approximation"""
    import math

    if approximate == "tanh":
        # Tanh approximation
        return 0.5 * x * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
            )
        )
    else:
        # Exact GELU using erf
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

### Decomposition with Shape Guards

```python
@register_decomposition([torch.ops.mylib.layer_norm])
def layer_norm_decomposition(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """LayerNorm decomposition"""
    # Compute mean and variance over last len(normalized_shape) dims
    reduction_dims = list(range(-len(normalized_shape), 0))

    mean = input.mean(dim=reduction_dims, keepdim=True)
    variance = input.var(dim=reduction_dims, unbiased=False, keepdim=True)

    # Normalize
    normalized = (input - mean) / torch.sqrt(variance + eps)

    # Apply affine transformation
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias

    return normalized
```

### Selective Decomposition

```python
@register_decomposition([torch.ops.mylib.attention])
def attention_decomposition(query, key, value, mask=None, dropout_p=0.0):
    """Decompose attention - but only for small cases"""

    # For large cases, let Inductor use specialized template
    if query.size(-1) > 64 or query.size(-2) > 512:
        return NotImplemented  # Let Inductor use template

    # Standard attention for small cases
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if mask is not None:
        scores = scores + mask

    attn_weights = torch.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attn_weights = torch.dropout(attn_weights, p=dropout_p, train=True)

    return torch.matmul(attn_weights, value)
```

---

## Lowering Approach

**For advanced users** - direct control over Inductor IR generation.

See [COMMON-PATTERNS.md#custom-lowerings](COMMON-PATTERNS.md#custom-lowerings) for detailed Inductor lowering patterns.

### When to Use Lowering

- Need custom fusion logic
- Performance-critical operation
- Special memory layout requirements
- Want to use kernel templates

### Quick Example

```python
from torch._inductor.lowering import register_lowering
from torch._inductor import ops

@register_lowering(torch.ops.mylib.muladd)
def muladd_lowering(a, b, c):
    """Lower to Inductor IR"""
    def inner_fn(idx):
        a_val = ops.load(a, idx)
        b_val = ops.load(b, idx)
        c_val = ops.load(c, idx)
        return ops.add(ops.mul(a_val, b_val), c_val)

    return ops.pointwise(
        device=a.get_device(),
        dtype=a.get_dtype(),
        inner_fn=inner_fn,
        ranges=a.get_size()
    )
```

---

## Testing

### Basic Compilation Test

```python
import torch

# Define and register op (FakeTensor + Decomposition/Lowering)
@custom_op("mylib::muladd", mutates_args=())
def muladd(a, b, c):
    return (a * b) + c

@torch.library.register_fake("mylib::muladd")
def _(a, b, c):
    torch._check(a.shape == b.shape == c.shape)
    return torch.empty_like(a)

@register_decomposition([torch.ops.mylib.muladd])
def muladd_decomp(a, b, c):
    return torch.add(torch.mul(a, b), c)

# Test compilation
@torch.compile(backend="inductor")
def test_fn(a, b, c):
    return torch.ops.mylib.muladd(a, b, c)

a = torch.randn(100, device='cuda')
b = torch.randn(100, device='cuda')
c = torch.randn(100, device='cuda')

# Should compile and run
result = test_fn(a, b, c)
print("✓ Compilation successful")

# Verify correctness
expected = (a * b) + c
assert torch.allclose(result, expected)
print("✓ Correctness verified")
```

### Dynamic Shapes Test

```python
def test_dynamic_shapes():
    """Test with varying input sizes"""

    @torch.compile(backend="inductor", dynamic=True)
    def fn(a, b, c):
        return torch.ops.mylib.muladd(a, b, c)

    # Different sizes should compile once with dynamic shapes
    for size in [10, 100, 1000]:
        a = torch.randn(size, device='cuda')
        b = torch.randn(size, device='cuda')
        c = torch.randn(size, device='cuda')

        result = fn(a, b, c)
        expected = (a * b) + c
        assert torch.allclose(result, expected)

    print("✓ Dynamic shapes work correctly")
```

### Inductor Test Template

```python
from torch._inductor.test_case import TestCase

class TestMyOps(TestCase):
    def test_muladd(self):
        """Test muladd operator with Inductor"""

        def fn(a, b, c):
            return torch.ops.mylib.muladd(a, b, c)

        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        c = torch.randn(100, 100, device='cuda')

        # Compile
        compiled_fn = torch.compile(fn, backend="inductor")

        # Compare outputs
        expected = fn(a, b, c)
        actual = compiled_fn(a, b, c)

        self.assertEqual(expected, actual)

    def test_muladd_cpu(self):
        """Test on CPU backend"""

        def fn(a, b, c):
            return torch.ops.mylib.muladd(a, b, c)

        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        c = torch.randn(100, 100)

        compiled_fn = torch.compile(fn, backend="inductor")

        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c))
```

---

## Best Practices

### Registration Best Practices

✅ **Always register FakeTensor kernel**
- Required for torch.compile to trace through your op
- Test with symbolic shapes

✅ **Start with decomposition**
- Easier to implement and maintain
- Sufficient for most use cases
- Can switch to lowering if needed

✅ **Validate inputs thoroughly**
- Use `torch._check()` in FakeTensor kernel
- Provide clear error messages
- Test edge cases

✅ **Test both eager and compiled**
- Ensure consistent behavior
- Test with different backends (Inductor, eager)
- Test CPU and CUDA

✅ **Document expected behavior**
- Clear docstrings
- Shape/dtype/device requirements
- Performance characteristics

### Common Pitfalls

❌ **FakeTensor does computation**
```python
# BAD - computes actual result
@torch.library.register_fake("mylib::op")
def _(x):
    return x.relu().sum()  # Don't compute!

# GOOD - metadata only
@torch.library.register_fake("mylib::op")
def _(x):
    return torch.empty((), dtype=x.dtype, device=x.device)
```

❌ **Missing shape validation**
```python
# BAD - no validation
@torch.library.register_fake("mylib::matmul")
def _(a, b):
    return torch.empty(a.shape[0], b.shape[1])  # Will crash if shapes don't match

# GOOD - validate first
@torch.library.register_fake("mylib::matmul")
def _(a, b):
    torch._check(a.dim() == 2 and b.dim() == 2)
    torch._check(a.shape[1] == b.shape[0],
                 lambda: f"Inner dims must match: {a.shape[1]} vs {b.shape[0]}")
    return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
```

❌ **Decomposition returns wrong type**
```python
# BAD - returns Python number
@register_decomposition([torch.ops.mylib.sum])
def _(x):
    return x.sum().item()  # Returns Python float!

# GOOD - returns Tensor
@register_decomposition([torch.ops.mylib.sum])
def _(x):
    return x.sum()  # Returns 0-d Tensor
```

### Performance Tips

🚀 **Decomposition allows fusion**
- Inductor automatically fuses decomposed ops
- Often better than hand-written kernels

🚀 **Use templates for complex ops**
- Matrix multiplications, convolutions
- See [COMMON-PATTERNS.md#custom-kernel-templates](COMMON-PATTERNS.md#custom-kernel-templates)

🚀 **Profile compiled code**
```python
import torch._inductor.config as config

# Enable debug output to see fusion
config.trace.enabled = True

@torch.compile(backend="inductor")
def fn(x):
    return torch.ops.mylib.my_op(x)

fn(torch.randn(100, device='cuda'))
# Check stdout for fusion decisions
```

---

## Complete Example

```python
import torch
from torch import Tensor
from torch.library import custom_op
from torch._inductor.decomposition import register_decomposition

# 1. Define custom operation
@custom_op("mylib::fused_gelu", mutates_args=())
def fused_gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """GELU activation with optional tanh approximation"""
    if approximate == "tanh":
        import math
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        ))
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 2. Register FakeTensor kernel
@torch.library.register_fake("mylib::fused_gelu")
def _(x, approximate="none"):
    # Validate
    torch._check(x.dtype in [torch.float32, torch.float16, torch.bfloat16],
                 lambda: f"Unsupported dtype: {x.dtype}")
    torch._check(approximate in ["none", "tanh"],
                 lambda: f"Invalid approximate: {approximate}")

    # Return same shape/dtype/device
    return torch.empty_like(x)

# 3. Register decomposition (optional - for automatic fusion)
@register_decomposition([torch.ops.mylib.fused_gelu])
def fused_gelu_decomp(x, approximate="none"):
    import math
    if approximate == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        ))
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 4. Test
@torch.compile(backend="inductor")
def model(x):
    return torch.ops.mylib.fused_gelu(x, approximate="tanh")

x = torch.randn(1000, 1000, device='cuda')
output = model(x)
print(f"✓ Compiled GELU output shape: {output.shape}")
```

---

## Resources

**PyTorch Documentation**:
- Custom Operators: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- torch.compile: https://pytorch.org/docs/stable/torch.compiler.html

**Related Guides**:
- Inductor lowering patterns: [COMMON-PATTERNS.md](COMMON-PATTERNS.md)
- Architecture details: [ARCHITECTURE.md](ARCHITECTURE.md)

---

**Summary**: To make custom ops work with `torch.compile()`:
1. Register FakeTensor kernel (required)
2. Choose decomposition (easier) or lowering (advanced)
3. Test with eager and compiled modes
4. Validate all edge cases
