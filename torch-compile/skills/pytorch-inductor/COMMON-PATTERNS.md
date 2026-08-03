# TorchInductor Common Patterns

Practical patterns for extending and optimizing TorchInductor internals.

**For operator registration**: See [COMPILE-OPERATOR-REGISTRATION.md](COMPILE-OPERATOR-REGISTRATION.md)
**For architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)

## Table of Contents

1. [Custom Lowerings](#custom-lowerings)
2. [Custom Decompositions](#custom-decompositions)
3. [Custom Kernel Templates](#custom-kernel-templates)
4. [Custom FX Passes](#custom-fx-passes)
5. [Layout Constraints](#layout-constraints)
6. [Custom Backends](#custom-backends)
7. [Workflows](#workflows)
8. [Testing](#testing)
9. [Debugging](#debugging)

---

## Custom Lowerings

**When to use**: Need custom IR generation, special fusion, or kernel templates.

### Basic Pointwise Lowering

```python
from torch._inductor.lowering import register_lowering
from torch._inductor import ops

@register_lowering(aten.gelu)
def gelu_lowering(x, approximate="none"):
    """Custom GELU lowering for better fusion"""

    def inner_fn(idx):
        x_val = ops.load(x, idx)

        if approximate == "tanh":
            # Constants
            sqrt_2_pi = ops.constant(0.7978845608, x.get_dtype())
            coeff = ops.constant(0.044715, x.get_dtype())
            half = ops.constant(0.5, x.get_dtype())
            one = ops.constant(1.0, x.get_dtype())

            # GELU ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            x_cubed = ops.mul(ops.mul(x_val, x_val), x_val)
            inner = ops.add(x_val, ops.mul(coeff, x_cubed))
            tanh_arg = ops.mul(sqrt_2_pi, inner)
            tanh_val = ops.tanh(tanh_arg)
            return ops.mul(ops.mul(half, x_val), ops.add(one, tanh_val))
        else:
            # Exact: GELU(x) = 0.5 * x * (1 + erf(x / √2))
            sqrt_2 = ops.constant(1.4142135623730951, x.get_dtype())
            half = ops.constant(0.5, x.get_dtype())
            one = ops.constant(1.0, x.get_dtype())

            erf_arg = ops.div(x_val, sqrt_2)
            erf_val = ops.erf(erf_arg)
            return ops.mul(ops.mul(half, x_val), ops.add(one, erf_val))

    return ops.pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size()
    )
```

### Reduction Lowering

```python
@register_lowering(aten.var.dim)
def var_lowering(x, dim, unbiased=True, keepdim=False):
    """Variance with fusion opportunity"""
    from torch._inductor.lowering import make_reduction

    # First compute mean
    mean = ops.mean(x, dim, keepdim=True)

    # Reduction for variance: E[(x - μ)²]
    def squared_diff_fn(idx, rindex):
        x_val = ops.load(x, idx + [rindex])
        mean_val = ops.load(mean, idx)
        diff = ops.sub(x_val, mean_val)
        return ops.mul(diff, diff)

    sum_sq = make_reduction(
        dtype=x.get_dtype(),
        inner_fn=squared_diff_fn,
        reduction_ranges=get_reduction_size(x, dim),
        reduction_type="sum"
    )

    # Divide by N or N-1
    n = ops.reduction_numel(x, dim)
    if unbiased:
        n = ops.sub(n, ops.constant(1, torch.int64))

    var = ops.div(sum_sq, ops.to_dtype(n, x.get_dtype()))

    if not keepdim:
        var = ops.squeeze(var, dim)

    return var
```

### Template-Based Lowering

```python
@register_lowering(aten.linear)
def linear_lowering(x, weight, bias=None):
    """Use optimized GEMM template"""
    from torch._inductor.kernel.mm import tuned_mm

    # x @ weight.T (use tuned GEMM)
    result = tuned_mm(x, weight, transpose_b=True)

    # Add bias if present
    if bias is not None:
        result = ops.add(result, bias)

    return result
```

---

## Custom Decompositions

**When to use**: Operation is composition of existing ops, standard fusion sufficient.

### Simple Decomposition

```python
from torch._inductor.decomposition import register_decomposition

@register_decomposition([aten.gelu])
def gelu_decomp(x, approximate="none"):
    """Decompose GELU for automatic fusion"""
    import math

    if approximate == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        ))
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

### Conditional Decomposition

```python
@register_decomposition([aten.scaled_dot_product_attention])
def sdpa_decomp(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """SDPA decomposition - only for small cases"""

    # Let template handle large cases (Flash Attention)
    if query.size(-1) > 64 or query.size(-2) > 512:
        return NotImplemented

    # Standard attention for small cases
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        seq_len = query.size(-2)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    if attn_mask is not None:
        scores = scores + attn_mask

    attn_weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, value)
```

---

## Custom Kernel Templates

**When to use**: Complex operations needing hand-optimized kernels (GEMM, conv, attention).

### Triton GEMM Template

```python
from torch._inductor.codegen.triton import TritonTemplate

class CustomGemmTemplate(TritonTemplate):
    """Optimized GEMM with auto-tuning"""

    def __init__(self, input_nodes, layout):
        super().__init__(
            name="custom_gemm",
            input_nodes=input_nodes,
            layout=layout,
        )

    def render(self, kernel, template_buffer_node):
        """Generate Triton kernel code"""

        return r"""
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def {kernel_name}(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
""".format(kernel_name=f"gemm_{id(self)}")
```

---

## Custom FX Passes

**When to use**: Graph-level optimizations, pattern fusion.

### Pattern Matching Replacement

```python
from torch._inductor.pattern_matcher import register_replacement, fwd_only

@register_replacement(
    # Pattern to match
    lambda x, w, b: F.linear(x, w, b).relu(),
    # Replacement
    lambda x, w, b: torch.ops.mylib.fused_linear_relu(x, w, b),
    # Validation
    extra_check=lambda match: match.kwargs.get("bias") is not None,
    pass_dicts=[fwd_only],
)
def fuse_linear_relu(match):
    """Fuse linear + relu"""
    pass  # Decorator does the work
```

### Manual FX Pass

```python
from torch.fx import GraphModule
from torch._inductor.fx_passes import PatternMatcherPass

class FuseAddMMPass(PatternMatcherPass):
    """Fuse add(mm(a, b), c) → addmm(c, a, b)"""

    def __call__(self, graph: GraphModule) -> int:
        count = 0

        for node in graph.graph.nodes:
            if self._is_pattern(node):
                self._apply_fusion(node)
                count += 1

        if count > 0:
            graph.graph.lint()
            graph.recompile()

        return count

    def _is_pattern(self, node):
        """Match: add(mm(a, b), c)"""
        return (node.op == "call_function" and
                node.target == aten.add.Tensor and
                node.args[0].target == aten.mm.default)

    def _apply_fusion(self, node):
        """Replace with addmm"""
        mm_node = node.args[0]
        bias = node.args[1]
        a, b = mm_node.args

        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                aten.addmm.default,
                args=(bias, a, b)
            )

        node.replace_all_uses_with(new_node)
        node.graph.erase_node(node)
        node.graph.erase_node(mm_node)
```

---

## Layout Constraints

**When to use**: Control memory layout for performance.

### Channels-Last for Conv

```python
from torch._inductor.lowering import (
    register_layout_constraint,
    require_channels_last,
)

@register_layout_constraint(aten.conv2d)
def conv2d_layout(node, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Force channels-last for conv2d"""
    input = require_channels_last(input)
    weight = require_channels_last(weight)
    return (input, weight, bias, stride, padding, dilation, groups), {}
```

### Contiguous Layout

```python
from torch._inductor.lowering import require_contiguous

@register_layout_constraint(aten.my_custom_op)
def my_op_layout(node, input, weight):
    """Force contiguous inputs"""
    input = require_contiguous(input)
    weight = require_contiguous(weight)
    return (input, weight), {}
```

---

## Custom Backends

**When to use**: Add support for new hardware.

### Backend Template

```python
from torch._inductor.codegen.common import CodeGen

class MyDeviceScheduling(CodeGen):
    """Code generator for custom device"""

    @classmethod
    def get_backend_features(cls, device):
        return {
            "device_type": "my_device",
            "supports_parallel": True,
            "max_threads": 1024,
        }

    @classmethod
    def get_kernel_class(cls):
        return MyDeviceKernel

    def codegen_nodes(self, nodes):
        """Generate code for nodes"""
        for node in nodes:
            kernel = self.create_kernel(node)
            kernel.codegen()
            self.kernels.append(kernel)

# Register backend
def get_scheduling_for_device(device):
    if device.type == "my_device":
        return MyDeviceScheduling
    # ... other devices
```

---

## Workflows

### Adding New ATen Operator

```bash
# 1. Write lowering
# File: torch/_inductor/lowering.py
@register_lowering(aten.my_new_op)
def my_op_lowering(x, param):
    def inner_fn(idx):
        x_val = ops.load(x, idx)
        # ... compute
        return result
    return ops.pointwise(...)

# 2. Test
# File: test/inductor/test_ops.py
def test_my_op(self):
    def fn(x):
        return aten.my_new_op(x, 5)

    x = torch.randn(10, 10, device='cuda')
    compiled = torch.compile(fn, backend="inductor")
    self.assertEqual(fn(x), compiled(x))

# 3. Run
pytest test/inductor/test_ops.py::test_my_op -v
```

### Optimizing Model

```python
import torch._inductor.config as config

# Enable aggressive optimizations
config.max_autotune = True
config.coordinate_descent_tuning = True
config.triton.cudagraphs = True

# Compile and profile
model = torch.compile(MyModel(), backend="inductor")

with torch.profiler.profile() as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Debugging Compilation

```python
# 1. Enable debug mode
import torch._inductor.config as config
config.debug = True
config.trace.enabled = True

# 2. Try compiling
@torch.compile(backend="inductor")
def fn(x):
    return problematic_op(x)

try:
    fn(x)
except Exception as e:
    print(f"Error: {e}")

# 3. Check generated code
# /tmp/torchinductor_<user>/: Look for errors

# 4. Check for fallbacks
from torch._inductor import metrics
if metrics.fallback_ops:
    print("Ops without lowerings:", metrics.fallback_ops)
```

---

## Testing

### Basic Test

```python
from torch._inductor.test_case import TestCase

class TestMyOps(TestCase):
    def test_pointwise_op(self):
        def fn(x, y):
            return torch.ops.mylib.my_op(x, y)

        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')

        compiled = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(x, y), compiled(x, y))
```

### Dynamic Shapes

```python
def test_dynamic(self):
    @torch.compile(backend="inductor", dynamic=True)
    def fn(x):
        return x.relu().sum()

    # Should compile once
    for bs in [1, 4, 8, 16]:
        x = torch.randn(bs, 128, device='cuda')
        result = fn(x)
        self.assertEqual(result.shape, ())
```

---

## Debugging

### Enable Logging

```python
import torch._inductor.config as config

# All debug output
config.debug = True
config.trace.enabled = True
config.trace.graph_diagram = True
```

### Inspect Generated Code

```python
import os, glob, tempfile

cache_dir = os.path.join(tempfile.gettempdir(), f"torchinductor_{os.getenv('USER')}")
latest = max(glob.glob(f"{cache_dir}/*"), key=os.path.getmtime)
print(f"Generated code: {latest}")

for file in glob.glob(f"{latest}/*.py"):
    print(f"\n=== {file} ===")
    print(open(file).read())
```

### Check Fusion

```python
config.trace.enabled = True

@torch.compile(backend="inductor")
def fn(x):
    return x.relu().add(1).mul(2)

fn(torch.randn(100, device='cuda'))
# Look in stdout for: "Fusing: [Pointwise(...), Pointwise(...), ...]"
```

### Profile

```python
import torch.profiler as profiler

model = torch.compile(MyModel(), backend="inductor")

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")
```

---

## Summary

**Inductor Extension Points**:
- ✅ **Lowerings**: Custom IR generation
- ✅ **Decompositions**: Composition of existing ops
- ✅ **Templates**: Hand-optimized kernels
- ✅ **FX Passes**: Graph optimizations
- ✅ **Layout Constraints**: Memory layout control
- ✅ **Backends**: New hardware support

**For operator registration basics**: [COMPILE-OPERATOR-REGISTRATION.md](COMPILE-OPERATOR-REGISTRATION.md)
**For architecture details**: [ARCHITECTURE.md](ARCHITECTURE.md)
