# Triton Template System

Development reference for Inductor's Triton template infrastructure.

**File**: `torch/_inductor/select_algorithm.py`

## Table of Contents

1. [Class Hierarchy](#class-hierarchy)
2. [TritonTemplate (Factory)](#tritontemplate-factory)
3. [TritonTemplateKernel (Instance)](#tritontemplatkernel-instance)
4. [TritonTemplateCaller (Benchmark Wrapper)](#tritontemplatecaller-benchmark-wrapper)
5. [Code Generation Flow](#code-generation-flow)
6. [Integration Points](#integration-points)
7. [Key Methods](#key-methods)

---

## Class Hierarchy

```
KernelTemplate (base class)
    ↓
TritonTemplate (template factory, line 1761)
    ↓ creates via kernel_type
TritonTemplateKernel (concrete kernel instance, line 388)
    ↓ extends
TritonKernel (codegen/triton.py:2504)
```

**Relationship**: TritonTemplate is a **factory** that creates TritonTemplateKernel instances.

---

## TritonTemplate (Factory)

**Location**: `select_algorithm.py:1761`

**Purpose**: Reusable template definition that generates kernel instances.

### Key Attributes

```python
class TritonTemplate(KernelTemplate):
    kernel_type: type[Any] = TritonTemplateKernel  # Factory relationship
    index_counter = itertools.count()
    all_templates: dict[str, "TritonTemplate"] = {}  # Global registry

    def __init__(
        self,
        name: str,
        grid: Any,                    # Grid function: (m,n,k,meta) -> (grid_x, grid_y, grid_z)
        source: str,                  # Jinja2 template source code
        debug=False,
        cache_codegen_enabled_for_template=False,
        prologue_loads_all_inputs=False,
        always_freeze_layout: bool = False,
    ):
        self.template = self._template_from_string(source)  # jinja2.Template
        TritonTemplate.all_templates[name] = self           # Register globally
```

### Critical Methods

**`generate_and_load()`** (line 1833): Main entry point for kernel generation
- Takes input nodes, tuning parameters (num_stages, num_warps), layout
- Renders Jinja2 template with concrete values
- Creates TritonTemplateKernel instance
- Returns compiled kernel or None

**`maybe_append_choice()`** (line 1807): Autotuning integration
- Wraps `generate()` with error handling
- Appends ChoiceCaller to choices list
- Returns None on success, NotImplementedError on failure

**Template rendering** (line 1965):
```python
template = kernel.render(self.template, kwargs, caching_enabled)
code = template.finalize_all()
```

### Usage Pattern

```python
# Template definition (once, typically in kernel/mm.py)
mm_template = TritonTemplate(
    name="matmul",
    grid=lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N, meta['BLOCK_N']), 1),
    source=jinja2_template_string,
)

# Kernel generation (per compilation)
kernel = mm_template.generate_and_load(
    input_nodes=(a, b),
    num_stages=2,
    num_warps=4,
    layout=output_layout,
    kwargs={'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
)
```

---

## TritonTemplateKernel (Instance)

**Location**: `select_algorithm.py:388`

**Purpose**: Concrete kernel instance with specific parameters that renders template to code.

### Key Attributes

```python
class TritonTemplateKernel(TritonKernel):
    def __init__(
        self,
        kernel_name,
        input_nodes: tuple[ir.IRNode, ...],
        output_node,
        defines,                    # Template constants (BLOCK_M, BLOCK_N, etc.)
        num_stages,                 # Triton pipeline stages
        num_warps,                  # Thread warps per block
        grid_fn,                    # Grid calculation function
        meta,                       # Template kwargs dict
        call_sizes,                 # Symbolic sizes for grid
        num_consumer_groups=0,
        num_buffers_warp_spec=0,
        use_jit=False,
        tma_store=False,
        epilogue_fn=identity,
        subgraphs: Optional[list[ir.ComputedBuffer]] = None,
        workspace_arg: Optional[WorkspaceArg] = None,
        ...
    ):
```

### Responsibilities

1. **Code rendering**: Fills Jinja2 template with concrete values
2. **Prologue/epilogue fusion**: Integrates fused operations
3. **Subgraph handling**: Manages nested computations
4. **Workspace management**: Handles temporary buffers
5. **Grid calculation**: Computes CUDA launch dimensions

### Created By

```python
# Inside TritonTemplate.generate_and_load() at line 1930
def make_kernel():
    return self.kernel_type(  # kernel_type = TritonTemplateKernel
        kernel_name=kernel_name,
        output_node=fake_out,
        workspace_arg=workspace_arg,
        **kernel_options,  # num_stages, num_warps, meta, etc.
    )
```

### Rendering Process

```python
# Line 1965 in generate_and_load()
template = kernel.render(self.template, kwargs, caching_enabled)
code = template.finalize_all()  # Returns final Triton code string
```

---

## TritonTemplateCaller (Benchmark Wrapper)

**Location**: `select_algorithm.py:2391`

**Purpose**: Wraps kernel for benchmarking and autotuning.

```python
class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def __init__(
        self,
        name,
        input_nodes,
        layout,
        make_kernel_render,         # Callable that creates kernel
        description,
        bmreq,                      # TritonBenchmarkRequest
        log_info: Optional[dict] = None,
        mutated_inputs=None,
        workspace_arg: Optional[WorkspaceArg] = None,
        allowed_prologue_inps: Optional[OrderedSet[str]] = None,
        hint_override: Optional[int] = None,
    ):
        self.make_kernel_render = make_kernel_render
        self.bmreq: TritonBenchmarkRequest = bmreq
```

### Role in Autotuning

- Part of `choices` list in autotuning
- Provides benchmarking interface
- Tracks performance metrics
- Selected by autotune or heuristics

---

## Code Generation Flow

### Complete Pipeline

```
1. FX Graph (from Dynamo)
   ↓
2. Lowering (lowering.py) - Identifies matmul/conv operations
   ↓
3. Template Selection (kernel/mm.py, kernel/conv.py)
   ↓  template.maybe_append_choice(choices, **kwargs)
4. TritonTemplate.generate_and_load()
   ↓  Creates TritonTemplateKernel instance
5. Kernel Rendering
   ↓  kernel.render(template, kwargs, caching_enabled)
6. Jinja2 Template → Triton Code
   ↓  template.finalize_all()
7. Triton Compilation
   ↓  triton.jit decorator, compile to PTX
8. Execution
   ↓  kernel[grid](*args)
```

### Key Functions

**Template Definition** (kernel/mm.py):
```python
aten_mm_template = TritonTemplate(
    name="mm_default",
    grid=mm_grid,
    source=MATMUL_TEMPLATE,  # Jinja2 string
)
```

**Choice Generation** (kernel/mm.py):
```python
def mm_options(config, m, n, k, layout, inputs):
    for BLOCK_M, BLOCK_N, BLOCK_K in configs:
        aten_mm_template.maybe_append_choice(
            choices,
            input_nodes=inputs,
            layout=layout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
```

**Autotuning** (select_algorithm.py):
```python
return autotune_select_algorithm(op, choices, input_nodes, layout)
```

---

## Integration Points

### 1. Lowering Registration

Templates integrate via lowering functions:

```python
# In kernel/mm.py
@register_lowering(aten.mm)
def mm_lowering(mat1, mat2):
    m, k = mat1.get_size()
    k2, n = mat2.get_size()

    # Generate choices (includes templates)
    choices = mm_options(config, m, n, k, layout, (mat1, mat2))

    # Autotune or heuristic selection
    return autotune_select_algorithm("mm", choices, (mat1, mat2), layout)
```

### 2. IR Node Creation

Templates create IR nodes via TritonTemplateCaller:

```python
return ir.TritonTemplateBuffer(
    layout=layout,
    inputs=input_nodes,
    make_kernel_render=template.make_kernel_render(...),
)
```

### 3. Scheduling Integration

TemplateBuffers participate in normal scheduling:
- Analyzed for fusion opportunities
- Memory planning includes template outputs
- Scheduler respects template boundaries

---

## Key Methods

### TritonTemplate Methods

**`generate_and_load()`**: Main kernel generation
- Caching logic via `_generated_code_cache`
- Creates TritonTemplateKernel instance
- Renders template to code
- Returns GenerateAndLoadResult or None

**`maybe_append_choice()`**: Autotuning wrapper
- Error-safe choice generation
- Returns None (success) or NotImplementedError

**`_template_from_string()`**: Jinja2 setup
- Creates jinja2.Template from source string
- Sets up template environment

### TritonTemplateKernel Methods

**`render()`**: Template rendering
- Fills Jinja2 template with kwargs
- Handles prologue/epilogue fusion
- Returns renderable code object

**`def_kernel()`**: Kernel definition setup
- Inherited from TritonKernel
- Sets up kernel signature and metadata

### Common Kwargs

**Template parameters** (passed to `generate_and_load()`):
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K` - Block sizes
- `num_stages` - Triton pipeline stages (typically 2-5)
- `num_warps` - Thread warps per block (typically 2-8)
- `SPLIT_K` - K-dimension splitting (for matmul)
- `GROUP_M` - M-dimension grouping (for wave quantization)

**Meta dict** example:
```python
meta = {
    'BLOCK_M': 64,
    'BLOCK_N': 64,
    'BLOCK_K': 32,
    'SPLIT_K': 1,
}
```

---

## Development Notes

### Finding Templates

**Matmul**: `torch/_inductor/kernel/mm.py`
**Convolution**: `torch/_inductor/kernel/conv.py`
**Attention**: `torch/_inductor/kernel/flex_attention.py`
**Custom**: Search for `TritonTemplate(name=` in codebase

### Common Patterns

**Template caching**:
```python
if caching_enabled:
    cache_key = self._generated_code_cache.make_key(...)
    # Check cache before rendering
```

**Grid function**:
```python
def mm_grid(m, n, k, meta):
    return (
        triton.cdiv(m, meta['BLOCK_M']),
        triton.cdiv(n, meta['BLOCK_N']),
        meta.get('SPLIT_K', 1),
    )
```

**Template registration**:
```python
TritonTemplate.all_templates[name] = self  # In __init__
```

### Debugging

**Enable debug output**:
```python
config.debug = True
config.trace.enabled = True
```

**Inspect generated code**:
```python
# Code written to /tmp/torchinductor_<user>/
# Look for triton_*.py files
```

**Check template selection**:
```python
# Add logging in maybe_append_choice()
log.info(f"Template {self.name}: {choice}")
```

---

## Related Files

**Templates**: `kernel/mm.py`, `kernel/conv.py`, `kernel/flex_attention.py`
**Codegen**: `codegen/triton.py` (TritonKernel base class)
**IR**: `ir.py` (TritonTemplateBuffer, TritonTemplateCallerBase)
**Selection**: `select_algorithm.py` (autotune_select_algorithm)
**Config**: `config.py` (template-related config options)

---

**Status**: Development reference for Triton template system ✅
