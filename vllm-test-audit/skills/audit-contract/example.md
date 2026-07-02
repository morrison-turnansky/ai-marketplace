# Worked Examples

Two examples showing the full Phase 1 → Phase 2 pipeline: one COINCIDENTALLY_CORRECT, one STRONG_CONTRACT.

## Example 1: test_cascade_attention (COINCIDENTALLY_CORRECT)

From `tests/v1/e2e/general/test_cascade_attention.py`:

```python
single_prompt = [example_system_message + prompt]
responses = llm.generate(single_prompt, sampling_params)  # temp=0.0
ref_output = responses[0].outputs[0].text

# (Probably) Use cascade attention.
prompts = [example_system_message + prompt] * 64
responses = llm.generate(prompts, sampling_params)
for response in responses:
    assert response.outputs[0].text == ref_output  # EXACT string match
```

**Why it's coincidentally correct:** Compares batch=1 vs batch=64 via exact string equality without `VLLM_BATCH_INVARIANT`. No strong contract (Not Strong #6). PyTorch issue 182700 showed this breaks.

### Phase 1 output

```python
import sys
sys.path.insert(0, "${CLAUDE_PLUGIN_ROOT}/scripts")
from output_object import AuditCandidate, AuditReport

report = AuditReport(
    test_files_in_scope=1,
    candidates_analyzed=1,
    candidates=[
        AuditCandidate(
            candidate="test_cascade_attention",
            file="tests/v1/e2e/general/test_cascade_attention.py",
            line=43,
            comparison="batch=1 vs batch=64",
            oracle="exact text equality (assert .text ==)",
            helper="direct assertion",
            batch_invariant_enabled=False,
            code_path_verified=False,
            fixtures="@create_new_process_for_each_test()",
            c1_weak_oracle="yes — exact string == on generated text",
            c2_realistic_breakage="yes — PyTorch #182700, cuBLAS kernel selection changes with batch size",
            c3_no_strong_contract="yes — Not Strong #6: batch size invariance without BI mode",
            classification="COINCIDENTALLY_CORRECT",
            coincidentally_correct=True,
            code_snippet="prompts = [example_system_message + prompt] * 64\nresponses = llm.generate(prompts, sampling_params)\nfor response in responses:\n    assert response.outputs[0].text == ref_output",
        ),
    ],
)

expected_tests = open("../test_list.csv").read().strip().splitlines()
report.verify_coverage(expected_tests)
report.write_split("../audit-cc.json", "../audit-not-cc.json")
```

### Phase 2 output (AGREE)

```python
import sys
sys.path.insert(0, "${CLAUDE_PLUGIN_ROOT}/scripts")
from output_object import ReviewCandidate, ReviewReport

report = ReviewReport(
    test_files_in_scope=1,
    candidates_analyzed=1,
    phase_1_agreed=1,
    phase_1_reclassified=0,
    candidates=[
        ReviewCandidate(
            candidate="test_cascade_attention",
            phase_1_classification="COINCIDENTALLY_CORRECT",
            phase_1_coincidentally_correct=True,
            review="AGREE",
            file="tests/v1/e2e/general/test_cascade_attention.py",
            line=43,
            comparison="batch=1 vs batch=64",
            oracle="exact text equality (assert .text ==)",
            helper="direct assertion",
            batch_invariant_enabled=False,
            code_path_verified=False,
            fixtures="@create_new_process_for_each_test()",
            c1_weak_oracle="agree — exact string == on generated text",
            c2_realistic_breakage="agree — cuBLAS kernel selection changes with batch size",
            c3_no_strong_contract="agree — Not Strong #6: batch size invariance without BI mode",
            classification="COINCIDENTALLY_CORRECT",
            coincidentally_correct=True,
            code_snippet="prompts = [example_system_message + prompt] * 64\nfor response in responses:\n    assert response.outputs[0].text == ref_output",
        ),
    ],
)

report.write_to_file("../review-cc.json")
```

## Example 2: test_cpu_offload (Phase 2 reclassifies)

Shows Phase 2 catching a Phase 1 error. Phase 1 classified as COINCIDENTALLY_CORRECT, Phase 2 identifies Strong Contract #5.

### Phase 1 output (incorrect)

```python
AuditCandidate(
    candidate="test_cpu_offload",
    file="tests/basic_correctness/test_cpu_offload.py",
    line=42,
    comparison="normal loading vs CPU offload loading",
    oracle="exact dict equality (compare_two_settings)",
    helper="compare_two_settings",
    batch_invariant_enabled=False,
    code_path_verified=False,
    fixtures="none relevant",
    c1_weak_oracle="yes — exact dict equality via compare_two_settings",
    c2_realistic_breakage="yes — different loading paths may use different kernels",
    c3_no_strong_contract="yes — no contract found",
    classification="COINCIDENTALLY_CORRECT",
    coincidentally_correct=True,
    code_snippet="compare_two_settings(model, base_args, offload_args)",
)
```

### Phase 2 output (RECLASSIFY)

```python
ReviewCandidate(
    candidate="test_cpu_offload",
    phase_1_classification="COINCIDENTALLY_CORRECT",
    phase_1_coincidentally_correct=True,
    review="RECLASSIFY — Phase 1 missed Strong Contract #5",
    file="tests/basic_correctness/test_cpu_offload.py",
    line=42,
    comparison="normal loading vs CPU offload loading",
    oracle="exact dict equality (compare_two_settings)",
    helper="compare_two_settings",
    batch_invariant_enabled=False,
    code_path_verified=False,
    fixtures="none relevant",
    c1_weak_oracle="agree — exact dict equality",
    c2_realistic_breakage="disagree — data movement uses identical kernels, no numeric divergence",
    c3_no_strong_contract="disagree — Strong Contract #5: CPU offload must not change math",
    classification="STRONG_CONTRACT",
    coincidentally_correct=False,
    code_snippet="compare_two_settings(model, base_args, offload_args)",
)
```

C2 is "disagree" and C3 cites Strong Contract #5, so Phase 2 reclassifies. `coincidentally_correct` flips from `True` to `False`.
