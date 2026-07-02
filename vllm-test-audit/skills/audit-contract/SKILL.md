---
name: audit-contract
description: >-
  Reference-only skill defining the numeric stability audit contract. Contains
  the 3 inclusion criteria, strong contracts, not-strong-by-default clauses,
  assertion pattern guidance, and output formats for both phases. Loaded by
  audit-agent and review-agent as shared knowledge.
---

# Numeric Stability Audit Contract

Single source of truth for the vLLM test oracle auditor. Both Phase 1 (audit-agent) and Phase 2 (review-agent) load this skill for criteria definitions, clause numbering, and output formats.

## Inclusion Criteria

A test is "coincidentally correct" only when **all three** are true:

1. **Weak oracle** — depends on exact generated text/token/logprob equality, match-ratio equality, or another weak generated-output oracle.
2. **Realistic breakage** — a PyTorch numeric/scheduling/compiler change has a realistic chance of changing the asserted value.
3. **No strong contract** — no vLLM/PyTorch/product contract requires the two compared executions to be bitwise/text identical.

## Not Coincidentally Correct by Default

These patterns fail one or more criteria and should be classified as NOT_REALISTIC:

1. **Difference-only tests** (`assert a != b`) — numeric drift unlikely to flip inequality into equality (criterion 2 fails).
2. **Smoke/liveness tests** (`assert len(output) > 0`) — FP changes don't produce empty output (criterion 2 fails).

## Strong Contracts

Treat these as strong enough to classify as STRONG_CONTRACT unless the test adds another weak oracle on top:

1. Eager vs eager with the same request sequence, same engine state, and deterministic sampling.
2. **Kernel tolerance tests** (`assert_close(atol=...)`, `torch.allclose`) — the tolerance IS the contract. These test numeric precision of discrete compute operations, not LLM output.
3. Same compile mode/artifact/config vs itself. Do NOT generalize to different compile strategies or fused distributed passes.
4. Eager vs cudagraph for the same graph/execution family.
5. CPU offload, prefetch offload, sleep/wake restoration, reload, tensorizer, and KV-transfer restoration — data movement/restoration should not change model math.
6. Streaming vs non-streaming response reconstruction — API transport contract.
7. Duplicate identical requests in the same batch with the same sampling settings — only when `VLLM_BATCH_INVARIANT` is enabled. Without it, different batch positions can produce different output due to cuBLAS kernel selection and accumulation order differences.
8. Same prompt with the same explicit seed in the same engine/request setup.
9. Spec decode exact matching only when the test explicitly forces batch-invariant mode/kernels.
10. Tests under `tests/v1/determinism/` get `VLLM_BATCH_INVARIANT=1` from the autouse `conftest.py` fixture — account for that before classifying as ordinary batch-invariance assumptions.

## Not Strong By Default

These remain suspicious unless the test explicitly establishes a stronger contract:

1. Eager vs compile.
2. Non-compiled vs compiled mode parity hidden inside `compare_two_settings` or `compare_all_settings`.
3. Different compile strategies, graph partitioning strategies, or fused distributed compile passes versus baseline.
4. Tensor parallel vs pipeline parallel vs expert parallel exact generated output equality.
5. Sequence parallel, async TP, or fused distributed compile-pass parity against an unfused baseline.
6. Batch size invariance, including BS=1 vs BS=N, unless batch-invariant kernels/mode are explicitly enabled by the test.
7. Cascade attention vs non-cascade attention when the comparison also changes batch geometry.
8. Spec decoding vs base decoding exact text/token/rank matching, or exact-match ratios, when target verification changes batch geometry and the test does not force batch-invariant mode.
9. Prompt text vs prompt_embeds equality when the only oracle is final generated text.
10. Single request vs first item in a larger multimodal batch, unless the test forces batch-invariant execution.

## Assertion Pattern Guidance

**Exact equality assertions — apply all 3 criteria:**
- `compare_two_settings` / `compare_all_settings` — exact dict equality across two server configs
- `check_outputs_equal` — exact text + token ID equality between two output sequences
- `validate_generated_texts` — exact text equality, cross-runtime (vLLM vs HuggingFace)
- Direct `assert .text ==` or `assert output_ids ==` — inline exact comparison
- Batch duplication (`[prompt] * N`) followed by exact equality check

**Threshold-based assertions — check if threshold is principled:**
- `check_answers` with `accept_rate` — loose match ratio (default 70%)
- Match-ratio patterns (`matches >= int(0.6 * total)`) — spec decode acceptance thresholds
- `check_accuracy` with percentage thresholds — element-wise match ratio

**Tolerance-based assertions — generally contracted, flag only if tolerance is unreasonable:**
- `check_logprobs_close` — allows token divergence, checks top-k logprob overlap
- `check_embeddings_close` — cosine similarity with configurable tolerance
- `torch.testing.assert_close` / `torch.allclose` — numeric tolerance is the contract
- `pytest.approx` — explicit relative/absolute tolerance

## Classifications

| Classification | Meaning | Action |
|---|---|---|
| COINCIDENTALLY_CORRECT | All 3 criteria met | Needs fixing — add BI mode, tolerance, or golden strings |
| STRONG_CONTRACT | Strong contract exists (cite clause) | Remove from list — exact match is correct by design |
| NOT_REALISTIC | Drift won't change outcome | Remove from list — breakage is not realistic |

## Output Format

Both agents write structured JSON using `${CLAUDE_PLUGIN_ROOT}/scripts/output_object.py`. See [example.md](example.md) for complete worked examples.

**Phase 1** uses `AuditCandidate` and `AuditReport`, calls `write_split()` to produce `audit-cc.json` and `audit-not-cc.json`.

**Phase 2** uses `ReviewCandidate` and `ReviewReport`, writes to `review-cc.json`.

Fields per candidate:

| Field | Phase 1 | Phase 2 | Description |
|-------|---------|---------|-------------|
| candidate | yes | yes | Test function name |
| file | yes | yes | Test file path |
| line | yes | yes | Line number of assertion |
| comparison | yes | yes | What two executions are compared |
| oracle | yes | yes | Assertion type |
| helper | yes | yes | Helper function or "direct assertion" |
| batch_invariant_enabled | yes | yes | Whether VLLM_BATCH_INVARIANT is set |
| code_path_verified | yes | yes | Whether test asserts feature ran |
| fixtures | yes | yes | Relevant autouse fixtures |
| c1_weak_oracle | yes | yes | Phase 1: "yes/no — reason". Phase 2: "agree/disagree — reason" |
| c2_realistic_breakage | yes | yes | Same pattern |
| c3_no_strong_contract | yes | yes | Cite clause number |
| classification | yes | yes | COINCIDENTALLY_CORRECT / STRONG_CONTRACT / NOT_REALISTIC |
| coincidentally_correct | yes | yes | true / false |
| code_snippet | yes | yes | Assertion and surrounding context |
| phase_1_classification | — | yes | What Phase 1 said |
| phase_1_coincidentally_correct | — | yes | What Phase 1 said (true/false) |
| review | — | yes | AGREE / RECLASSIFY — reason |
