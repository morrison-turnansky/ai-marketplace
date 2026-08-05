---
name: test-refactor-review
description: "Review PyTorch test refactoring PRs for hw_classification, code patterns, decorators, and PR structure. Use when reviewing refactoring draft PRs, when asked to review refactoring changes, or when the user mentions 'review refactoring PR'."
---

# PyTorch Test Refactoring PR Review

Review any device-agnostic test refactoring PR against upstream PyTorch maintainer expectations. Modeled after PyTorch's own `/pr-review` skill.

## Usage Modes

- **No argument**: Ask what to review
- **PR number**: `/test-refactor-review 192128` — fetches via `gh pr diff 192128 --repo pytorch/pytorch`
- **PR URL**: `/test-refactor-review https://github.com/pytorch/pytorch/pull/192128`
- **Local branch**: `/test-refactor-review my-branch` — diffs against `main`
- **Detailed mode**: `/test-refactor-review 192128 detailed` — adds inline comments with exact line numbers

## Review Philosophy

These principles are adapted from PyTorch's upstream `/pr-review` skill and govern every review:

1. **Only report problems** — no praise, no "looks good", no filler. If every section passes, the review is short.
2. **Investigate, don't guess** — before flagging an issue, read the original file (not just the diff). Spawn sub-agents to read surrounding code when needed. Never flag something you haven't verified.
3. **Focus on what CI cannot check** — CI catches linting, import order, and syntax. Focus on classification correctness, structural integrity (no deleted tests, setUp preserved), pattern adherence, and decorator usage. Don't duplicate CI's work.
4. **Everything is a must-fix** — no nits, no suggestions, no "consider doing X". If it's worth mentioning, it blocks approval.
5. **Be specific and actionable** — every finding includes the file path, line number, what's wrong, and the exact fix. Never say "this might be wrong" — say what IS wrong and what to change.
6. **Match existing patterns** — read how the file was structured before the PR. If the file uses `TestFooDeviceType` naming, don't flag it for not using `TestFooDevice`.
7. **Assume competence** — don't explain what `hw_classification` is or how `instantiate_device_type_tests` works. Only explain non-obvious context (e.g., why a profiler test must stay CUDA-specific).

## Review Workflow

### Step 1: Understand Context

```bash
gh pr view <number> --repo pytorch/pytorch --json title,body,files
gh pr diff <number> --repo pytorch/pytorch
```

Determine:
- Which test file(s) are being refactored
- How many classes are modified
- Whether this is a split, a classification-only change, or a full device-agnostic conversion
- Whether the file is in a specialized domain (profiler, dynamo, distributed)

### Step 2: Deep Review

Walk through every changed line against [review-checklist.md](review-checklist.md). Check each category:
- Classification — hw_classification on every class, correct enum values, ACCELERATOR ↔ instantiate_device_type_tests consistency
- Structure — test count preserved, setUp/tearDown duplicated, helpers at first use
- Code Patterns — no hardcoded devices, correct API conversions, profiler API
- Decorators — correct skip/only/dtype decorators, multi-device pattern
- CI Compliance — Owner comment, main block, PR size
- PR Format — title, single concern, test plan, AI disclosure

### Step 3: Check Refactoring Guidelines

Apply domain-specific rules from [refactoring-guidelines.md](refactoring-guidelines.md):
- If profiler file → check profiler organization rules (TestProfiler vs TestProfilerDevice vs TestProfilerCUDA)
- If dynamo file → check what NOT to parameterize (pure tracing tests stay GENERIC)
- Check common mistakes table for known pitfalls
- Check anti-pattern table for severity-tagged issues

### Step 4: Cross-Reference PRECEDENTS.md

If the `pytorch-test-refactor` plugin is installed, read its `PRECEDENTS.md` for reviewer corrections that apply to this file's domain. PRECEDENTS.md overrides generic guidance.

### Step 5: Formulate Review

Structure findings by category. Only include categories that have problems — skip clean categories entirely.

### Step 6: Fact-Check

For each finding:
1. Re-read the relevant code in the original file (pre-PR) and the diff
2. Confirm the issue is real, not a misread of the diff
3. Verify the suggested fix is correct
4. If uncertain, spawn a sub-agent to investigate

Drop any finding that doesn't survive fact-checking.

## Output Format

```markdown
## PR Review: #<number>

### Summary
<What the PR does in one sentence>

**Verdict: Approve / Request Changes / Needs Discussion**

### Classification
<Problems only — hw_classification errors, wrong enum values, missing classifications>

### Structure
<Problems only — deleted tests, missing setUp, wrong only_for, orphaned helpers>

### Code Patterns
<Problems only — hardcoded devices, wrong API usage, deprecated APIs>

### Decorators
<Problems only — wrong skip/only/dtype decorators, missing migrations>

### CI Compliance
<Problems only — missing # Owner(s):, missing main block, PR too large>

### PR Format
<Problems only — wrong title, missing test plan, missing AI disclosure, unclassified > 0>

### Recommendation

**Approve** / **Request Changes** / **Needs Discussion**

<Brief justification — 1-2 sentences>
```

### Verdict Rules

- **Approve**: All checklist items pass. No critical or warning findings.
- **Request Changes**: Any critical finding (missing hw_classification, wrong enum, deleted tests, unclassified > 0). Also for 3+ warnings.
- **Needs Discussion**: Classification is ambiguous (e.g., profiler test that could be GENERIC or ACCELERATOR), or the refactoring approach conflicts with PRECEDENTS.md guidance.

### Section Rules

- Omit any section with no findings — don't write "No issues found"
- Each finding format: `**[file:line]** Description → Fix`
- Group related findings (e.g., multiple hardcoded devices in the same class)
- Order findings by severity within each section (critical first)

## Sub-Agent Usage

Spawn sub-agents when:
- The PR touches multiple files — one sub-agent per file
- A finding needs verification against the original file
- The file is in a domain you need specialist input on (dynamo, inductor, AOT)
- The diff is large (>500 lines) — split review across sub-agents by category

Each sub-agent should:
- Receive the specific file diff and the relevant checklist section
- Return findings in the standard format
- Be fact-checked before inclusion in the final review

## References

- [review-checklist.md](review-checklist.md) — Full review checklist by category
- [refactoring-guidelines.md](refactoring-guidelines.md) — Domain-specific rules, common mistakes, anti-patterns
- [HardwareClassification PR #186918](https://github.com/pytorch/pytorch/pull/186918) — Merged infrastructure
- [Tracking issue #185590](https://github.com/pytorch/pytorch/issues/185590) — Test refactoring initiative
