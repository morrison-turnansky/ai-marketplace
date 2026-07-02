---
name: generate-tests
description: Generate comprehensive test cases for any framework from issue tracker tickets with optional impact analysis
---

# Generic Test Generation Skill

Generate comprehensive test cases for any codebase from issue tracker tickets, with optional impact analysis and automated documentation.

## When to Use

- Generate tests for **any framework** (PyTorch, vLLM, TensorFlow, JAX, etc.)
- Works with **any issue tracker** (JIRA, GitHub, GitLab, Linear, etc.)
- Supports **any test framework** (pytest, unittest, Jest, Go test, etc.)
- Optional **impact analysis** (pluggable analyzers)
- Automated **documentation generation** (Excel, Markdown, etc.)

---

## Usage

### Basic (Minimal)

```bash
# Just generate tests
/generate-tests ISSUE-123

# Specify framework
/generate-tests ISSUE-123 --framework pytorch

# Full configuration
/generate-tests ISSUE-123 \
  --framework pytorch \
  --tracker jira \
  --analyzer torchtalk \
  --upload
```

### With Configuration File

```bash
# Uses config from .test-gen.yaml
/generate-tests ISSUE-123
```

---

## Configuration

### Option 1: Command-line Arguments

```bash
/generate-tests ISSUE-123 \
  --framework pytorch \              # Framework to test
  --source /path/to/source \         # Source code location
  --tracker jira \                   # Issue tracker type
  --tracker-url https://jira.com \   # Tracker URL
  --output /tmp/tests \              # Test output location
  --analyzer torchtalk \             # Impact analyzer (optional)
  --upload \                         # Upload results to tracker
  --format excel,markdown            # Output formats
```

### Option 2: Configuration File

Create `.test-gen.yaml` in your project:

```yaml
# Framework configuration
framework:
  name: pytorch              # pytorch, vllm, tensorflow, jax, custom
  source: /path/to/pytorch
  test_framework: pytest     # pytest, unittest, jest, go-test
  
# Issue tracker configuration
tracker:
  type: jira                 # jira, github, gitlab, linear
  url: https://company.atlassian.net
  credentials: ~/.config/test-gen/credentials.yaml
  project: AIPCC
  
# Test generation settings
generation:
  count: 5-6                 # Number of tests to generate
  output_dir: /tmp/tests
  test_file_pattern: "test_{component}.py"
  include_edge_cases: true
  include_regression: true
  
# Impact analysis (optional)
analyzer:
  type: torchtalk           # torchtalk, codeql, custom, none
  config: ~/.config/torchtalk/config.toml
  trace_depth: 5
  risk_levels: [low, medium, high]
  
# Documentation (optional)
documentation:
  formats: [excel, markdown]
  excel_template: ~/.config/test-gen/template.xlsx
  include_scenarios: true
  
# Upload settings (optional)
upload:
  enabled: true
  attachments: [excel, markdown]
  comments: [tests_summary, impact_analysis]
  labels: [automated-tests, generated]
```

---

## Architecture

### Phases

**Phase 1 - Fetch Issue**
- Retrieves issue from tracker
- Extracts relevant information
- Identifies affected components

**Phase 2 - Generate Tests**
- Creates test cases based on issue
- Follows framework conventions
- Includes edge cases and regression tests

**Phase 3 - Impact Analysis (Optional)**
- Analyzes code changes
- Traces dependencies
- Assesses risk level
- Identifies affected modules

**Phase 4 - Documentation (Optional)**
- Generates test documentation
- Creates Excel/Markdown reports
- Formats for stakeholders

**Phase 5 - Upload (Optional)**
- Uploads tests to repository
- Attaches documentation to issue
- Posts summary comments

---

## Framework Support

### Built-in Frameworks

**PyTorch**
- Test framework: pytest
- Impact analyzer: TorchTalk
- Test patterns: torch.testing._internal
- File patterns: test/test_*.py

**vLLM**
- Test framework: pytest
- Impact analyzer: TorchTalk (vLLM mode)
- Test patterns: vLLM test infrastructure
- File patterns: tests/*/test_*.py

**TensorFlow**
- Test framework: pytest/tf.test
- Impact analyzer: Custom
- Test patterns: tf.test.TestCase
- File patterns: tensorflow/python/*/tests/*_test.py

**JAX**
- Test framework: pytest/absltest
- Impact analyzer: Custom
- Test patterns: jax.test_util
- File patterns: tests/*_test.py

### Custom Frameworks

Define your own in `.test-gen.yaml`:

```yaml
framework:
  name: custom
  source: /path/to/code
  test_framework: pytest
  
  # Test file location logic
  test_file_resolver:
    type: pattern
    pattern: "tests/{component}/test_{feature}.py"
  
  # Test class/function naming
  naming:
    test_class: "Test{Component}"
    test_function: "test_{feature}_{scenario}"
  
  # Imports to add
  imports:
    - "from myframework.testing import TestCase"
    - "import myframework as mf"
  
  # Test decorators
  decorators:
    skip: "@skip_if_no_gpu"
    parametrize: "@parametrize"
```

---

## Issue Tracker Support

### Built-in Trackers

**JIRA**
- Credentials: API token
- Issue format: PROJECT-123
- Fields: summary, description, labels, attachments, comments

**GitHub Issues**
- Credentials: Personal access token
- Issue format: owner/repo#123
- Fields: title, body, labels, attachments, comments

**GitLab Issues**
- Credentials: Personal access token
- Issue format: project!123
- Fields: title, description, labels, attachments, notes

**Linear**
- Credentials: API key
- Issue format: TEAM-123
- Fields: title, description, labels, attachments, comments

### Custom Trackers

```yaml
tracker:
  type: custom
  api:
    base_url: https://tracker.company.com/api
    auth_type: bearer  # bearer, basic, api_key
    auth_header: Authorization
  
  endpoints:
    get_issue: "/issues/{issue_id}"
    add_comment: "/issues/{issue_id}/comments"
    upload_attachment: "/issues/{issue_id}/attachments"
  
  fields:
    summary: "title"
    description: "body"
    labels: "tags"
```

---

## Impact Analyzer Support

### Built-in Analyzers

**TorchTalk (PyTorch/vLLM)**
- Traces: Python → C++ → CUDA
- Output: Component graph, risk level, affected modules
- Config: ~/.config/torchtalk/config.toml

**CodeQL**
- Traces: Language-agnostic AST analysis
- Output: Call graph, data flow, affected files
- Config: codeql database path

**Custom Static Analysis**
- Traces: grep/ripgrep based
- Output: File patterns, function calls
- Config: Search patterns

**None**
- Skip impact analysis
- Just generate tests

### Custom Analyzers

```yaml
analyzer:
  type: custom
  
  # Command to run
  command: "/path/to/analyzer --input {source} --output {output}"
  
  # Expected output format
  output_format: json  # json, yaml, text
  
  # Output schema
  schema:
    components: list[string]
    risk_level: string
    affected_files: list[string]
    recommendations: list[string]
```

---

## Output Formats

### Test Files

Generated tests follow framework conventions:

```python
# PyTorch example
import torch
from torch.testing._internal.common_utils import TestCase

class TestFeature(TestCase):
    def test_basic_functionality(self):
        # ISSUE-123
        ...
    
    def test_edge_case(self):
        # ISSUE-123
        ...
```

### Documentation

**Excel Format:**
```
| Test ID | Scenario | Expected Behavior | Test Type | Priority |
|---------|----------|-------------------|-----------|----------|
| TC-1    | Basic    | ...              | Functional| High     |
| TC-2    | Edge     | ...              | Edge Case | Medium   |
```

**Markdown Format:**
```markdown
# Test Cases for ISSUE-123

## Summary
- Tests generated: 6
- Coverage: Component X, Y, Z
- Risk level: HIGH

## Tests

### TC-1: Basic Functionality
**Scenario:** ...
**Expected:** ...
**Type:** Functional
```

---

## Examples

### Example 1: PyTorch with JIRA

```bash
/generate-tests AIPCC-17797 \
  --framework pytorch \
  --tracker jira \
  --analyzer torchtalk \
  --upload
```

**Output:**
- 6 tests in `/tmp/pytorch_AIPCC-17797/test/test_cuda.py`
- Excel: `/tmp/TestCases_AIPCC-17797.xlsx`
- JIRA updated with 2 comments

### Example 2: vLLM with GitHub

```bash
/generate-tests vllm-project/vllm#12345 \
  --framework vllm \
  --tracker github \
  --analyzer torchtalk \
  --upload
```

**Output:**
- 4 tests in `/tmp/vllm_12345/tests/core/test_feature.py`
- Markdown: `/tmp/TestCases_12345.md`
- GitHub comment with analysis

### Example 3: Custom Framework with GitLab

```bash
/generate-tests PROJECT!456 \
  --framework custom \
  --tracker gitlab \
  --analyzer codeql \
  --format markdown
```

**Output:**
- Tests in configured location
- Markdown documentation
- GitLab note with summary

### Example 4: Minimal (Just Generate Tests)

```bash
/generate-tests ISSUE-123
```

**Output:**
- Tests only (no analysis, no upload)
- Uses defaults from `.test-gen.yaml`

---

## Advanced Features

### Workflow Customization

Override workflow phases:

```yaml
workflow:
  phases:
    - fetch_issue
    - generate_tests
    - run_tests          # NEW: Run tests before upload
    - impact_analysis
    - documentation
    - upload
  
  # Custom hooks
  hooks:
    pre_generation: "/path/to/hook.sh"
    post_generation: "/path/to/validator.sh"
    pre_upload: "/path/to/review.sh"
```

### Test Templates

Use custom test templates:

```yaml
generation:
  templates:
    basic: "~/.test-gen/templates/basic_test.py.jinja"
    edge_case: "~/.test-gen/templates/edge_test.py.jinja"
    regression: "~/.test-gen/templates/regression_test.py.jinja"
  
  template_vars:
    author: "Test Generation Bot"
    date_format: "%Y-%m-%d"
```

### Multi-Framework Projects

```yaml
frameworks:
  - name: backend
    framework: pytorch
    source: /path/to/pytorch
    test_pattern: "test/test_*.py"
  
  - name: frontend
    framework: react
    source: /path/to/ui
    test_pattern: "src/**/*.test.tsx"
    test_framework: jest
```

---

## Output Schema

```json
{
  "issue_key": "ISSUE-123",
  "framework": "pytorch",
  "tests_generated": {
    "count": 6,
    "file_path": "/tmp/pytorch_ISSUE-123/test/test_feature.py",
    "start_line": 489,
    "lines_added": 326,
    "test_class": "TestFeature",
    "tests": [
      {
        "name": "test_basic_functionality",
        "description": "...",
        "type": "functional"
      }
    ]
  },
  "impact_analysis": {
    "analyzer": "torchtalk",
    "risk_level": "high",
    "components_identified": ["ComponentA", "ComponentB"],
    "affected_modules": ["module1", "module2"],
    "existing_tests": ["test_file1.py", "test_file2.py"],
    "recommended_tests": ["test_integration.py"]
  },
  "documentation": {
    "excel": "/tmp/TestCases_ISSUE-123.xlsx",
    "markdown": "/tmp/TestCases_ISSUE-123.md"
  },
  "upload": {
    "attachments_uploaded": ["TestCases_ISSUE-123.xlsx"],
    "comments_posted": ["tests_summary", "impact_analysis"]
  }
}
```

---

## Error Handling

**Graceful degradation:**

1. **Issue fetch fails** → Prompt user for manual input
2. **Test generation fails** → Return partial results
3. **Impact analysis fails** → Skip analysis, continue
4. **Upload fails** → Save locally, return paths
5. **Analyzer not available** → Fall back to basic analysis

**Validation:**

- Verify source code exists
- Check tracker credentials
- Validate test file syntax
- Ensure no test name collisions

---

## Extension Points

### Custom Test Generators

```python
# ~/.test-gen/generators/custom_generator.py

def generate_tests(issue_data, config):
    """
    Custom test generation logic.
    
    Args:
        issue_data: Parsed issue information
        config: Configuration from .test-gen.yaml
    
    Returns:
        {
            "tests": [...],
            "file_path": "...",
            "count": N
        }
    """
    tests = []
    # Your custom logic here
    return {"tests": tests, "file_path": "...", "count": len(tests)}
```

Register in config:

```yaml
generation:
  generator: custom
  generator_path: ~/.test-gen/generators/custom_generator.py
```

### Custom Impact Analyzers

```python
# ~/.test-gen/analyzers/custom_analyzer.py

def analyze_impact(issue_data, source_path, config):
    """
    Custom impact analysis logic.
    
    Returns:
        {
            "risk_level": "high",
            "components": [...],
            "affected_modules": [...]
        }
    """
    # Your analysis logic
    return result
```

---

## Migration Guide

### From PyTorch-Specific Workflow

**Before:**
```javascript
Workflow({
  scriptPath: "/path/to/test-gen-simple.js",
  args: "AIPCC-12345"
})
```

**After:**
```bash
# One-time setup
cat > .test-gen.yaml << EOF
framework:
  name: pytorch
  source: /path/to/pytorch
tracker:
  type: jira
  url: https://redhat.atlassian.net
analyzer:
  type: torchtalk
EOF

# Use skill
/generate-tests AIPCC-12345
```

### From vLLM-Specific Workflow

**Before:**
```javascript
Workflow({
  scriptPath: "/path/to/test-gen-vllm-jira.js",
  args: "AIPCC-18571"
})
```

**After:**
```bash
# Update config
sed -i 's/name: pytorch/name: vllm/' .test-gen.yaml

# Use skill
/generate-tests AIPCC-18571
```

---

## Troubleshooting

**Issue not found:**
```bash
# Check credentials
/generate-tests ISSUE-123 --debug

# Manual override
/generate-tests --issue-data '{"summary": "...", "description": "..."}'
```

**Tests not generated:**
```bash
# Verbose mode
/generate-tests ISSUE-123 --verbose

# Use different generator
/generate-tests ISSUE-123 --generator basic
```

**Impact analysis fails:**
```bash
# Skip analysis
/generate-tests ISSUE-123 --no-analyze

# Use different analyzer
/generate-tests ISSUE-123 --analyzer custom
```

---

## Best Practices

1. **Start simple** - Use minimal config, add features as needed
2. **Version config** - Check `.test-gen.yaml` into git
3. **Test locally** - Run without `--upload` first
4. **Review tests** - Always review generated tests before merging
5. **Iterate** - Use `--no-analyze` for faster iteration
6. **Document** - Add framework-specific patterns to config

---

## See Also

- [Configuration Reference](CONFIG.md)
- [Framework Adapters](FRAMEWORKS.md)
- [Impact Analyzers](ANALYZERS.md)
- [Issue Tracker Integrations](TRACKERS.md)
- [Examples](EXAMPLES.md)
