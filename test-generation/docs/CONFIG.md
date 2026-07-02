# Configuration Reference

## Complete Configuration File

`.test-gen.yaml` - Place in your project root or `~/.config/test-gen/config.yaml` for global config.

```yaml
# ============================================================================
# Framework Configuration
# ============================================================================

framework:
  # Framework name (built-in: pytorch, vllm, tensorflow, jax, or custom)
  name: pytorch
  
  # Path to source code
  source: /path/to/pytorch
  
  # Test framework to use (pytest, unittest, jest, go-test, etc.)
  test_framework: pytest
  
  # Test file location resolver
  # Type: pattern (use pattern matching) or custom (use script)
  test_file_resolver:
    type: pattern
    # Pattern variables: {component}, {feature}, {module}
    pattern: "test/test_{component}.py"
  
  # Test naming conventions
  naming:
    test_class: "Test{Component}"        # {Component} = capitalized component name
    test_function: "test_{feature}_{scenario}"  # lowercase with underscores
  
  # Imports to add to generated test files
  imports:
    - "import torch"
    - "from torch.testing._internal.common_utils import TestCase"
  
  # Common decorators
  decorators:
    skip: "@skipIf"
    parametrize: "@parametrize"
    device_specific: "@onlyAccelerator"

# ============================================================================
# Issue Tracker Configuration  
# ============================================================================

tracker:
  # Tracker type (jira, github, gitlab, linear, or custom)
  type: jira
  
  # Base URL
  url: https://company.atlassian.net
  
  # Credentials file path (contains API token/key)
  credentials: ~/.config/test-gen/credentials.yaml
  
  # Project key (for JIRA) or repo (for GitHub/GitLab)
  project: AIPCC
  
  # Custom API endpoints (for custom tracker type)
  api:
    get_issue: "/rest/api/3/issue/{issue_id}"
    add_comment: "/rest/api/3/issue/{issue_id}/comment"
    upload_attachment: "/rest/api/3/issue/{issue_id}/attachments"

# ============================================================================
# Test Generation Settings
# ============================================================================

generation:
  # Number of tests to generate (can be range like "5-6")
  count: 5-6
  
  # Output directory for generated tests
  output_dir: /tmp/tests
  
  # Test file naming pattern
  test_file_pattern: "test_{component}.py"
  
  # Include edge case tests
  include_edge_cases: true
  
  # Include regression tests
  include_regression: true
  
  # Test templates directory
  templates:
    basic: ~/.test-gen/templates/basic_test.py.jinja
    edge_case: ~/.test-gen/templates/edge_test.py.jinja
    regression: ~/.test-gen/templates/regression_test.py.jinja
  
  # Template variables
  template_vars:
    author: "Test Generation Bot"
    date_format: "%Y-%m-%d"
    copyright: "Copyright (c) 2026"

# ============================================================================
# Impact Analysis Configuration
# ============================================================================

analyzer:
  # Analyzer type (torchtalk, codeql, custom, or none to skip)
  type: torchtalk
  
  # Analyzer-specific configuration file
  config: ~/.config/torchtalk/config.toml
  
  # Trace depth for call graph analysis
  trace_depth: 5
  
  # Risk levels to assess
  risk_levels: [low, medium, high]
  
  # For custom analyzer
  command: "/path/to/analyzer --input {source} --output {output}"
  output_format: json  # json, yaml, or text
  
  # Expected output schema (for validation)
  schema:
    components: list
    risk_level: string
    affected_files: list

# ============================================================================
# Documentation Configuration
# ============================================================================

documentation:
  # Formats to generate (excel, markdown, pdf, html)
  formats: [excel, markdown]
  
  # Excel-specific settings
  excel_template: ~/.test-gen/templates/test_cases.xlsx
  excel_columns:
    - Test ID
    - Scenario
    - Expected Behavior
    - Test Type
    - Priority
    - Status
  
  # Markdown template
  markdown_template: ~/.test-gen/templates/test_cases.md.jinja
  
  # Include test scenarios
  include_scenarios: true
  
  # Include test code snippets
  include_code: false

# ============================================================================
# Upload Configuration
# ============================================================================

upload:
  # Enable/disable uploading to tracker
  enabled: true
  
  # Attachments to upload (excel, markdown, pdf)
  attachments: [excel]
  
  # Comments to post (tests_summary, impact_analysis, both)
  comments: [tests_summary, impact_analysis]
  
  # Labels to add to the issue
  labels: [automated-tests, generated]
  
  # Assign issue after upload
  assignee: null  # or username
  
  # Update issue status
  transition: null  # or "In Progress", "Ready for Review", etc.

# ============================================================================
# Workflow Customization
# ============================================================================

workflow:
  # Phases to execute (comment out to skip)
  phases:
    - fetch_issue
    - generate_tests
    # - run_tests           # Optional: Run tests before upload
    - impact_analysis
    - documentation
    - upload
  
  # Lifecycle hooks
  hooks:
    pre_generation: null   # Script to run before generation
    post_generation: null  # Script to run after generation
    pre_upload: null       # Script to run before upload
  
  # Timeout for each phase (seconds)
  timeouts:
    fetch_issue: 60
    generate_tests: 600
    impact_analysis: 300
    documentation: 120
    upload: 180

# ============================================================================
# Advanced Settings
# ============================================================================

advanced:
  # Enable debug mode
  debug: false
  
  # Verbose logging
  verbose: false
  
  # Parallel execution
  parallel: false
  
  # Retry failed operations
  retry:
    enabled: true
    max_attempts: 3
    backoff: exponential
  
  # Cache settings
  cache:
    enabled: true
    dir: ~/.cache/test-gen
    ttl: 86400  # 24 hours

# ============================================================================
# Multi-Framework Projects
# ============================================================================

# Override framework per project component
projects:
  backend:
    framework: pytorch
    source: /path/to/pytorch
    test_pattern: "test/test_*.py"
  
  frontend:
    framework: react
    source: /path/to/ui
    test_pattern: "src/**/*.test.tsx"
    test_framework: jest
  
  api:
    framework: fastapi
    source: /path/to/api
    test_pattern: "tests/test_*.py"
```

## Credentials File Format

`~/.config/test-gen/credentials.yaml`:

```yaml
# JIRA
jira:
  email: user@company.com
  token: ATATT3xFfGF0...

# GitHub
github:
  token: ghp_...

# GitLab
gitlab:
  token: glpat-...

# Linear
linear:
  api_key: lin_api_...

# Custom
custom:
  auth_type: bearer  # bearer, basic, api_key
  token: custom_token_...
```

## Environment Variables

Override config with environment variables:

```bash
export TEST_GEN_FRAMEWORK=pytorch
export TEST_GEN_SOURCE=/path/to/source
export TEST_GEN_TRACKER=jira
export TEST_GEN_UPLOAD=true
export TEST_GEN_ANALYZER=torchtalk
```

Priority: CLI args > Environment > Config file > Defaults
