# Test Generation Plugin

Generic test generation for any framework with automated impact analysis and issue tracker integration.

## Features

✅ **Framework Agnostic** - Works with PyTorch, vLLM, TensorFlow, JAX, React, and custom frameworks  
✅ **Multi-Tracker Support** - JIRA, GitHub Issues, GitLab Issues, Linear  
✅ **Impact Analysis** - TorchTalk, CodeQL, or custom analyzers  
✅ **Auto Documentation** - Excel and Markdown test case documentation  
✅ **JIRA Integration** - Automatic upload of tests and analysis  
✅ **Highly Configurable** - YAML-based configuration with sensible defaults

## Quick Start

### 1. Install Plugin

```bash
claude plugin marketplace add TorchedHat/ai-marketplace
claude plugin install test-generation
```

### 2. Configure (One-Time Setup)

Create `.test-gen.yaml` in your project:

```yaml
framework:
  name: pytorch
  source: /path/to/pytorch
  test_framework: pytest

tracker:
  type: jira
  url: https://your-company.atlassian.net
  credentials: ~/.config/test-gen/credentials.yaml

analyzer:
  type: torchtalk
  config: ~/.config/torchtalk/config.toml

upload:
  enabled: true
```

Create `~/.config/test-gen/credentials.yaml`:

```yaml
jira:
  email: your-email@company.com
  token: YOUR_JIRA_API_TOKEN
```

### 3. Generate Tests

```bash
/generate-tests AIPCC-12345
```

That's it! The skill will:
1. Fetch the JIRA issue
2. Generate 5-6 pytest tests
3. Perform TorchTalk impact analysis
4. Create Excel documentation
5. Upload everything to JIRA

## Examples

### PyTorch + JIRA + TorchTalk

```bash
/generate-tests AIPCC-17797
```

**Output:**
- 6 tests in `/tmp/pytorch_AIPCC-17797/test/test_cuda.py`
- Excel: `/tmp/TestCases_AIPCC-17797.xlsx`
- JIRA updated with test summary + impact analysis

### vLLM + GitHub

```bash
/generate-tests vllm-project/vllm#12345 \
  --framework vllm \
  --tracker github
```

**Output:**
- 4-6 tests in vLLM test directory
- Markdown documentation
- GitHub comment with analysis

### TensorFlow + GitLab

```bash
/generate-tests PROJECT!456 \
  --framework tensorflow \
  --tracker gitlab
```

### Minimal (No Upload, No Analysis)

```bash
/generate-tests ISSUE-123 --no-upload --no-analyze
```

## Preset Configurations

We provide ready-to-use configs for common setups:

### PyTorch (JIRA + TorchTalk)

```bash
cp config/presets/pytorch-jira.yaml .test-gen.yaml
# Edit paths and credentials
/generate-tests AIPCC-12345
```

### vLLM (GitHub + TorchTalk)

```bash
cp config/presets/vllm-github.yaml .test-gen.yaml
# Edit paths and credentials
/generate-tests owner/repo#123
```

### Custom Framework

```bash
cp config/presets/custom-template.yaml .test-gen.yaml
# Customize for your framework
/generate-tests ISSUE-123
```

## Documentation

- **[SKILL.md](skills/generate-tests/SKILL.md)** - Complete skill documentation
- **[CONFIG.md](docs/CONFIG.md)** - Configuration reference
- **[FRAMEWORKS.md](docs/FRAMEWORKS.md)** - Framework support guide
- **[TRACKERS.md](docs/TRACKERS.md)** - Issue tracker integrations
- **[ANALYZERS.md](docs/ANALYZERS.md)** - Impact analyzer guide
- **[EXAMPLES.md](examples/EXAMPLES.md)** - Real-world examples

## Supported Frameworks

### Built-in

- **PyTorch** - Full TorchTalk integration, pytest conventions
- **vLLM** - vLLM-specific test patterns, TorchTalk support
- **TensorFlow** - tf.test patterns
- **JAX** - jax.test_util patterns

### Custom

Any framework can be supported via configuration:

```yaml
framework:
  name: custom
  source: /path/to/code
  test_framework: pytest
  test_file_resolver:
    type: pattern
    pattern: "tests/{component}/test_{feature}.py"
```

## Supported Issue Trackers

- **JIRA** - Full integration with attachments and comments
- **GitHub Issues** - Comment and label support
- **GitLab Issues** - Note and label support
- **Linear** - Comment support
- **Custom** - API-based integration

## Supported Impact Analyzers

- **TorchTalk** - Cross-language tracing (Python → C++ → CUDA) for PyTorch/vLLM
- **CodeQL** - Language-agnostic static analysis
- **Custom** - Any command-line tool or script
- **None** - Skip impact analysis

## Migration from Workflow Scripts

If you have existing test generation workflows:

**Before:**
```javascript
Workflow({
  scriptPath: "/path/to/test-gen-simple.js",
  args: "AIPCC-12345"
})
```

**After:**
```bash
# One-time: Create config from your workflow settings
cp config/presets/pytorch-jira.yaml .test-gen.yaml

# Use skill
/generate-tests AIPCC-12345
```

The skill uses the same underlying workflow logic but with cleaner configuration.

## Real-World Results

### Example: AIPCC-17797 (PyTorch)

**Input:**
```bash
/generate-tests AIPCC-17797
```

**Output:**
- ✅ 6 tests generated (173 lines)
- ✅ Test file: `test/test_cuda.py` line 9994
- ✅ Excel uploaded to JIRA
- ✅ Impact analysis: HIGH risk, 6 functions traced, 9 modules affected
- ✅ 2 JIRA comments posted
- ⏱️ Execution time: 16 minutes

### Example: AIPCC-18571 (vLLM)

**Input:**
```bash
/generate-tests AIPCC-18571 --framework vllm
```

**Output:**
- ✅ 6 tests generated (326 lines)
- ✅ Test file: `tests/v1/core/test_single_type_kv_cache_manager.py` line 489
- ✅ Impact analysis: HIGH risk, 8 components, 10+ models identified
- ✅ Complete JIRA documentation
- ⏱️ Execution time: 18.5 minutes

## Architecture

```
┌─────────────┐
│   Issue     │
│  Tracker    │
└──────┬──────┘
       │ Fetch
       ▼
┌─────────────┐     ┌──────────────┐
│    Test     │────▶│   Impact     │
│ Generation  │     │  Analyzer    │
└──────┬──────┘     └──────┬───────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌──────────────┐
│Documentation│     │   Upload     │
│  Generator  │────▶│   Results    │
└─────────────┘     └──────────────┘
```

## Troubleshooting

**Issue not found:**
```bash
/generate-tests ISSUE-123 --debug
```

**Tests in wrong file:**
Check your `test_file_pattern` in config.

**Impact analysis fails:**
```bash
/generate-tests ISSUE-123 --no-analyze
```

**Upload fails:**
Verify credentials in `~/.config/test-gen/credentials.yaml`

## Contributing

Contributions welcome! Please:
1. Test with your framework
2. Add preset config if useful
3. Document any new features
4. Submit PR to ai-marketplace

## License

MIT License - See LICENSE file

## Acknowledgments

- Built for PyTorch QA workflows
- Powered by TorchTalk for impact analysis
- Tested with JIRA, GitHub, and GitLab
- Used in production for PyTorch hermetic testing
