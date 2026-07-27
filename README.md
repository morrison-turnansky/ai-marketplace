# AI Marketplace

[![GitHub](https://img.shields.io/badge/GitHub-ai--marketplace-blue?logo=github)](https://github.com/TorchedHat/ai-marketplace)

A curated collection of Claude Code plugins, skills, and agents for PyTorch development. This marketplace provides AI-powered tools for debugging, development, and optimization across the PyTorch ecosystem.

## What is This?

The PyTorch AI Marketplace is a **Claude Code plugin marketplace** that provides:

- **Pre-built skills** - Ready-to-use AI capabilities for common PyTorch workflows
- **Specialized agents** - Expert AI assistants for specific PyTorch subsystems
- **MCP integrations** - Semantic search and tooling for PyTorch codebases
- **Development tools** - Meta-skills for creating your own skills and agents

This marketplace serves the broader PyTorch community by packaging AI tools that understand PyTorch internals, debugging workflows, and development patterns.

## Contributing

We welcome contributions of new tools, skills, and agents!
See [CONTRIBUTION_GUIDELINES.md](CONTRIBUTION_GUIDELINES.md) for guidelines

## Available Tool Collections

### torch.compile Debugging Tools

Multi-agent system for PyTorch compiler development with 10 skills and 4 specialized agents.

**Features:**
- Stage-specific debugging (Dynamo, AOT Autograd, Inductor)
- Automatic bisection for compilation failures
- Semantic API search over PyTorch compiler modules
- Implementation guidance for compiler development

👉 **[Full documentation](torch-compile/TORCH_COMPILE_TOOLS.md)**

### Distributed Training Hang Diagnosis

Debugging tools for PyTorch distributed training hangs with 1 skill and 1 specialized agent.

**Features:**
- Classification of 7 hang types (barrier deadlock, p2p hang, init timeout, collective mismatch, NCCL timeout, unused parameter, forward order violation)
- Source-level ProcessGroupNCCL architecture reference (communicator lifecycle, watchdog, work queue)
- NCCL flight recorder interpretation and cross-rank comparison
- Anti-rationalization guardrails that prevent the agent from taking diagnostic shortcuts
- Per-pattern verification evidence checklists to confirm fixes

**Usage:**
```
/distributed-hang-diagnosis     # Load hang pattern reference
Use the hang-debugger agent     # Invoke the diagnostic agent
```

### PyTorch Test Refactoring Tools

Skill and agent for refactoring PyTorch test files to be device-agnostic, following the community test refactoring initiative ([pytorch/pytorch#185590](https://github.com/pytorch/pytorch/issues/185590)).

**Features:**
- 4-phase workflow: Analyze, Classify, Refactor, Verify
- Decision tree for 5 hardware classifications (GENERIC, DEVICE_GENERIC, DEVICE_SPECIFIC, MULTI_DEVICE_GENERIC, MULTI_DEVICE_SPECIFIC)
- Before/after code patterns from real merged PRs
- Specialist agent with delegation to compile agents for domain-specific test files

### Installation

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add TorchedHat/ai-marketplace

# Install the plugin(s)
claude plugin install ai-writer
claude plugin install deterministic-hook
claude plugin install torch-compile
claude plugin install torchtalk
claude plugin install pytorch-test-refactor
claude plugin install torch-distributed
```

## Using Marketplace Tools

### Slash Commands

Access skills directly:
```bash
/compile-overview              # torch.compile pipeline reference
/pytorch-dynamo               # Dynamo implementation guidance
/distributed-hang-diagnosis   # Distributed training hang patterns
/skill-writer                 # Create new skills
```

### Natural Language

Skills activate automatically based on context:
```
"Use the debug agent to debug code at file-name.py"
```

### Specialized Agents

Delegate complex tasks to expert agents:
- Use `compile-debug` agent for multi-stage compilation debugging
- Use `dynamo-expert` for Dynamo-specific questions
- Use `aot-expert` for AOT Autograd and gradient issues
- Use `inductor-expert` for lowering and codegen
- Use `hang-debugger` for distributed training hangs and NCCL deadlocks

## For Plugin Developers

This repository serves as a **reference implementation** for Claude Code plugin marketplace patterns:

- **Plugin packaging** - See `.claude-plugin/plugin.json` for plugin structure
- **Marketplace listing** - See `.claude-plugin/marketplace.json` for discovery
- **Skill development** - Use `/skill-writer` to create new skills
- **Agent development** - Use `/agent-writer` to create specialized agents
- **Plugin development** - Use `/plugin-writer` to create specialized agents
- **Auto-setup hooks** - See `hooks.json` and `scripts/ensure-setup.sh`

## Documentation

- **[TORCH_COMPILE_TOOLS.md](torch-compile/TORCH_COMPILE_TOOLS.md)** - torch.compile debugging tools documentation
- **[REPO_ARCH.md](REPO_ARCH.md)** - Architecture and development guide
- **[CLAUDE.md](CLAUDE.md)** - Code guidelines and testing


## Troubleshooting

### MCP Server Errors

```bash
# Check if steering is installed
which acp-steering-mcp

# If not, install manually
uv pip install git+https://github.com/ambient-code/steering.git
```

### Indexing Failed

```bash
# Verify PyTorch is installed
python3 -c "import torch; print(torch.__file__)"

# If PyTorch is installed from source, indexing should auto-detect it
# Override auto-detection if needed:
export PYTORCH_PATH=/path/to/pytorch
```

### Re-index PyTorch

```bash
# Remove existing indices
rm -rf ~/.acp/repos/dynamo ~/.acp/repos/inductor ~/.acp/repos/functorch

# Restart Claude Code to trigger re-indexing
claude --plugin-dir .
```

## License

Part of the PyTorch devcontainer tooling. See [LICENSE](LICENSE) for details.

## Links

- **Repository**: [github.com/TorchedHat/ai-marketplace](https://github.com/TorchedHat/ai-marketplace)
- **Issues**: [GitHub Issues](https://github.com/TorchedHat/ai-marketplace/issues)
- **Documentation**: See individual tool documentation in this repository
- **Contributing**: See [CONTRIBUTION_GUIDELINES.md](CONTRIBUTION_GUIDELINES.md) for guidelines
