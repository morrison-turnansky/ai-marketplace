# Repository Architecture

## Overview

Multi-plugin marketplace for PyTorch development tools. Hosts multiple Claude Code plugins including torch.compile debugging, distributed training debugging, and plugin authoring utilities.

## Project Structure

```
ai-marketplace/
├── .claude-plugin/
│   └── marketplace.json     # Marketplace catalog (lists all plugins)
├── torch-compile/           # PyTorch torch.compile debugging plugin
│   ├── .claude-plugin/
│   │   └── plugin.json      # Plugin configuration
│   ├── agents/              # Specialized AI agents
│   │   └── *.md             # Agent definitions
│   ├── skills/              # User-invocable skills
│   │   └── */SKILL.md       # Skill definitions
│   ├── hooks/               # Plugin lifecycle hooks
│   │   └── *.json           # Hook definitions
│   ├── scripts/             # Setup and maintenance scripts
│   │   └── *.sh             # Setup scripts
│   └── settings.json        # MCP server configurations
├── pytorch-test-refactor/   # PyTorch test refactoring tools
│   ├── .claude-plugin/
│   │   └── plugin.json
│   ├── skills/
│   │   └── test-refactor/   # test-refactor skill with supporting docs
│   └── agents/
│       └── test-refactor-agent.md
├── torch-distributed/       # PyTorch distributed training debugging plugin (minimal)
│   ├── .claude-plugin/
│   │   └── plugin.json      # Plugin configuration
│   ├── agents/
│   │   └── hang-debugger.md # Diagnostic agent (loads skill, diagnoses, fixes)
│   ├── skills/
│   │   └── distributed-hang-diagnosis/SKILL.md  # Reference: hang patterns, NCCL arch
│   ├── hooks/
│   │   └── hooks.json       # SessionStart hook
│   ├── scripts/
│   │   └── ensure-setup.sh  # Auto-setup (steering MCP, c10d index)
│   └── settings.json        # MCP server configurations
├── ai-writer/               # Plugin authoring tools
│   ├── .claude-plugin/
│   │   └── plugin.json
│   ├── skills/
│   │   └── */SKILL.md       # plugin-writer, skill-writer, agent-writer
│   ├── agents/
│   ├── hooks/
│   └── scripts/
├── pyproject.toml           # Package metadata
├── LICENSE
├── README.md
└── CLAUDE.md
```

## Architecture

### Marketplace System

**Multi-Plugin Structure:**
- Marketplace hosts multiple plugins as subdirectories
- Each plugin has its own `.claude-plugin/plugin.json`
- Marketplace catalog at `.claude-plugin/marketplace.json` lists all plugins
- Local plugins use relative paths (e.g., `"source": "./torch-compile"`)
- External plugins use git URLs

**Plugin Discovery:**
- Skills in each plugin's `skills/` directory are auto-discovered
- Agents in each plugin's `agents/` directory are auto-discovered
- No explicit listing needed in plugin.json (paths set to `"./skills/"` and agents array)

**Skill Naming:**
- `/plugin-name:skill-name` (e.g., `/torch-compile:compile-bisect`, `/ai-writer:plugin-writer`)

**Agent Definition Format:**
```yaml
---
name: agent-name
skills:
  - skill-one
  - skill-two
callable_agents:
  - other-agent
---
# Agent prompt...
```

### torch-compile Plugin

**Auto-Setup (SessionStart Hook):**
- Installs `acp-steering-mcp` package
- Indexes PyTorch modules (dynamo, inductor, functorch)
- Runs via `torch-compile/scripts/ensure-setup.sh`

### MCP Integration (torch-compile plugin)

**Steering Server:**
- Provides semantic search over PyTorch API documentation
- Indexed modules: torch._dynamo, torch._inductor, torch._functorch
- Configured via `torch-compile/settings.json`
- Auto-configured in plugin.json mcpServers section

**Usage:**
```bash
# Skills can query API docs automatically
# Example: "What's the signature for Pointwise.__init__?"
```

### Plugins

**torch-compile:**
- 8 skills: compile-bisect, compile-overview, compile-trace-{aot,dynamo,inductor}, pytorch-{aot,dynamo,inductor}
- 4 agents: aot-expert, compile-debug, dynamo-expert, inductor-expert
- 1 MCP server: steering (acp-steering-mcp)

**torch-distributed:**
- 1 skill: distributed-hang-diagnosis (hang patterns, NCCL architecture, debugging tools reference)
- 1 agent: hang-debugger (loads skill, diagnoses hang, proposes fix)
- 1 MCP server: steering (acp-steering-mcp, indexes torch/distributed)

**ai-writer:**
- 3 skills: plugin-writer, skill-writer, agent-writer
- Tools for creating and managing Claude Code plugins, skills, and agents

**pytorch-test-refactor:**
- 1 skill: test-refactor (with supporting files: CLASSIFICATION-GUIDE.md, PATTERNS.md, REFERENCES.md)
- 1 agent: test-refactor-agent (delegates to compile agents for domain-specific tests)
- Guides device-agnostic test refactoring with hw_classification for the PyTorch test suite

## Design Principles

### 1. Multi-Plugin Marketplace
Multiple plugins in one repository using subdirectory structure. Each plugin is self-contained with its own configuration, skills, agents, and MCP servers.

### 2. Direct Log Interpretation
Claude reads TORCH_LOGS output, NCCL_DEBUG logs, and debug files directly with skill guidance. No intermediate parsing - full context available for better analysis.

### 3. Start Minimal, Iterate (torch-distributed)
The distributed plugin starts with a single focused skill (hang diagnosis) validated against real PyTorch issues. Additional skills and agents will be added as the distributed team reaches consensus on what's needed.

### 4. Auto-Discovery
Skills and agents are discovered automatically from standard directories. No manual listing in plugin.json needed.

### 5. Portability
No hardcoded paths. Relative paths for local plugins. PyTorch source location auto-detected or configurable via `PYTORCH_PATH` environment variable.

### 6. Self-Contained
All dependencies installed automatically via SessionStart hooks. No manual setup required.

## Documentation

Documentation files:
- `README.md` - User-facing documentation, installation, and troubleshooting
- `REPO_ARCH.md` - This file (architecture and design principles)
- `CLAUDE.md` - Code guidelines and testing practices
- `CONTRIBUTION_GUIDELINES.md` - Development workflow and contribution guidelines

For contributing new plugins, skills, or agents, see [CONTRIBUTION_GUIDELINES.md](CONTRIBUTION_GUIDELINES.md).

## References

- **Plugin Documentation**: https://code.claude.com/docs/en/plugins
- **Skill Documentation**: https://code.claude.com/docs/en/skills
- **MCP Documentation**: https://code.claude.com/docs/en/mcp
