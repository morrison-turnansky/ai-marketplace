# torch.compile Debugging Tools

Multi-agent debugging system for PyTorch torch.compile development. Provides stage-specific skills, specialized agents, and semantic API search for Dynamo, AOT Autograd, and Inductor.

## Available Tools

### Skills (10)

**torch.compile debugging:**
- `compile-overview` - torch.compile pipeline architecture and stages reference
- `compile-bisect` - Automatic bisection to find failing operations
- `compile-trace-dynamo` - Debug Dynamo graph capture and tracing
- `compile-trace-aot` - Debug AOT Autograd functionalization and gradients
- `compile-trace-inductor` - Debug Inductor lowering and codegen

**PyTorch implementation:**
- `pytorch-dynamo` - Dynamo implementation guidance and patterns
- `pytorch-aot` - AOT Autograd implementation guidance
- `pytorch-inductor` - Inductor implementation guidance

**Skill/agent development:**
- `skill-writer` - Create new Claude Code skills
- `agent-writer` - Create specialized agent definitions

### Specialized Agents (4)

- `compile-debug` - Multi-stage compilation debugging coordinator
- `dynamo-expert` - Dynamo graph capture and tracing specialist
- `aot-expert` - AOT Autograd functionalization and gradient specialist
- `inductor-expert` - Inductor lowering and codegen specialist

### MCP Server Integration

**steering** - Semantic search over PyTorch modules:
- Dynamo (`torch/_dynamo`)
- Inductor (`torch/_inductor`)
- Functorch (`torch/_functorch`)
- Auto-indexed on first use from PyTorch source

## Usage

### Slash Commands

When using `--plugin-dir`:
```bash
/compile-overview
/pytorch-dynamo
/compile-trace-inductor
```

### Natural Language

Skills load automatically based on context:
```
"How do I debug a graph break in torch.compile?"
"Why isn't my reduction fusing?"
"Show me the Triton kernel for this code"
```

### Example Queries

**Using the compile debug agent:**

The agent is designed to identify the source of an issue not fix an issue.
It will write a report that can serve as a plan for a new agent to ingest.  

#### Do's

The agent is writen to handle queries like the following:
```
Use the torch compile debugging agent to debeug repro.py and file
a report at /path.
```

#### Dont's

The agent is not meant to fix issues.
```
Fix the issue at repro.py.
```

Why?

Context managment is critical for success of the agent. The agent uses a lot of 
deterministic software tools to minimize exploration of the code base 
and therefore keeping context small. Fixing issues requires significant
exploration of the code base and will easily fill context since PyTorch is a
huge repo. This leads to poor outcomes in general.

**PyTorch implementation work:**
```
How do I implement a new VariableTracker type?
Where do I add a lowering for my custom op?
What's the AOT partitioning algorithm?
```

## Prerequisites

1. **Claude Code** installed and configured
2. **Python environment** with `uv` available
3. **PyTorch source** (required for API documentation):
   - Auto-detected from your Python environment if PyTorch is installed from source
   - Override with `PYTORCH_PATH` environment variable if needed

## Architecture

See [REPO_ARCH.md](REPO_ARCH.md) for detailed architecture, including:
- Stage-specific debugging workflow
- Agent delegation patterns
- Skill trigger mechanisms
- MCP server integration
