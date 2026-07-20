#!/bin/bash
# Auto-setup script for torch-distributed plugin.
# Runs on SessionStart to ensure dependencies are available.

set -e

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INDICES="$HOME/.acp/repos"

if [ -t 1 ]; then
    QUIET=""
else
    QUIET="-q"
fi

echo "Checking torch-distributed dependencies..."

# 1. Verify PyTorch is importable
if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch is not installed. Install with: pip install torch"
    exit 1
fi

# 2. Install acp-steering-mcp if missing
if ! command -v acp-steering-mcp &> /dev/null; then
    echo "Installing acp-steering-mcp..."

    if command -v uv &> /dev/null; then
        if [ -n "$VIRTUAL_ENV" ]; then
            uv pip install $QUIET "git+https://github.com/ambient-code/steering.git" 2>/dev/null || {
                echo "Failed to install acp-steering-mcp (non-blocking)"
                exit 0
            }
        else
            uv pip install --system $QUIET "git+https://github.com/ambient-code/steering.git" 2>/dev/null || {
                echo "Failed to install acp-steering-mcp (non-blocking)"
                exit 0
            }
        fi
    elif command -v pip &> /dev/null; then
        pip install $QUIET "git+https://github.com/ambient-code/steering.git" 2>/dev/null || {
            echo "Failed to install acp-steering-mcp (non-blocking)"
            exit 0
        }
    else
        echo "Neither uv nor pip found"
        exit 0
    fi
fi

# 3. Index c10d module (only module needed for hang debugging)
if [ ! -f "$INDICES/c10d/steering.json" ]; then
    PYTORCH_SRC=""

    if [[ -n "${PYTORCH_PATH:-}" && -d "${PYTORCH_PATH}" ]]; then
        PYTORCH_SRC="$PYTORCH_PATH"
    else
        TORCH_LOCATION=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))" 2>/dev/null || echo "")
        if [[ -n "$TORCH_LOCATION" && -d "$TORCH_LOCATION/distributed" ]]; then
            PYTORCH_SRC="$(dirname "$TORCH_LOCATION")"
        fi
    fi

    if [ -n "$PYTORCH_SRC" ] && [ -d "$PYTORCH_SRC/torch/distributed" ]; then
        echo "Indexing c10d module (one-time setup)..."
        mkdir -p "$INDICES"
        cd "$PYTORCH_SRC"
        if command -v repomap &>/dev/null; then
            repomap ./torch/distributed --repo-name c10d --verbose > /dev/null 2>&1 && \
                echo "  c10d indexed" || echo "  c10d indexing failed (non-blocking)"
        else
            echo "  repomap not found, skipping c10d indexing (non-blocking)"
        fi
    else
        echo "PyTorch source not found. Set PYTORCH_PATH for API lookups."
    fi
fi

echo "torch-distributed ready"
