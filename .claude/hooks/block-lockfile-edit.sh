#!/usr/bin/env bash
# PreToolUse hook (Edit|Write): block direct edits to uv.lock.
#
# uv.lock is generated and must only change via uv (uv add/remove, uv lock,
# uv sync). A hand-edit can desync it from pyproject.toml and break resolution.
# Exit 2 blocks the tool call and feeds the reason back so Claude uses uv
# instead; any other path is allowed through (exit 0). Reads the tool-call JSON
# from stdin like ruff-on-edit.sh.
set -uo pipefail

file=$(jq -r '.tool_input.file_path // empty')

case "$file" in
  */uv.lock | uv.lock)
    {
      echo "Refusing to edit uv.lock directly — it is generated. Use uv instead:"
      echo "  uv add/remove <pkg>   change a dependency (updates pyproject + lock)"
      echo "  uv lock --upgrade     refresh the lock"
      echo "  uv sync               apply the lock to .venv"
    } >&2
    exit 2
    ;;
esac
exit 0
