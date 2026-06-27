#!/usr/bin/env bash
# PostToolUse hook (Edit|Write|MultiEdit): auto-format and lint-fix the edited
# Python file with ruff. Reads the tool-call JSON from stdin and acts only on
# .py files. Always exits 0 so a lint finding never blocks the edit; remaining
# violations print to the transcript and are caught by the Stop gate's checks.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

file=$(jq -r '.tool_input.file_path // empty')

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

uv run ruff format "$file"
uv run ruff check --fix "$file"
exit 0
