#!/usr/bin/env bash
# PostToolUse hook (Edit|Write): format the edited Python file with ruff.
# Reads the tool-call JSON from stdin and acts only on .py files.
#
# Format only — deliberately no `ruff check --fix`: an autofix fired after every
# edit would delete a not-yet-used import (F401) between an "add import" edit and
# the "add its first use" edit. Linting is gated once, when the code is stable,
# by the Stop hook (check-on-stop.sh). Always exits 0 so it can't block an edit.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

file=$(jq -r '.tool_input.file_path // empty')

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

uv run ruff format "$file"
exit 0
