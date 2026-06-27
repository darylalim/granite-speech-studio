#!/usr/bin/env bash
# Stop hook: full type-check + test gate, run once a task completes.
#
# - Skips entirely when no Python files have uncommitted changes (so a
#   conversation-only turn doesn't run the suite).
# - On failure, blocks the stop (exit 2) and feeds the output back so Claude
#   fixes it before finishing.
# - Loop-safe: if this stop is already a retry triggered by this hook
#   (stop_hook_active), it reports but allows the stop, so it can't loop forever.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

input=$(cat)

# Nothing Python changed -> nothing to gate.
if ! git status --porcelain 2>/dev/null | grep -qE '\.py"?$'; then
  exit 0
fi

active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')

ty_out=$(uv run ty check 2>&1); ty_status=$?
test_out=$(uv run pytest -q 2>&1); test_status=$?

if [ "$ty_status" -eq 0 ] && [ "$test_status" -eq 0 ]; then
  exit 0
fi

# Already retried once this turn: surface results but let the stop proceed.
if [ "$active" = "true" ]; then
  exit 0
fi

{
  echo "Stop gate failed — fix before finishing:"
  [ "$ty_status" -ne 0 ] && { echo "--- ty check ---"; echo "$ty_out"; }
  [ "$test_status" -ne 0 ] && { echo "--- pytest ---"; echo "$test_out"; }
} >&2
exit 2
