#!/usr/bin/env bash
# Stop hook: lint + type-check + test gate, run when Claude finishes a turn.
#
# - Skips when no Python files have uncommitted changes (so a conversation-only
#   turn doesn't run the suite). Note: .py changes committed earlier in the turn
#   leave a clean tree and are not re-gated here.
# - On failure, blocks the stop (exit 2) and feeds the output back so Claude
#   fixes it before finishing.
# - Loop-safe: if this stop is already a retry triggered by this hook
#   (stop_hook_active), it allows the stop (exit-0 output isn't re-shown anyway),
#   so it can't loop forever.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

input=$(cat)

# Nothing Python changed -> nothing to gate. --untracked-files=all expands a
# brand-new untracked directory so a new package of .py files is still seen; the
# grep also matches a .py followed by a quote (special chars) or " -> " (rename).
if ! git status --porcelain --untracked-files=all 2>/dev/null | grep -qE '\.py("|$| )'; then
  exit 0
fi

# Fail safe: if jq is missing or errors, treat as already-retried (allow the
# stop) rather than risk re-blocking forever.
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false') || active=true

ruff_out=$(uv run ruff check . 2>&1); ruff_status=$?
ty_out=$(uv run ty check 2>&1); ty_status=$?
test_out=$(uv run pytest -q 2>&1); test_status=$?

if [ "$ruff_status" -eq 0 ] && [ "$ty_status" -eq 0 ] && [ "$test_status" -eq 0 ]; then
  exit 0
fi

# Already retried once this turn: allow the stop.
if [ "$active" = "true" ]; then
  exit 0
fi

{
  echo "Stop gate failed — fix before finishing:"
  [ "$ruff_status" -ne 0 ] && { echo "--- ruff check ---"; echo "$ruff_out"; }
  [ "$ty_status" -ne 0 ] && { echo "--- ty check ---"; echo "$ty_out"; }
  [ "$test_status" -ne 0 ] && { echo "--- pytest ---"; echo "$test_out"; }
} >&2
exit 2
