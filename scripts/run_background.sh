#!/usr/bin/env bash
# Detach a toolkit command, preserving the user's Python environment.
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: bash scripts/run_background.sh LOG_FILE COMMAND [ARGS...]" >&2
  exit 2
fi
task_log=$1
shift
mkdir -p "$(dirname "$task_log")"
if [[ -e "$task_log" || -e "$task_log.pid" ]]; then
  echo "Log or PID file exists; choose a new log path." >&2
  exit 2
fi
nohup setsid "$@" >"$task_log" 2>&1 < /dev/null &
task_pid=$!
echo "$task_pid" > "$task_log.pid"
echo "Started PID $task_pid; log: $task_log; PID file: $task_log.pid"
