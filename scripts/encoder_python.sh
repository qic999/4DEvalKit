#!/usr/bin/env bash
set -euo pipefail
task_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${FOURDEVAL_ENCODER_PYTHON:?Set FOURDEVAL_ENCODER_PYTHON to the model environment Python}" \
  "${task_script_dir}/encoder_entry.py" "$@"
