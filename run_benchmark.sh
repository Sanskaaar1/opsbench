#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec env/bin/python benchmark.py "$@"
fi

exec srun \
  --partition=dev \
  --gres=gpu:nvidia_h200_3g.71gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --ntasks=1 \
  env/bin/python benchmark.py "$@"
