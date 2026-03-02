#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_gen_server_2000.sh [conda_env_name] [--dry-run]
# Example:
#   bash run_gen_server_2000.sh concdvae310

CONDA_ENV_NAME="concdvae310"
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    CONDA_ENV_NAME="$arg"
  fi
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_PATH="$ROOT_DIR/conf/gen/server_2000.yaml"
INPUT_CSV="$ROOT_DIR/src/model/mp20_format/general_less.csv"
MODEL_CKPT="$ROOT_DIR/src/model/mp20_format/epoch=330-step=17543.ckpt"
PRIOR_CKPT="$ROOT_DIR/src/model/mp20_format/prior_default-epoch=95-step=10176.ckpt"

cd "$ROOT_DIR"

# Rebuild .env for the current server path
cp .env_bak .env
bash writeenv.sh

# Preflight checks
for p in "$CFG_PATH" "$INPUT_CSV" "$MODEL_CKPT" "$PRIOR_CKPT"; do
  if [[ ! -f "$p" ]]; then
    echo "Missing required file: $p" >&2
    exit 1
  fi
done

# Activate conda if available
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"
fi

echo "Preflight: checking python dependencies..."
python - <<'PY'
import torch, pandas, hydra  # noqa: F401
print(f"Python OK | torch={torch.__version__} | cuda={torch.cuda.is_available()}")
PY

labels_count=$(( $(wc -l < "$INPUT_CSV") - 1 ))
batch_size=$(awk -F: '/^batch_size:/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$CFG_PATH")
num_batches=$(awk -F: '/^num_batches_to_samples:/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$CFG_PATH")
num_samples_per_z=$(awk -F: '/^num_samples_per_z:/{gsub(/[[:space:]]/,"",$2); print $2; exit}' "$CFG_PATH")
expected_total=$(( labels_count * batch_size * num_batches * num_samples_per_z ))
echo "Preflight: labels=$labels_count batch_size=$batch_size num_batches_to_samples=$num_batches num_samples_per_z=$num_samples_per_z"
echo "Preflight: expected total structures=$expected_total"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run completed successfully. Generation not started."
  exit 0
fi

python scripts/gen_crystal.py --config conf/gen/server_2000.yaml

echo "Generation finished. Expected total: 2000 structures (5 labels x 400 each)."
