#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY RUN] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

declare -a RUN_DIRS=()
if [[ "$#" -gt 0 ]]; then
  RUN_DIRS=("$@")
else
  while IFS= read -r path; do
    RUN_DIRS+=("$path")
  done < <(find "${ROOT_DIR}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
fi

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "No run directories provided and none found under ${ROOT_DIR}/runs"
  exit 1
fi

for run_dir in "${RUN_DIRS[@]}"; do
  pred_csv="${run_dir}/oof_predictions.csv"
  out_png="${run_dir}/scatter_plot.png"
  if [[ ! -f "${pred_csv}" ]]; then
    echo "Skipping ${run_dir}: missing ${pred_csv}"
    continue
  fi
  echo "Plotting ${pred_csv}"
  run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/scripts/scatter_plot.py" \
    --pred-csv "${pred_csv}" \
    --target-col y_true \
    --pred-col y_pred \
    --out-png "${out_png}"
done

echo "Scatter plotting finished."
