#!/usr/bin/env bash
#SBATCH --job-name=zero01
#SBATCH --account=p_peptide
#SBATCH --mail-user=py61jagu@uni-leipzig.de
#SBATCH --mail-type=END,FAIL
#SBATCH --time=02:00:00
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=/data/cat/ws/rasi995g-PET/log/slurm_%x_%A_%a.out
#SBATCH --error=/data/cat/ws/rasi995g-PET/log/slurm_%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"

if command -v module >/dev/null 2>&1; then
  module load Anaconda3
  source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
  conda activate /data/cat/ws/rasi995g-PET/conda
fi
PYTHON_BIN="${PYTHON_BIN:-/data/cat/ws/rasi995g-PET/conda/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_FALLBACK:-python}"
fi
export TORCH_HOME="${TORCH_HOME:-$SLURM_TMPDIR/torch}"

INPUT_CSV="${INPUT_CSV:-${ROOT_DIR}/data/zero_shot_mutants.csv}"
SEQ_COL="${SEQ_COL:-Sequence}"
OUTDIR="${OUTDIR:-${ROOT_DIR}/zeroshot_embeds}"

BATCH_SIZE_ESM2="${BATCH_SIZE_ESM2:-8}"
BATCH_SIZE_PROSST="${BATCH_SIZE_PROSST:-8}"
BATCH_SIZE_T5="${BATCH_SIZE_T5:-4}"
MAX_TOKENS_ESM1V="${MAX_TOKENS_ESM1V:-12000}"

mkdir -p "${OUTDIR}"

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY RUN] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    echo "Running ESM2 embeddings..."
    run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/scripts/embeds_scripts/extract_esm2_embeddings.py" \
      --input-csv "${INPUT_CSV}" \
      --seq-col "${SEQ_COL}" \
      --output-npz "${OUTDIR}/esm2.npz" \
      --model-size 650M \
      --batch-size "${BATCH_SIZE_ESM2}"
    ;;
  1)
    echo "Running ProSST embeddings..."
    run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/scripts/embeds_scripts/extract_prosst_embeddings.py" \
      --input-csv "${INPUT_CSV}" \
      --seq-col "${SEQ_COL}" \
      --output-npz "${OUTDIR}/prosst.npz" \
      --model-id AI4Protein/ProSST-2048 \
      --batch-size "${BATCH_SIZE_PROSST}"
    ;;
  2)
    echo "Running ProtT5 embeddings..."
    run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/scripts/embeds_scripts/extract_prot_t5_embeddings.py" \
      --input-csv "${INPUT_CSV}" \
      --seq-col "${SEQ_COL}" \
      --output-npz "${OUTDIR}/protT5.npz" \
      --model-id Rostlab/prot_t5_xl_uniref50 \
      --batch-size "${BATCH_SIZE_T5}"
    ;;
  3)
    echo "Running ProtT5 half-encoder embeddings..."
    run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/scripts/embeds_scripts/extract_prot_t5_embeddings.py" \
      --input-csv "${INPUT_CSV}" \
      --seq-col "${SEQ_COL}" \
      --output-npz "${OUTDIR}/protT5_half.npz" \
      --model-id Rostlab/prot_t5_xl_half_uniref50-enc \
      --batch-size "${BATCH_SIZE_T5}"
    ;;
  4)
    echo "Running ProstT5 embeddings..."
    run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/scripts/embeds_scripts/extract_prot_t5_embeddings.py" \
      --input-csv "${INPUT_CSV}" \
      --seq-col "${SEQ_COL}" \
      --output-npz "${OUTDIR}/prostT5.npz" \
      --model-id Rostlab/ProstT5 \
      --batch-size "${BATCH_SIZE_T5}"
    ;;
  5)
    echo "Running ESM1v ensemble embeddings..."
    run_cmd "${PYTHON_BIN}" "${ROOT_DIR}/scripts/embeds_scripts/extract_esm1v_embeddings.py" \
      --input-csv "${INPUT_CSV}" \
      --seq-col "${SEQ_COL}" \
      --output-npz "${OUTDIR}/esm1v.npz" \
      --model-names esm1v_t33_650M_UR90S_1,esm1v_t33_650M_UR90S_2,esm1v_t33_650M_UR90S_3,esm1v_t33_650M_UR90S_4,esm1v_t33_650M_UR90S_5 \
      --max-tokens "${MAX_TOKENS_ESM1V}"
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Done. Outputs in ${OUTDIR}"
