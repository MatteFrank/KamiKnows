#!/bin/bash
# Run one KamiKnows command on a GPU node with a private Ollama server.
#
# Usage (from the repository root):
#   sbatch --job-name=KK-qa-qwen4b scripts/run_ollama.slurm \
#       scripts/generate.py --retrieval-dir outputs/<retrieval run> \
#       --output-dir outputs/<new qa run> --model qwen3:4b-instruct
#
# The script starts Ollama on a port derived from the job id, appends
# --base-url to the command, and at the end copies the SLURM and Ollama logs
# into the command's --output-dir. SBATCH options can be overridden on the
# sbatch command line (e.g. --time=02:00:00).
#
#SBATCH --job-name=KK
#SBATCH --account=kamiknows
#SBATCH --qos=kamiknows
#SBATCH --partition=bare-metal-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/e/eapicell/KamiKnows/outputs/_slurm_logs/%x-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PYTHON="${PYTHON:-/home/e/eapicell/.conda/envs/kamiknows/bin/python}"
KK_ROOT="${KK_ROOT:-/storage-hpc/kamiknows}"
OLLAMA_BIN="$KK_ROOT/software/ollama/bin/ollama"
export OLLAMA_MODELS="$KK_ROOT/models/ollama"
export OLLAMA_HOST="127.0.0.1:$((11400 + ${SLURM_JOB_ID:-0} % 500))"
LOG_DIR=outputs/_slurm_logs
OLLAMA_LOG="$LOG_DIR/ollama-${SLURM_JOB_ID:-local}.log"

if [ $# -eq 0 ]; then
  echo "usage: sbatch scripts/run_ollama.slurm <script.py> [args...]" >&2
  exit 2
fi
if [ ! -x "$OLLAMA_BIN" ] || [ ! -d "$OLLAMA_MODELS" ]; then
  echo "Ollama executable or model directory missing under $KK_ROOT" >&2
  exit 1
fi

OUTPUT_DIR=""
previous=""
for arg in "$@"; do
  [ "$previous" = "--output-dir" ] && OUTPUT_DIR="$arg"
  previous="$arg"
done

mkdir -p "$LOG_DIR"
"$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

finish() {
  kill "$OLLAMA_PID" 2>/dev/null || true
  if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    cp "$OLLAMA_LOG" "$OUTPUT_DIR/ollama.log" 2>/dev/null || true
    cp "$LOG_DIR/${SLURM_JOB_NAME:-KK}-${SLURM_JOB_ID:-local}.log" "$OUTPUT_DIR/slurm.log" 2>/dev/null || true
  fi
}
trap finish EXIT

for _ in $(seq 60); do
  curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null && break
  kill -0 "$OLLAMA_PID" 2>/dev/null || { echo "Ollama exited; see $OLLAMA_LOG" >&2; exit 1; }
  sleep 2
done
curl -sf "http://$OLLAMA_HOST/api/tags" > /dev/null || { echo "Ollama did not start" >&2; exit 1; }
echo "Ollama ready on $OLLAMA_HOST"

"$PYTHON" -u "$@" --base-url "http://$OLLAMA_HOST"
