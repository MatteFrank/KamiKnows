#!/bin/bash
#SBATCH --job-name=KK-np1024
#SBATCH --account=kamiknows
#SBATCH --qos=kamiknows
#SBATCH --partition=bare-metal-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/e/eapicell/KamiKnows/outputs/_slurm_logs/%x-%j.log

set -euo pipefail
cd /home/e/eapicell/KamiKnows
# Same setup as ..._genmeta (json format); only --num-predict changes, 512 -> 1024.
RUN_DIR=outputs/qa_semantic_fd100_nobm_minilm_explicit_qwen4b_evalv3p_top5_np1024
if [ -e "$RUN_DIR" ]; then
  echo "Output already exists: $RUN_DIR; choose a new name" >&2
  exit 1
fi
KK_ROOT="${KK_ROOT:-/storage-hpc/kamiknows}"
OLLAMA_BIN="$KK_ROOT/software/ollama/bin/ollama"
export PATH="$KK_ROOT/software/ollama/bin:$PATH"
export OLLAMA_MODELS="$KK_ROOT/models/ollama"
if [ ! -x "$OLLAMA_BIN" ] || [ ! -d "$OLLAMA_MODELS" ]; then
  echo "Shared Ollama executable or model directory unavailable under $KK_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
SLURM_LOG="outputs/_slurm_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"
export OLLAMA_HOST=127.0.0.1:11454
"$OLLAMA_BIN" serve > "$RUN_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
export QA_OLLAMA_PID="$OLLAMA_PID"
# On exit: stop Ollama, keep a copy of this script and of the SLURM log in the run directory.
trap 'kill "$OLLAMA_PID" 2>/dev/null || true; cp scripts/run_qa_qwen4b_evalv3p_np1024_v1.slurm "$RUN_DIR/run.slurm"; cp "$SLURM_LOG" "$RUN_DIR/" 2>/dev/null || true' EXIT
/home/e/eapicell/.conda/envs/kamiknows/bin/python - <<'PYREADY'
import os, time, requests
server_pid = int(os.environ["QA_OLLAMA_PID"])
for attempt in range(60):
    try:
        os.kill(server_pid, 0)
        response=requests.get('http://127.0.0.1:11454/api/tags', timeout=2)
        response.raise_for_status()
        names=[m['name'] for m in response.json().get('models', [])]
        if 'qwen3:4b-instruct' not in names:
            raise SystemExit('qwen3:4b-instruct is not available; aborting before benchmark')
        print('Ollama ready; qwen3:4b-instruct available', flush=True)
        break
    except requests.RequestException:
        time.sleep(2)
else:
    raise SystemExit('Ollama failed to start')
PYREADY
/home/e/eapicell/.conda/envs/kamiknows/bin/python -u scripts/run_qa_grounded_v1.py \
 --rag-ready-dir outputs/rag_ready_fastcalo_semantic_firstdrop100_no_backmatter_v1 \
 --retrieval-results-file outputs/retrieval_explicit_sources_semantic_eval_v3_paired_top5/minilm_retrieval_v1.jsonl \
 --eval-questions-file data/eval/rag_eval_set_v3_paired.jsonl \
 --output-dir "$RUN_DIR" \
 --report-path "docs/task_reports/$(basename "$RUN_DIR").md" \
 --source-policy explicit_arxiv \
 --top-k 5 --models qwen3:4b-instruct --temperature 0.0 --top-p 0.9 --num-predict 1024 \
 --timeout-seconds 300 --base-url http://127.0.0.1:11454
