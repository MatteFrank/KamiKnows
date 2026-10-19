#!/bin/bash
#SBATCH --job-name=KK-v3-qwen4b
#SBATCH --account=kamiknows
#SBATCH --qos=kamiknows
#SBATCH --partition=bare-metal-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/e/eapicell/KamiKnows/outputs/slurm-qwen4b-v3-paired-%j.log

set -euo pipefail
cd /home/e/eapicell/KamiKnows
RUN_DIR=outputs/qa_grounded_qwen4b_minilm_eval_v3_paired_top5
# Ogni esecuzione richiede una directory nuova per preservare i risultati.
if [ -e "$RUN_DIR" ]; then
  echo "Output already exists: $RUN_DIR; choose a new RUN_DIR" >&2
  exit 1
fi
# Installazione nativa condivisa: nessun container e nessun download di modelli.
KK_ROOT="${KK_ROOT:-/storage-hpc/kamiknows}"
OLLAMA_BIN="$KK_ROOT/software/ollama/bin/ollama"
export PATH="$KK_ROOT/software/ollama/bin:$PATH"
export OLLAMA_MODELS="$KK_ROOT/models/ollama"
if [ ! -x "$OLLAMA_BIN" ] || [ ! -d "$OLLAMA_MODELS" ]; then
  echo "Shared Ollama executable or model directory unavailable under $KK_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
export OLLAMA_HOST=127.0.0.1:11450
"$OLLAMA_BIN" serve > "$RUN_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!
export QA_OLLAMA_PID="$OLLAMA_PID"
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT
/home/e/eapicell/.conda/envs/kamiknows/bin/python - <<'PYREADY'
import os, time, requests
server_pid = int(os.environ["QA_OLLAMA_PID"])
for attempt in range(60):
    try:
        os.kill(server_pid, 0)
        response=requests.get('http://127.0.0.1:11450/api/tags', timeout=2)
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
 --report-path docs/task_reports/qa_grounded_qwen4b_minilm_eval_v3_paired_top5.md \
 --source-policy explicit_arxiv \
 --top-k 5 --models qwen3:4b-instruct --temperature 0.0 --top-p 0.9 --num-predict 512 \
 --timeout-seconds 300 --base-url http://127.0.0.1:11450
