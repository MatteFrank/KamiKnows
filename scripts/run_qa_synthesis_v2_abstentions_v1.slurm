#!/bin/bash
#SBATCH --job-name=KK-synthesis-v2
#SBATCH --account=kamiknows
#SBATCH --qos=kamiknows
#SBATCH --partition=bare-metal-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/e/eapicell/KamiKnows/outputs/slurm-synthesis-v2-%j.log

set -euo pipefail
cd /home/e/eapicell/KamiKnows
RUN_DIR=outputs/qa_qwen4b_synthesis_v2_abstentions_v1
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
export OLLAMA_HOST=127.0.0.1:11451
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
        response=requests.get('http://127.0.0.1:11451/api/tags', timeout=2)
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
/home/e/eapicell/.conda/envs/kamiknows/bin/python -u scripts/replay_qa_prompt_profile_v1.py \
 --baseline-dir outputs/qa_grounded_qwen4b_minilm_eval_v3_paired_top5 \
 --output-dir "$RUN_DIR" --prompt-profile synthesis_v2 \
 --base-url http://127.0.0.1:11451 \
 --question-ids eval_v3_pair_01_implicit eval_v3_pair_06_explicit eval_v3_pair_07_explicit eval_v3_pair_07_implicit
