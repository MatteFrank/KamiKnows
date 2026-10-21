"""Step 4 - generate: answer the retrieved questions with an Ollama model and validate citations.

Needs a running Ollama server (on SLURM use scripts/run_ollama.slurm).

Examples:
    python scripts/generate.py \\
        --retrieval-dir outputs/retrieval_semantic_fd100_nobm_dense_explicit_paired_top5 \\
        --output-dir outputs/qa_semantic_fd100_nobm_dense_explicit_qwen4b_paired_top5 \\
        --model qwen3:4b-instruct

    # same contexts, other prompt, only some questions
    python scripts/generate.py --retrieval-dir ... --output-dir ... --model qwen3:4b-instruct \\
        --prompt synthesis --question-ids pair_01_implicit pair_07_explicit
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.generation import PROMPTS, run_generation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-dir", type=Path, required=True, help="output of retrieve.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen3:4b-instruct")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="grounded",
                        help="grounded: strict abstention; synthesis: allows partial answers (default: grounded)")
    parser.add_argument("--question-ids", nargs="+", help="answer only these questions")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature; 0 makes answers reproducible (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling (default: 0.9)")
    parser.add_argument("--num-predict", type=int, default=1024, help="maximum answer tokens (default: 1024)")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="per answer (default: 300)")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434",
                        help="Ollama server; set by run_ollama.slurm (default: http://127.0.0.1:11434)")
    args = parser.parse_args(argv)
    try:
        summary = run_generation(retrieval_dir=args.retrieval_dir, output_dir=args.output_dir, model=args.model,
                                 prompt=args.prompt, question_ids=args.question_ids,
                                 temperature=args.temperature, top_p=args.top_p, num_predict=args.num_predict,
                                 timeout_seconds=args.timeout_seconds, base_url=args.base_url)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"generate failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
