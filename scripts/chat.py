"""Interactive chat: retrieval + grounded generation for questions typed in the terminal.

Each question is answered independently (previous turns are not in the prompt).
Every turn is saved in outputs/chat/<session>/turns.jsonl.

Example:
    python scripts/chat.py --dataset-dir outputs/chunks_fastcalo_semantic_fd100_nobm \\
        --retriever dense --model qwen3:4b-instruct --base-url http://127.0.0.1:11434
"""

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kamiknows.common import create_output_dir, open_jsonl, read_jsonl, read_manifest, utc_now, write_manifest
from kamiknows.embeddings import EMBEDDING_MODELS
from kamiknows.generation import PROMPTS, answer_question, check_model
from kamiknows.retrieval import build_index, retrieve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="output of chunk.py")
    parser.add_argument("--retriever", choices=["tfidf", "dense"], default="dense",
                        help="tfidf: lexical; dense: embeddings (default: dense)")
    parser.add_argument("--embedding-model", choices=sorted(EMBEDDING_MODELS), default="all-MiniLM-L6-v2",
                        help="used by --retriever dense (default: all-MiniLM-L6-v2)")
    parser.add_argument("--device", default="cpu", help="torch device for the encoder (default: cpu)")
    parser.add_argument("--top-k", type=int, default=5, help="passages per question (default: 5)")
    parser.add_argument("--source-policy", choices=["none", "explicit_arxiv"], default="explicit_arxiv",
                        help="explicit_arxiv: restrict to the arXiv ids written in the question (default)")
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen3:4b-instruct")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="grounded",
                        help="grounded: strict abstention; synthesis: allows partial answers (default: grounded)")
    parser.add_argument("--num-predict", type=int, default=1024, help="maximum answer tokens (default: 1024)")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="per answer (default: 300)")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434",
                        help="Ollama server (default: http://127.0.0.1:11434)")
    parser.add_argument("--show-passages", action="store_true", help="print full passage text")
    parser.add_argument("--session-dir", type=Path, help="default: outputs/chat/<random id>")
    args = parser.parse_args(argv)
    try:
        read_manifest(args.dataset_dir, "chunk")
        check_model(args.model, args.base_url)
        chunks = read_jsonl(args.dataset_dir / "chunks.jsonl")
        print(f"Indexing {len(chunks)} chunks with {args.retriever}...", flush=True)
        index = build_index(args.retriever, chunks, embedding_model=args.embedding_model, device=args.device)
        session_dir = create_output_dir(args.session_dir or Path("outputs/chat") / uuid.uuid4().hex)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"chat failed: {exc}", file=sys.stderr)
        return 1

    options = {"temperature": 0.0, "top_p": 0.9, "num_predict": args.num_predict}
    turns = 0
    print("Type a question, or /exit. Answers are not scientifically validated.")
    with open_jsonl(session_dir / "turns.jsonl") as write:
        while True:
            try:
                question = input("KamiKnows> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question in {"/exit", "/quit"}:
                break
            if not question:
                continue
            turns += 1
            passages, constraint = retrieve(index, chunks, question, top_k=args.top_k,
                                            source_policy=args.source_policy)
            context = {"question_id": f"chat_{turns:04d}", "question": question,
                       "source_constraint": constraint, "passages": passages}
            print(f"Sources requested: {', '.join(constraint['requested_arxiv_ids']) or 'none'}"
                  f" ({constraint['status']})")
            for p in passages:
                text = p["text"] if args.show_passages else p["text"][:240]
                print(f"[{p['label']}] {p['arxiv_id']} / {p['section_heading']} (score {p['score']:.3f})\n{text}")
            print("Generating...", flush=True)
            answer = answer_question(context, model=args.model, prompt=args.prompt, options=options,
                                     base_url=args.base_url, timeout_seconds=args.timeout_seconds)
            write({"created_at": utc_now(), **context, "answer": answer})
            if answer["answer_status"] == "backend_error":
                print(f"Backend error: {answer['backend_error']}")
                continue
            print(f"Answer: {answer['answer']}")
            cited = [p for p in passages if p["label"] in answer["citation_validation"]["valid_labels"]]
            if cited:
                print("Sources: " + "; ".join(f"[{p['label']}] {p['arxiv_id']} / {p['section_heading']}" for p in cited))
            print(f"Status: {answer['answer_status']}; citations: {answer['citation_validation']['status']}"
                  + (f"; warnings: {', '.join(answer['warnings'])}" if answer["warnings"] else ""))

    params = {key: value for key, value in vars(args).items() if key not in {"show_passages", "session_dir"}}
    write_manifest(session_dir, step="chat", params=params,
                   inputs={"chunks": args.dataset_dir / "chunks.jsonl"}, summary={"turns": turns})
    print(f"Session saved to {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
