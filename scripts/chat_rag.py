"""Chat RAG da terminale: retrieval locale, fonti esplicite e QA tramite Ollama."""

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kamiknows.rag.load_dataset import read_jsonl
from kamiknows.rag.outputs import write_json
from kamiknows.rag.qa_grounded_v1 import (
    annotate_registry_availability, build_default_model_registry,
    build_retrieved_qa_chunk, generate_model_answer_records,
)
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1
from kamiknows.rag.source_constraints import plan_source_constraints, select_balanced_results
from kamiknows.run_metadata import utc_now_iso


class ChatRetriever:
    """Carica l'indice una volta; MiniLM è opzionale e richiede cache locale reale."""

    def __init__(self, chunks, name):
        self.chunks = chunks
        self.name = name
        self.encoder = None
        if name == "tfidf":
            self.tfidf = TfidfRetrieverV1(chunks)
        else:
            from kamiknows.rag.embedding_registry import get_embedding_model_config
            from kamiknows.rag.embedding_selection import EmbeddingEncoder, build_embedding_text
            self.encoder = EmbeddingEncoder(get_embedding_model_config("all-MiniLM-L6-v2"),
                                            backend="sentence-transformers")
            self.encoder.model.to("cpu")
            self.vectors = [self.encode(build_embedding_text(chunk)) for chunk in chunks]

    def encode(self, text):
        # Stessa politica dei benchmark per i testi lunghi: finestre esatte di token,
        # media pesata e normalizzazione; nessun troncamento o fallback hash.
        from kamiknows.rag_ready.semantic_dataset import encode_token_windows
        model = self.encoder.model
        ids = model.tokenizer(text, add_special_tokens=False, truncation=False,
                              verbose=False)["input_ids"]
        if len(ids) + model.tokenizer.num_special_tokens_to_add(pair=False) > model.max_seq_length:
            return encode_token_windows(model, ids)[0]
        return self.encoder.encode([text], text_kind="passage")[0]

    def retrieve(self, question, top_k, source_policy, question_id):
        indices, ids, diagnostics = plan_source_constraints(self.chunks, question, top_k, source_policy)
        results = []
        if indices:
            count = len(indices) if ids else top_k
            if self.name == "tfidf":
                results = [build_retrieved_qa_chunk(chunk=r.chunk, rank=r.rank, score=r.score)
                           for r in self.tfidf.retrieve(question, top_k=count, candidate_indices=indices)]
            else:
                from kamiknows.rag.embedding_selection import retrieve_embedding_chunks
                results = retrieve_embedding_chunks(query_vector=self.encode(question),
                    chunk_vectors=[self.vectors[i] for i in indices],
                    chunks=[self.chunks[i] for i in indices], top_k=count)
            results = select_balanced_results(results, ids, top_k)
            # Dense retrieval salva preview: il prompt deve ricevere il testo completo.
            lookup = {c["chunk_id"]: c for c in self.chunks}
            results = [build_retrieved_qa_chunk(chunk=lookup[r["chunk_id"]],
                        rank=r["rank"], score=r["score"]) for r in results]
        return {"question_id": question_id, "question": question, "question_type": "interactive",
                "expected_behavior": "not_annotated", "top_k": top_k,
                "retriever": {"name": f"{self.name}_local_v1", "version": "chat_rag_v1"},
                "source_constraint": diagnostics, "retrieved_chunks": results,
                "retrieval_eval": {}}


def run_chat(*, retriever, model_config, args, session_dir, input_fn=input, output_fn=print):
    """Una domanda indipendente per turno: non accumula la cronologia nel prompt."""
    known_ids = [c["chunk_id"] for c in retriever.chunks]
    output_fn("Scrivi una domanda; /exit per uscire. Risposte da revisionare, non validate scientificamente.")
    turn = 0
    # Salva subito ogni contesto e ogni risposta completata, anche in caso di errori backend.
    with (session_dir / "contexts.jsonl").open("x", encoding="utf-8") as contexts, \
         (session_dir / "answers.jsonl").open("x", encoding="utf-8") as answers:
        while True:
            try:
                question = input_fn("KamiKnows> ").strip()
                if question.lower() in {"/exit", "/quit"}:
                    break
                if not question:
                    continue
                turn += 1
                context = retriever.retrieve(question, args.top_k, args.source_policy, f"chat_{turn:04d}")
                contexts.write(json.dumps(context, ensure_ascii=False) + "\n")
                contexts.flush()
                diag = context["source_constraint"]
                output_fn("Fonti richieste: " + (", ".join(diag["requested_arxiv_ids"]) or "nessuna; corpus completo"))
                output_fn("Stato retrieval: " + diag["status"])
                for c in context["retrieved_chunks"]:
                    text = c["text"] if args.show_context else " ".join(c["text"].split())[:240]
                    output_fn(f"{c['citation_label']} {c['arxiv_id']} / {c['section']} (score={c['score']:.4f})\n{text}")
                output_fn("Qwen sta generando la risposta...", flush=True)
                record = generate_model_answer_records(model_config=model_config,
                    retrieved_contexts=[context], known_chunk_ids=known_ids,
                    base_url=args.base_url, timeout_seconds=args.timeout_seconds)[0]
                record["created_at"] = utc_now_iso()
                record["source_constraint"] = diag
                answers.write(json.dumps(record, ensure_ascii=False) + "\n")
                answers.flush()
                output_fn("Risposta: " + record["answer"])
                output_fn(f"Stato: {record['answer_status']}; citazioni: {record['citation_validation']['status']}")
                if record["warnings"]:
                    output_fn("Warning: " + ", ".join(record["warnings"]))
                if record["backend_error"]:
                    output_fn("Errore backend: " + record["backend_error"]["message"])
            except (EOFError, KeyboardInterrupt):
                output_fn("\nSessione interrotta; i turni completati sono salvati.")
                break
    output_fn(f"Sessione salvata in: {session_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-ready-dir", type=Path, required=True)
    parser.add_argument("--retriever", choices=["tfidf", "minilm"], default="tfidf")
    parser.add_argument("--source-policy", choices=["none", "explicit_arxiv"], default="explicit_arxiv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11445")
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--show-context", action="store_true", help="Mostra i passaggi completi.")
    parser.add_argument("--session-dir", type=Path, help="Directory nuova; default outputs/chat_rag/<UUID>.")
    args = parser.parse_args(argv)
    try:
        if min(args.top_k, args.num_predict, args.timeout_seconds) < 1:
            raise ValueError("top-k, num-predict and timeout-seconds must be positive")
        session_dir = args.session_dir or Path("outputs/chat_rag") / uuid.uuid4().hex
        if session_dir.exists():
            raise FileExistsError(f"Session directory already exists: {session_dir}")
        chunk_path = args.rag_ready_dir / "chunks_v1.jsonl"
        chunks = read_jsonl(chunk_path)
        ids = [c.get("chunk_id") for c in chunks]
        if not chunks or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Dataset must have nonempty, unique chunk IDs")
        registry = annotate_registry_availability(registry=build_default_model_registry(
            model_names=[args.model], num_predict=args.num_predict), base_url=args.base_url)
        model = registry["models"][0]
        if model["availability_status"] != "available":
            raise RuntimeError(model.get("availability_notes", "Ollama/model unavailable"))
        print(f"Caricamento di {len(chunks)} chunk e indice {args.retriever}...", flush=True)
        retriever = ChatRetriever(chunks, args.retriever)
        session_dir.mkdir(parents=True, exist_ok=False)
        write_json(session_dir / "session_manifest.json", {
            "created_at": utc_now_iso(), "chunks_file": str(chunk_path.resolve()),
            "chunks_sha256": hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
            "chunks_count": len(chunks), "retriever": args.retriever,
            "source_policy": args.source_policy, "top_k": args.top_k, "model": model,
            "base_url": args.base_url, "timeout_seconds": args.timeout_seconds,
            "query_policy": "raw user question; no benchmark labels or section-keyword expansion",
            "conversation_policy": "independent questions; no prior turns in prompt"})
        run_chat(retriever=retriever, model_config=model, args=args, session_dir=session_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Chat RAG failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
