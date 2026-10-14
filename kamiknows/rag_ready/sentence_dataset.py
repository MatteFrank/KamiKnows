"""Build a separate sentence dataset from the baseline's exact paper selection."""

import hashlib
import json
from pathlib import Path

from kamiknows.fulltext.sentence_chunking import generate_sentence_chunks, split_sentences
from kamiknows.fulltext.section_filter import filter_sections
from kamiknows.rag.load_dataset import load_rag_ready_dataset
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag_ready.chunk_metadata_v1 import build_chunks_v1
from kamiknows.run_metadata import utc_now_iso


def build_sentence_dataset(*, reference_dir: Path, output_dir: Path,
                           target_words: int = 400, overlap_words: int = 60,
                           chunk_generator=generate_sentence_chunks,
                           strategy: str = "sentence_groups_v1",
                           exclude_backmatter: bool = False) -> dict:
    """Read only the baseline papers; refuse any existing output directory."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    dataset = load_rag_ready_dataset(reference_dir)
    domains = {c.get("metadata", {}).get("domain") for c in dataset["chunks"]}
    domains.discard(None)
    if len(domains) != 1:
        raise ValueError("reference chunks must identify one domain")
    domain = next(iter(domains))
    chunks, sources, papers, exclusions = [], [], [], []
    for original in dataset["papers"]:
        paper = dict(original)
        path = Path(paper["paper_dir"]) / "sections.json"
        content = path.read_bytes()
        sections = json.loads(content)
        if exclude_backmatter:
            sections, audit = filter_sections(sections)
            exclusions.extend({"paper_id": paper["paper_id"], "arxiv_id": paper["arxiv_id"],
                               "source_file": str(path), **row} for row in audit)
        generated = chunk_generator(
            paper_id=paper["paper_id"], arxiv_id=paper["arxiv_id"], title=paper["title"],
            sections=sections, target_words=target_words, overlap_words=overlap_words)
        # Verify every normalized sentence is covered, in section order.
        expected = [(s.get("section_id"), sentence) for s in sections
                    for sentence in split_sentences(str(s.get("text") or ""))]
        covered = []
        last_section, last_end = object(), 0
        for chunk in generated:
            sid = chunk["section_id"]
            if sid != last_section:
                last_section, last_end = sid, 0
            span = chunk["sentence_span"]
            sentences = split_sentences(chunk["text"])
            covered.extend((sid, sentence) for sentence in sentences[max(0, last_end-span["start"]):])
            last_end = span["end_exclusive"]
            chunk["metadata"] = {"domain": domain, "source_file": str(path),
                                  "parsing_status": paper["parsing_status"],
                                  "chunking_strategy": strategy}
        if covered != expected:
            raise ValueError(f"sentence coverage failed for {path}")
        paper["chunks_count"] = len(generated)
        paper["files"] = {**paper.get("files", {}), "chunks": str(output_dir / "chunks.jsonl")}
        papers.append(paper)
        chunks.extend(generated)
        sources.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
    if not chunks:
        raise ValueError("no sentence chunks were generated")
    enriched, _ = build_chunks_v1(chunks=chunks, papers=papers, equations=dataset["equations"])
    for raw, chunk in zip(chunks, enriched):
        chunk["chunk_id"] = raw["chunk_id"]
        chunk["sentence_span"] = raw["sentence_span"]
        chunk["section_id"] = raw["section_id"]
        chunk["word_count"] = raw["word_count"]
        if "cut_reason" in raw:
            chunk["cut_reason"] = raw["cut_reason"]
        if raw["oversize_sentence"]:
            chunk["quality_flags"].append("oversize_sentence")
    sizes = sorted(len(c["text"].split()) for c in chunks)
    manifest = {
        "manifest_version": ("sentence_dataset_v1" if strategy == "sentence_groups_v1"
                             else "semantic_dataset_v1"), "created_at": utc_now_iso(),
        "reference_dir": str(reference_dir), "source_sections": sources,
        "papers_count": len(papers), "chunks_v1_count": len(chunks),
        "equations_count": len(dataset["equations"]),
        "eval_questions_count": len(dataset["eval_questions"]),
        "chunking": {"strategy": strategy, "target_words": target_words,
                     "overlap_words": overlap_words, "size_unit": "whitespace_words",
                     "overlap_policy": "whole-sentence suffix within budget",
                     "long_sentence_policy": "keep intact and flag", "section_boundaries": True},
        "lengths": {"min": sizes[0], "max": sizes[-1], "mean": sum(sizes)/len(sizes)},
        "validation": {"sentence_coverage": "PASS", "oversize_sentence_count":
                       sum(c["oversize_sentence"] for c in chunks)},
        "notes": ["Heuristic sentence splitting requires spot checks on scientific text.",
                  ("Acknowledgments/bibliography filtering enabled; inspect exclusion audit."
                   if exclude_backmatter else
                   "Bibliography, parser artifacts and partial extraction are retained."),
                  "Legacy eval questions are copied for compatibility; select v2 explicitly."]}
    if exclude_backmatter:
        manifest["section_filter"] = {
            "enabled": True, "policy": "exclude_acknowledgments_bibliography_v1",
            "audit_file": "section_exclusions_v1.jsonl",
            "excluded_sections": sum(r["action"] == "exclude_section" for r in exclusions),
            "removed_trailing_markers": sum(r["action"] == "remove_trailing_marker" for r in exclusions),
            "removed_words": sum(r["removed_words"] for r in exclusions),
            "coverage_scope": "retained sections after filtering",
            "limitation": "Unlabelled bibliography within scientific sections is not inferred."}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "chunks_v1.jsonl", enriched)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "papers.jsonl", papers)
    write_jsonl(output_dir / "equations.jsonl", dataset["equations"])
    write_jsonl(output_dir / "eval_questions_v0.jsonl", dataset["eval_questions"])
    write_json(output_dir / "rag_manifest_v1.json", manifest)
    if exclude_backmatter:
        write_jsonl(output_dir / "section_exclusions_v1.jsonl", exclusions)
    return manifest
