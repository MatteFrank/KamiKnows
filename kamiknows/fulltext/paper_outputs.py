"""Per-paper output generation for the Fase 1F full-text pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from kamiknows.fulltext.arxiv_source import (
    SourceDownloadResult,
    download_arxiv_source,
    error_entry,
    safe_paper_id,
)
from kamiknows.fulltext.chunking import count_words, generate_chunks
from kamiknows.fulltext.latex_parser import (
    flatten_latex_file,
    parse_latex_document,
    select_main_tex_file,
)
from kamiknows.ingestion.arxiv_downloader import fetch_arxiv_metadata_by_id, normalize_arxiv_id


def write_json(data: Any, path: Path) -> Path:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    """Write dictionaries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _metadata_output(
    metadata: dict[str, Any],
    *,
    processing_status: str,
    source_status: str,
) -> dict[str, Any]:
    return {
        "arxiv_id": metadata.get("arxiv_id", ""),
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "abstract": metadata.get("abstract", ""),
        "categories": metadata.get("categories", []),
        "source_url": metadata.get("url", ""),
        "source_status": source_status,
        "processing_status": processing_status,
    }


def _files_map(paper_dir: Path) -> dict[str, str | None]:
    files = {
        "metadata": paper_dir / "metadata.json",
        "source_download": paper_dir / "source_download.json",
        "flat_tex": paper_dir / "flat.tex",
        "plain_text": paper_dir / "plain_text.txt",
        "paper": paper_dir / "paper.json",
        "sections": paper_dir / "sections.json",
        "equations": paper_dir / "equations.json",
        "chunks": paper_dir / "chunks.jsonl",
    }
    return {
        name: str(path) if path.exists() else None
        for name, path in files.items()
    }


def write_paper_outputs_from_latex_source(
    *,
    metadata: dict[str, Any],
    source_root: Path,
    paper_dir: Path,
    source_download: SourceDownloadResult | None = None,
) -> dict[str, Any]:
    """Parse one local LaTeX source tree and write all per-paper outputs."""
    normalized_id = normalize_arxiv_id(str(metadata.get("arxiv_id", "")))
    paper_id = safe_paper_id(normalized_id)
    title = str(metadata.get("title") or normalized_id)
    paper_dir.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    sections: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    plain_text = ""
    source_type = "latex_source"

    if source_download is None:
        source_download = SourceDownloadResult(
            arxiv_id=normalized_id,
            attempted_source_download=False,
            source_available=True,
            source_type="latex_source",
            downloaded_files=[],
            errors=[],
            source_root=source_root,
        )

    main_tex = select_main_tex_file(source_root)
    if main_tex is None:
        errors.append(error_entry("latex_main_file_not_found", "No likely main .tex file was found."))
    else:
        try:
            flat_tex = flatten_latex_file(main_tex, root_dir=source_root)
            (paper_dir / "flat.tex").write_text(flat_tex, encoding="utf-8")
            parsed = parse_latex_document(flat_tex, source_type=source_type)
            errors.extend(parsed.errors)
            warnings.extend(parsed.warnings)
            sections = parsed.sections
            equations = parsed.equations
            plain_text = parsed.plain_text
            if plain_text:
                (paper_dir / "plain_text.txt").write_text(plain_text + "\n", encoding="utf-8")
            chunks = generate_chunks(
                paper_id=paper_id,
                arxiv_id=normalized_id,
                title=title,
                sections=sections,
            )
            if not chunks:
                errors.append(error_entry("chunk_generation_empty", "No chunks were generated."))
        except OSError as exc:
            errors.append(error_entry("latex_flatten_error", str(exc)))
        except Exception as exc:  # keep one bad paper from stopping the pilot
            errors.append(error_entry("unknown_error", str(exc)))

    if source_download.errors:
        errors.extend(source_download.errors)

    write_json(sections, paper_dir / "sections.json")
    write_json(equations, paper_dir / "equations.json")
    write_jsonl(chunks, paper_dir / "chunks.jsonl")

    if errors:
        parsing_status = "partial" if plain_text and chunks else "failed"
    elif warnings:
        parsing_status = "partial"
    else:
        parsing_status = "success"

    metadata_json = _metadata_output(
        metadata,
        processing_status=parsing_status,
        source_status=source_download.source_type,
    )
    write_json(metadata_json, paper_dir / "metadata.json")
    write_json(source_download.to_dict(), paper_dir / "source_download.json")

    paper_json = {
        "paper_id": paper_id,
        "arxiv_id": normalized_id,
        "title": title,
        "abstract": metadata.get("abstract", ""),
        "parsing_status": parsing_status,
        "source_type": source_download.source_type,
        "sections_count": len(sections),
        "equations_count": len(equations),
        "chunks_count": len(chunks),
        "plain_text_word_count": count_words(plain_text),
        "errors": errors,
        "warnings": warnings,
        "files": {},
    }
    write_json(paper_json, paper_dir / "paper.json")
    paper_json["files"] = _files_map(paper_dir)
    write_json(paper_json, paper_dir / "paper.json")
    return paper_json


def write_failed_paper_outputs(
    *,
    arxiv_id: str,
    paper_dir: Path,
    metadata: dict[str, Any] | None,
    source_download: SourceDownloadResult | None,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    """Write minimal failed outputs for a paper that cannot be parsed."""
    normalized_id = normalize_arxiv_id(arxiv_id)
    paper_id = safe_paper_id(normalized_id)
    metadata = metadata or {
        "arxiv_id": normalized_id,
        "title": "",
        "authors": [],
        "abstract": "",
        "categories": [],
        "url": "",
    }
    if source_download is None:
        source_download = SourceDownloadResult(
            arxiv_id=normalized_id,
            attempted_source_download=False,
            source_available=False,
            source_type="error",
        )
    paper_dir.mkdir(parents=True, exist_ok=True)
    write_json([], paper_dir / "sections.json")
    write_json([], paper_dir / "equations.json")
    write_jsonl([], paper_dir / "chunks.jsonl")
    write_json(
        _metadata_output(
            metadata,
            processing_status="failed",
            source_status=source_download.source_type,
        ),
        paper_dir / "metadata.json",
    )
    write_json(source_download.to_dict(), paper_dir / "source_download.json")
    paper_json = {
        "paper_id": paper_id,
        "arxiv_id": normalized_id,
        "title": metadata.get("title", ""),
        "abstract": metadata.get("abstract", ""),
        "parsing_status": "failed",
        "source_type": source_download.source_type,
        "sections_count": 0,
        "equations_count": 0,
        "chunks_count": 0,
        "plain_text_word_count": 0,
        "errors": errors + list(source_download.errors),
        "warnings": [],
        "files": {},
    }
    write_json(paper_json, paper_dir / "paper.json")
    paper_json["files"] = _files_map(paper_dir)
    write_json(paper_json, paper_dir / "paper.json")
    return paper_json


def process_arxiv_paper(
    *,
    arxiv_id: str,
    papers_dir: Path,
    timeout_seconds: int = 60,
    metadata_fetcher: Callable[..., dict[str, Any]] = fetch_arxiv_metadata_by_id,
    source_downloader: Callable[..., SourceDownloadResult] = download_arxiv_source,
) -> dict[str, Any]:
    """Fetch metadata/source for one arXiv ID and write per-paper artifacts."""
    normalized_id = normalize_arxiv_id(arxiv_id)
    paper_id = safe_paper_id(normalized_id)
    paper_dir = papers_dir / paper_id
    source_dir = paper_dir / "source"

    metadata: dict[str, Any] | None = None
    source_download: SourceDownloadResult | None = None
    errors: list[dict[str, str]] = []

    try:
        metadata = metadata_fetcher(normalized_id, timeout_seconds=timeout_seconds)
    except TypeError:
        metadata = metadata_fetcher(normalized_id)
    except Exception as exc:
        errors.append(error_entry("metadata_error", str(exc)))
        return write_failed_paper_outputs(
            arxiv_id=normalized_id,
            paper_dir=paper_dir,
            metadata=metadata,
            source_download=source_download,
            errors=errors,
        )

    try:
        source_download = source_downloader(
            normalized_id,
            source_dir,
            timeout_seconds=timeout_seconds,
        )
    except TypeError:
        source_download = source_downloader(normalized_id, source_dir)
    except Exception as exc:
        source_download = SourceDownloadResult(
            arxiv_id=normalized_id,
            attempted_source_download=True,
            source_available=False,
            source_type="error",
            source_root=source_dir,
            errors=[error_entry("source_download_error", str(exc))],
        )

    if not source_download.source_available or source_download.source_root is None:
        return write_failed_paper_outputs(
            arxiv_id=normalized_id,
            paper_dir=paper_dir,
            metadata=metadata,
            source_download=source_download,
            errors=errors,
        )

    return write_paper_outputs_from_latex_source(
        metadata=metadata,
        source_root=source_download.source_root,
        paper_dir=paper_dir,
        source_download=source_download,
    )
