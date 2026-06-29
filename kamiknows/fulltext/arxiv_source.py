"""arXiv source download helpers for the Fase 1F full-text pilot.

The module prefers arXiv e-print source archives. PDF fallback is deliberately
not parsed in Fase 1F; when a downloaded source is a PDF, the code records an
explicit ``pdf_fallback_skipped`` error instead of adding heavy dependencies.
"""

from __future__ import annotations

import gzip
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from kamiknows.ingestion.arxiv_downloader import normalize_arxiv_id

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"

ERROR_CATEGORIES = (
    "metadata_error",
    "source_download_error",
    "source_unavailable",
    "archive_extract_error",
    "latex_main_file_not_found",
    "latex_flatten_error",
    "section_extraction_weak",
    "equation_extraction_weak",
    "plain_text_empty",
    "chunk_generation_empty",
    "pdf_fallback_skipped",
    "pdf_parse_error",
    "unknown_error",
)


@dataclass(slots=True)
class SourceDownloadResult:
    """Result of one arXiv source/e-print download attempt."""

    arxiv_id: str
    attempted_source_download: bool = False
    source_available: bool = False
    source_type: str = "unavailable"
    downloaded_files: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    source_root: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "attempted_source_download": self.attempted_source_download,
            "source_available": self.source_available,
            "source_type": self.source_type,
            "downloaded_files": self.downloaded_files,
            "errors": list(self.errors),
        }


def error_entry(category: str, message: str) -> dict[str, str]:
    """Build one normalized parsing/download error entry."""
    if category not in ERROR_CATEGORIES:
        category = "unknown_error"
    return {"category": category, "message": message}


def safe_paper_id(arxiv_id: str) -> str:
    """Return a stable filesystem-safe paper identifier from an arXiv ID."""
    normalized = normalize_arxiv_id(arxiv_id).lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _is_probably_pdf(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF")


def _decode_text(data: bytes) -> str | None:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _looks_like_latex(text: str) -> bool:
    return "\\documentclass" in text or "\\begin{document}" in text


def _add_file(result: SourceDownloadResult, path: Path) -> None:
    result.downloaded_files.append(str(path))


def _safe_extract_tar(archive_path: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    destination_resolved = destination.resolve()

    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            target = destination / member.name
            try:
                target.resolve().relative_to(destination_resolved)
            except ValueError as exc:
                raise RuntimeError(f"unsafe archive path: {member.name}") from exc

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as output:
                output.write(source.read())
            extracted.append(target)

    return extracted


def _write_single_latex_source(data: bytes, destination: Path) -> Path | None:
    text = _decode_text(data)
    if text is None or not _looks_like_latex(text):
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _extract_downloaded_source(download_path: Path, source_dir: Path) -> SourceDownloadResult:
    arxiv_id = source_dir.parent.name
    result = SourceDownloadResult(
        arxiv_id=arxiv_id,
        attempted_source_download=True,
        source_root=source_dir,
    )
    _add_file(result, download_path)

    extracted_dir = source_dir / "extracted"
    try:
        extracted = _safe_extract_tar(download_path, extracted_dir)
    except (tarfile.TarError, RuntimeError, OSError):
        extracted = []

    tex_files = [path for path in extracted if path.suffix.lower() == ".tex"]
    if tex_files:
        result.source_available = True
        result.source_type = "latex_source"
        result.source_root = extracted_dir
        result.downloaded_files.extend(str(path) for path in tex_files)
        return result

    raw_data = download_path.read_bytes()
    if _is_probably_pdf(raw_data):
        result.source_type = "unavailable"
        result.errors.append(
            error_entry(
                "pdf_fallback_skipped",
                "arXiv e-print response appears to be a PDF; PDF parsing is skipped in Fase 1F.",
            )
        )
        return result

    try:
        decompressed = gzip.decompress(raw_data)
    except OSError:
        decompressed = b""

    if decompressed:
        if _is_probably_pdf(decompressed):
            pdf_path = source_dir / "source.pdf"
            pdf_path.write_bytes(decompressed)
            _add_file(result, pdf_path)
            result.source_type = "unavailable"
            result.errors.append(
                error_entry(
                    "pdf_fallback_skipped",
                    "Compressed arXiv source is a PDF; PDF parsing is skipped in Fase 1F.",
                )
            )
            return result
        tex_path = _write_single_latex_source(decompressed, source_dir / "source.tex")
        if tex_path is not None:
            _add_file(result, tex_path)
            result.source_available = True
            result.source_type = "latex_source"
            result.source_root = source_dir
            return result

    tex_path = _write_single_latex_source(raw_data, source_dir / "source.tex")
    if tex_path is not None:
        _add_file(result, tex_path)
        result.source_available = True
        result.source_type = "latex_source"
        result.source_root = source_dir
        return result

    text = _decode_text(raw_data) or ""
    if "<html" in text.lower() or "not found" in text.lower():
        result.source_type = "unavailable"
        result.errors.append(error_entry("source_unavailable", "arXiv source is unavailable."))
    else:
        result.source_type = "error"
        result.errors.append(
            error_entry("archive_extract_error", "Downloaded source was not a readable LaTeX archive.")
        )
    return result


def download_arxiv_source(
    arxiv_id: str,
    source_dir: Path,
    *,
    timeout_seconds: int = 60,
) -> SourceDownloadResult:
    """Download and unpack arXiv source when available."""
    normalized_id = normalize_arxiv_id(arxiv_id)
    source_dir.mkdir(parents=True, exist_ok=True)
    result = SourceDownloadResult(
        arxiv_id=normalized_id,
        attempted_source_download=True,
        source_root=source_dir,
    )

    url = ARXIV_EPRINT_URL.format(arxiv_id=normalized_id)
    download_path = source_dir / "source_download.bin"

    try:
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "KamiKnows-Fase1F/0.1"},
        )
    except requests.RequestException as exc:
        result.source_type = "error"
        result.errors.append(error_entry("source_download_error", str(exc)))
        return result

    if response.status_code == 404:
        result.source_type = "unavailable"
        result.errors.append(error_entry("source_unavailable", f"arXiv returned HTTP {response.status_code}."))
        return result
    if response.status_code >= 400:
        result.source_type = "error"
        result.errors.append(error_entry("source_download_error", f"arXiv returned HTTP {response.status_code}."))
        return result

    try:
        download_path.write_bytes(response.content)
    except OSError as exc:
        result.source_type = "error"
        result.errors.append(error_entry("source_download_error", f"could not save source: {exc}"))
        return result

    extracted = _extract_downloaded_source(download_path, source_dir)
    extracted.arxiv_id = normalized_id
    return extracted
