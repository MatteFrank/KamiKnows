"""Tests for per-paper full-text output generation."""

from __future__ import annotations

import json
from pathlib import Path

from kamiknows.fulltext.arxiv_source import SourceDownloadResult
from kamiknows.fulltext.paper_outputs import write_paper_outputs_from_latex_source


def test_write_paper_outputs_from_tiny_latex_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Introduction}\n"
        "Fast calorimeter simulation uses a surrogate response model.\n"
        "\\begin{equation}\n"
        "E_{vis} = f(E_{true})\n"
        "\\end{equation}\n"
        "\\section{Conclusion}\n"
        "The method requires validation.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    paper_dir = tmp_path / "paper"
    metadata = {
        "arxiv_id": "1705.02355v2",
        "title": "Fast calorimeter simulation",
        "authors": ["Example Author"],
        "abstract": "A tiny test abstract.",
        "categories": ["hep-ex"],
        "url": "https://arxiv.org/abs/1705.02355v2",
    }
    source_download = SourceDownloadResult(
        arxiv_id="1705.02355v2",
        attempted_source_download=False,
        source_available=True,
        source_type="latex_source",
        downloaded_files=[],
        errors=[],
        source_root=source_root,
    )

    paper = write_paper_outputs_from_latex_source(
        metadata=metadata,
        source_root=source_root,
        paper_dir=paper_dir,
        source_download=source_download,
    )

    assert paper["parsing_status"] == "success"
    assert paper["sections_count"] == 2
    assert paper["equations_count"] == 1
    assert paper["chunks_count"] >= 1
    assert (paper_dir / "metadata.json").exists()
    assert (paper_dir / "source_download.json").exists()
    assert (paper_dir / "flat.tex").exists()
    assert (paper_dir / "plain_text.txt").exists()
    assert (paper_dir / "sections.json").exists()
    assert (paper_dir / "equations.json").exists()
    assert (paper_dir / "chunks.jsonl").exists()
    assert (paper_dir / "paper.json").exists()

    sections = json.loads((paper_dir / "sections.json").read_text(encoding="utf-8"))
    assert sections[0]["heading"] == "Introduction"
    assert "[EQUATION_001]" in (paper_dir / "plain_text.txt").read_text(encoding="utf-8")
