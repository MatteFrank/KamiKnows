import gzip
import io
import tarfile

import pytest

from kamiknows import parsing
from kamiknows.common import read_jsonl, read_manifest

LATEX = r"""\documentclass{article}
\begin{document}
\title{Ignored}
\section{Introduction}
We use a GAN \cite{goodfellow}. % a comment
\begin{equation} E = m c^2 \end{equation}
\subsection{Setup}
Cost is 5\% lower.
\section*{Acknowledgments}
Thanks.
\end{document}
"""


def test_parse_latex_sections_and_equations():
    sections, equations, warnings = parsing.parse_latex(LATEX)
    assert [(s["heading"], s["level"]) for s in sections] == [
        ("Introduction", 1), ("Setup", 2), ("Acknowledgments", 1)]
    assert "[REF]" in sections[0]["text"] and "comment" not in sections[0]["text"]
    assert "[EQUATION_001]" in sections[0]["text"]
    assert equations[0]["section_id"] == "sec_001" and "placeholder" not in equations[0]
    assert warnings == []


def test_flatten_inlines_inputs(tmp_path):
    (tmp_path / "main.tex").write_text("\\documentclass{article}\\input{body}")
    (tmp_path / "body.tex").write_text("BODY \\input{missing}")
    flat = parsing.flatten_latex_file(tmp_path / "main.tex", tmp_path)
    assert "BODY" in flat and "% unresolved input: missing" in flat


def test_unpack_tar_and_single_gzipped_tex(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = LATEX.encode()
        info = tarfile.TarInfo("paper/main.tex")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    root = parsing.unpack_source(buffer.getvalue(), tmp_path / "a")
    assert parsing.select_main_tex_file(root).name == "main.tex"

    (tmp_path / "b").mkdir()
    root = parsing.unpack_source(gzip.compress(LATEX.encode()), tmp_path / "b")
    assert (root / "source.tex").exists()

    (tmp_path / "c").mkdir()
    with pytest.raises(parsing.SourceError):
        parsing.unpack_source(b"%PDF-1.5 ...", tmp_path / "c")


def test_ids_and_paper_ids(tmp_path):
    ids = tmp_path / "ids.txt"
    ids.write_text("# corpus\narXiv:1705.02355v2\nhttps://arxiv.org/abs/2106.05285v3  # note\n1705.02355v2\n")
    assert parsing.read_id_list(ids) == ["1705.02355v2", "2106.05285v3"]
    assert parsing.paper_id_for("1705.02355v2") == "1705_02355v2"


def test_run_parsing_records_failures(tmp_path, monkeypatch):
    def fake_download(arxiv_id, source_dir, *, timeout_seconds):
        if arxiv_id == "2222.00002v1":
            raise parsing.SourceError("e-print is a PDF")
        source_dir.mkdir(parents=True)
        (source_dir / "main.tex").write_text(LATEX)
        return source_dir

    monkeypatch.setattr(parsing, "fetch_metadata", lambda arxiv_id, timeout_seconds: {"title": f"T {arxiv_id}"})
    monkeypatch.setattr(parsing, "download_source", fake_download)
    ids = tmp_path / "ids.txt"
    ids.write_text("1111.00001v1\n2222.00002v1\n")
    summary = parsing.run_parsing(ids_file=ids, output_dir=tmp_path / "run", timeout_seconds=5)
    assert summary["status_counts"] == {"success": 1, "failed": 1}
    papers = read_jsonl(tmp_path / "run" / "papers.jsonl")
    assert papers[1]["errors"] == ["SourceError: e-print is a PDF"]
    assert read_manifest(tmp_path / "run", "parse")["summary"]["sections"] == 3
    with pytest.raises(FileExistsError):
        parsing.run_parsing(ids_file=ids, output_dir=tmp_path / "run", timeout_seconds=5)
