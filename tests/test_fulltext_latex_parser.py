"""Tests for minimal LaTeX parsing helpers."""

from __future__ import annotations

from pathlib import Path

from kamiknows.fulltext.latex_parser import (
    extract_equations,
    extract_sections,
    flatten_latex_file,
    parse_latex_document,
    select_main_tex_file,
)


def test_select_main_tex_file_prefers_documentclass(tmp_path: Path) -> None:
    (tmp_path / "appendix.tex").write_text("appendix " * 500, encoding="utf-8")
    main = tmp_path / "main.tex"
    main.write_text(
        "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n",
        encoding="utf-8",
    )

    assert select_main_tex_file(tmp_path) == main


def test_flatten_latex_file_resolves_simple_input(tmp_path: Path) -> None:
    main = tmp_path / "main.tex"
    main.write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{section1}\n\\end{document}\n",
        encoding="utf-8",
    )
    (tmp_path / "section1.tex").write_text("\\section{Intro}\nBody text.", encoding="utf-8")

    flattened = flatten_latex_file(main, root_dir=tmp_path)

    assert "\\section{Intro}" in flattened
    assert "Body text." in flattened


def test_extract_sections_finds_simple_section_and_subsection() -> None:
    latex = (
        "\\begin{document}\n"
        "\\section{Introduction}\n"
        "This is the first section.\n"
        "\\subsection{Method}\n"
        "This is the method text.\n"
        "\\end{document}\n"
    )

    sections = extract_sections(latex)

    assert [section["heading"] for section in sections] == ["Introduction", "Method"]
    assert sections[0]["level"] == 1
    assert sections[1]["level"] == 2
    assert "first section" in sections[0]["text"]


def test_extract_equations_replaces_display_equations() -> None:
    latex = "Before \\begin{equation}E = mc^2\\end{equation} after."

    replaced, equations = extract_equations(latex)

    assert "[EQUATION_001]" in replaced
    assert equations[0]["equation_id"] == "eq_001"
    assert "E = mc^2" in equations[0]["raw_latex"]
    assert "Before" in equations[0]["context_before"]


def test_parse_latex_document_preserves_equation_placeholders() -> None:
    latex = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\section{Result}\n"
        "We use\n"
        "\\[ a^2 + b^2 = c^2 \\]\n"
        "in the detector response model.\n"
        "\\end{document}\n"
    )

    parsed = parse_latex_document(latex)

    assert parsed.sections[0]["heading"] == "Result"
    assert "[EQUATION_001]" in parsed.plain_text
    assert parsed.equations[0]["section_id"] == "sec_001"
    assert parsed.errors == []
