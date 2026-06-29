"""Small LaTeX parsing helpers for the Fase 1F full-text pilot."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kamiknows.fulltext.arxiv_source import error_entry

SECTION_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^{}]+)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\{([^{}]+)\}")
EQUATION_RE = re.compile(
    r"\\begin\{(?P<env>equation\*?|align\*?|gather\*?)\}(?P<env_body>.*?)\\end\{(?P=env)\}"
    r"|\\\[(?P<bracket_body>.*?)\\\]"
    r"|\$\$(?P<dollar_body>.*?)\$\$",
    re.S,
)
WORD_RE = re.compile(r"\b[\w'-]+\b")


@dataclass(slots=True)
class ParsedLatexDocument:
    """Parsed LaTeX content ready for per-paper artifacts."""

    plain_text: str
    sections: list[dict[str, Any]]
    equations: list[dict[str, Any]]
    warnings: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


def count_words(text: str) -> int:
    """Count simple word-like tokens."""
    return len(WORD_RE.findall(text or ""))


def select_main_tex_file(source_dir: Path) -> Path | None:
    """Select the likely main TeX file from a source tree."""
    tex_files = [path for path in Path(source_dir).rglob("*.tex") if path.is_file()]
    if not tex_files:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        value = path.stat().st_size
        lowered_name = path.stem.lower()
        if "\\documentclass" in text:
            value += 1_000_000
        if "\\begin{document}" in text:
            value += 500_000
        if lowered_name in {"main", "paper", "ms", "article"}:
            value += 25_000
        return value, len(text), str(path)

    return max(tex_files, key=score)


def _resolve_input_path(root_dir: Path, raw_name: str) -> Path | None:
    raw_name = raw_name.strip()
    candidates = [root_dir / raw_name]
    if not raw_name.endswith(".tex"):
        candidates.append(root_dir / f"{raw_name}.tex")
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def flatten_latex_file(main_tex: Path, root_dir: Path | None = None) -> str:
    """Flatten simple ``\\input{}`` and ``\\include{}`` references."""
    root = Path(root_dir) if root_dir is not None else main_tex.parent
    seen: set[Path] = set()

    def flatten(path: Path) -> str:
        resolved = path.resolve()
        if resolved in seen:
            return ""
        seen.add(resolved)
        text = path.read_text(encoding="utf-8", errors="ignore")

        def replace(match: re.Match[str]) -> str:
            include_path = _resolve_input_path(root, match.group(1))
            if include_path is None:
                return f"\n% unresolved input: {match.group(1)}\n"
            return "\n" + flatten(include_path) + "\n"

        return INPUT_RE.sub(replace, text)

    return flatten(main_tex)


def strip_latex_comments(text: str) -> str:
    """Strip simple LaTeX comments while preserving escaped percent signs."""
    stripped_lines: list[str] = []
    for line in text.splitlines():
        index = 0
        cut = len(line)
        while True:
            percent = line.find("%", index)
            if percent == -1:
                break
            if percent > 0 and line[percent - 1] == "\\":
                index = percent + 1
                continue
            cut = percent
            break
        stripped_lines.append(line[:cut])
    return "\n".join(stripped_lines)


def _extract_equation_body(match: re.Match[str]) -> str:
    return (
        match.group("env_body")
        or match.group("bracket_body")
        or match.group("dollar_body")
        or ""
    ).strip()


def _clean_latex_text(text: str) -> str:
    text = re.sub(r"\\begin\{document\}|\\end\{document\}", " ", text)
    text = re.sub(r"\\(title|author|date)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\maketitle", " ", text)
    text = re.sub(r"\\label\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:cite|ref|eqref)\{[^{}]*\}", " [REF] ", text)
    text = re.sub(r"\\(?:emph|textbf|textit|mathrm|mathbf)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    text = re.sub(r"\\begin\{[^{}]+\}|\\end\{[^{}]+\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("~", " ")
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _document_body(text: str) -> str:
    begin = text.find("\\begin{document}")
    if begin != -1:
        text = text[begin + len("\\begin{document}") :]
    end = text.find("\\end{document}")
    if end != -1:
        text = text[:end]
    return text


def extract_equations(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Replace display equations with stable placeholders and return raw equations."""
    equations: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        equation_id = f"eq_{len(equations) + 1:03d}"
        placeholder = f"[EQUATION_{len(equations) + 1:03d}]"
        raw_latex = match.group(0).strip()
        body = _extract_equation_body(match)
        start, end = match.span()
        context_before = _clean_latex_text(text[max(0, start - 300) : start])[-300:]
        context_after = _clean_latex_text(text[end : end + 300])[:300]
        equations.append(
            {
                "equation_id": equation_id,
                "placeholder": placeholder,
                "raw_latex": raw_latex if raw_latex else body,
                "section_id": None,
                "context_before": context_before,
                "context_after": context_after,
            }
        )
        return f" {placeholder} "

    replaced = EQUATION_RE.sub(replace, text)
    return replaced, equations


def _section_matches(text: str) -> list[re.Match[str]]:
    return list(SECTION_RE.finditer(text))


def extract_sections(text: str, *, source_type: str = "latex_source") -> list[dict[str, Any]]:
    """Extract section-like blocks from LaTeX text with equation placeholders."""
    body = _document_body(text)
    matches = _section_matches(body)
    sections: list[dict[str, Any]] = []

    if not matches:
        plain = _clean_latex_text(body)
        if not plain:
            return []
        return [
            {
                "section_id": "sec_001",
                "level": 1,
                "heading": "Full text",
                "text": plain,
                "word_count": count_words(plain),
                "source_type": source_type,
                "_raw_start": 0,
                "_raw_end": len(body),
            }
        ]

    for index, match in enumerate(matches, start=1):
        next_start = matches[index].start() if index < len(matches) else len(body)
        section_body = body[match.end() : next_start]
        heading = _clean_latex_text(match.group(2)) or f"Section {index}"
        plain = _clean_latex_text(section_body)
        sections.append(
            {
                "section_id": f"sec_{index:03d}",
                "level": SECTION_LEVELS[match.group(1)],
                "heading": heading,
                "text": plain,
                "word_count": count_words(plain),
                "source_type": source_type,
                "_raw_start": match.start(),
                "_raw_end": next_start,
            }
        )

    return sections


def _assign_equation_sections(
    equations: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
    for equation in equations:
        placeholder = equation["placeholder"]
        for section in sections:
            if placeholder in section.get("text", ""):
                equation["section_id"] = section["section_id"]
                break


def parse_latex_document(flat_tex: str, *, source_type: str = "latex_source") -> ParsedLatexDocument:
    """Parse flattened LaTeX into plain text, sections, and equations."""
    warnings: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    without_comments = strip_latex_comments(flat_tex)
    with_placeholders, equations = extract_equations(without_comments)
    sections = extract_sections(with_placeholders, source_type=source_type)

    if not sections:
        warnings.append(error_entry("section_extraction_weak", "No sections were extracted from LaTeX."))
    if not equations:
        warnings.append(error_entry("equation_extraction_weak", "No display equations were extracted from LaTeX."))

    _assign_equation_sections(equations, sections)

    public_sections = [
        {key: value for key, value in section.items() if not key.startswith("_")}
        for section in sections
    ]
    public_equations = [
        {key: value for key, value in equation.items() if key != "placeholder"}
        for equation in equations
    ]
    plain_text = "\n\n".join(
        f"{section['heading']}\n{section['text']}".strip()
        for section in public_sections
        if section.get("text")
    ).strip()
    if not plain_text:
        errors.append(error_entry("plain_text_empty", "LaTeX parsing produced empty plain text."))

    return ParsedLatexDocument(
        plain_text=plain_text,
        sections=public_sections,
        equations=public_equations,
        warnings=warnings,
        errors=errors,
    )
