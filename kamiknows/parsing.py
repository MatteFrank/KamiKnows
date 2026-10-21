"""Step 1: download arXiv LaTeX sources and parse them into sections and equations.

Output layout of a parse run::

    papers.jsonl                      one record per requested paper (status, counts, errors)
    papers/<paper_id>/paper.json      same record
    papers/<paper_id>/sections.json   [{section_id, level, heading, text}]
    papers/<paper_id>/equations.json  [{equation_id, section_id, raw_latex, context_before, context_after}]
    papers/<paper_id>/flat.tex        main .tex with \\input/\\include inlined
    papers/<paper_id>/source/         downloaded e-print and extracted files

Only LaTeX sources are supported: PDF-only papers are recorded as failed.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from kamiknows.common import create_output_dir, write_json, write_jsonl, write_manifest

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "KamiKnows/1.0"

SECTION_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^{}]+)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\{([^{}]+)\}")
EQUATION_RE = re.compile(
    r"\\begin\{(?P<env>equation\*?|align\*?|gather\*?)\}(?P<env_body>.*?)\\end\{(?P=env)\}"
    r"|\\\[(?P<bracket_body>.*?)\\\]"
    r"|\$\$(?P<dollar_body>.*?)\$\$",
    re.S,
)


class SourceError(RuntimeError):
    """The arXiv e-print is missing or is not a usable LaTeX source."""


# arXiv identifiers and metadata

def normalize_arxiv_id(value: str) -> str:
    """Accept '2301.00001v2', 'arXiv:2301.00001v2' or an abs/pdf URL."""
    cleaned = value.strip().removeprefix("arXiv:").removeprefix("arxiv:")
    for marker in ("/abs/", "/pdf/"):
        if marker in cleaned:
            cleaned = cleaned.rsplit(marker, maxsplit=1)[-1]
    cleaned = cleaned.removesuffix(".pdf").strip("/")
    if not cleaned:
        raise ValueError(f"empty arXiv id: {value!r}")
    return cleaned


def paper_id_for(arxiv_id: str) -> str:
    """Filesystem-safe paper id, e.g. '1705.02355v2' -> '1705_02355v2'."""
    return re.sub(r"[^a-z0-9]+", "_", normalize_arxiv_id(arxiv_id).lower()).strip("_")


def read_id_list(path: Path) -> list[str]:
    """One arXiv id per line; '#' starts a comment; duplicates are dropped."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(normalize_arxiv_id(line))
    if not ids:
        raise ValueError(f"no arXiv ids in {path}")
    return list(dict.fromkeys(ids))


def fetch_metadata(arxiv_id: str, *, timeout_seconds: int) -> dict[str, Any]:
    response = requests.get(
        ARXIV_API_URL, params={"id_list": arxiv_id, "max_results": 1}, timeout=timeout_seconds
    )
    response.raise_for_status()
    entry = ET.fromstring(response.text).find("atom:entry", ATOM)
    if entry is None:
        raise SourceError(f"arXiv API returned no entry for {arxiv_id}")

    def text(tag: str) -> str:
        return " ".join((entry.findtext(f"atom:{tag}", default="", namespaces=ATOM)).split())

    title = text("title")
    if not title:
        raise SourceError(f"arXiv entry for {arxiv_id} has no title")
    return {
        "title": title,
        "abstract": text("summary"),
        "authors": [
            " ".join(author.findtext("atom:name", default="", namespaces=ATOM).split())
            for author in entry.findall("atom:author", ATOM)
        ],
        "categories": [c.attrib.get("term", "") for c in entry.findall("atom:category", ATOM)],
        "published": text("published"),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


# Source download

def download_source(arxiv_id: str, source_dir: Path, *, timeout_seconds: int) -> Path:
    """Download the e-print and return the directory that holds the .tex files."""
    response = requests.get(
        ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id),
        timeout=timeout_seconds,
        headers={"User-Agent": USER_AGENT},
    )
    if response.status_code >= 400:
        raise SourceError(f"arXiv e-print returned HTTP {response.status_code}")
    source_dir.mkdir(parents=True)
    (source_dir / "eprint.bin").write_bytes(response.content)
    return unpack_source(response.content, source_dir)


def unpack_source(data: bytes, source_dir: Path) -> Path:
    """An e-print is a (gzipped) tar archive, a gzipped single .tex, or a PDF."""
    extracted = source_dir / "extracted"
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            _safe_extract(archive, extracted)
        if any(extracted.rglob("*.tex")):
            return extracted
    except tarfile.TarError:
        pass
    try:
        data = gzip.decompress(data)
    except OSError:
        pass
    if data.lstrip().startswith(b"%PDF"):
        raise SourceError("e-print is a PDF; only LaTeX sources are supported")
    for encoding in ("utf-8", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if "\\documentclass" in text or "\\begin{document}" in text:
        (source_dir / "source.tex").write_text(text, encoding="utf-8")
        return source_dir
    raise SourceError("e-print is not a readable LaTeX source")


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract regular files only, refusing paths that escape the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.getmembers():
        target = destination / member.name
        if not target.resolve().is_relative_to(root):
            raise SourceError(f"unsafe path in archive: {member.name}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            source = archive.extractfile(member)
            if source is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())


# LaTeX parsing

def select_main_tex_file(source_dir: Path) -> Path | None:
    """Prefer files with \\documentclass and \\begin{document}, then common names, then size."""
    def score(path: Path) -> tuple[int, int, str]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        value = path.stat().st_size
        if "\\documentclass" in text:
            value += 1_000_000
        if "\\begin{document}" in text:
            value += 500_000
        if path.stem.lower() in {"main", "paper", "ms", "article"}:
            value += 25_000
        return value, len(text), str(path)

    tex_files = [path for path in Path(source_dir).rglob("*.tex") if path.is_file()]
    return max(tex_files, key=score) if tex_files else None


def flatten_latex_file(main_tex: Path, root_dir: Path) -> str:
    """Inline \\input{} and \\include{} recursively; each file is inlined once."""
    seen: set[Path] = set()

    def resolve(name: str) -> Path | None:
        name = name.strip()
        for candidate in (root_dir / name, root_dir / f"{name}.tex"):
            if candidate.is_file():
                return candidate
        return None

    def flatten(path: Path) -> str:
        if path.resolve() in seen:
            return ""
        seen.add(path.resolve())

        def replace(match: re.Match[str]) -> str:
            included = resolve(match.group(1))
            if included is None:
                return f"\n% unresolved input: {match.group(1)}\n"
            return "\n" + flatten(included) + "\n"

        return INPUT_RE.sub(replace, path.read_text(encoding="utf-8", errors="ignore"))

    return flatten(main_tex)


def strip_latex_comments(text: str) -> str:
    """Drop everything after an unescaped '%' on each line."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def clean_latex_text(text: str) -> str:
    """Reduce LaTeX markup to plain text; citations/references become [REF]."""
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
    return re.sub(r"\s+", " ", text).strip()


def extract_equations(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Replace display equations with [EQUATION_nnn] placeholders."""
    equations: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        number = len(equations) + 1
        body = match.group("env_body") or match.group("bracket_body") or match.group("dollar_body") or ""
        start, end = match.span()
        equations.append({
            "equation_id": f"eq_{number:03d}",
            "placeholder": f"[EQUATION_{number:03d}]",
            "section_id": None,
            "raw_latex": match.group(0).strip() or body.strip(),
            "context_before": clean_latex_text(text[max(0, start - 300):start])[-300:],
            "context_after": clean_latex_text(text[end:end + 300])[:300],
        })
        return f" [EQUATION_{number:03d}] "

    return EQUATION_RE.sub(replace, text), equations


def extract_sections(text: str) -> list[dict[str, Any]]:
    """Split the document body at \\section, \\subsection and \\subsubsection."""
    begin = text.find("\\begin{document}")
    if begin != -1:
        text = text[begin + len("\\begin{document}"):]
    end = text.find("\\end{document}")
    if end != -1:
        text = text[:end]

    matches = list(SECTION_RE.finditer(text))
    if not matches:
        plain = clean_latex_text(text)
        return [{"section_id": "sec_001", "level": 1, "heading": "Full text", "text": plain}] if plain else []
    sections = []
    for index, match in enumerate(matches, start=1):
        next_start = matches[index].start() if index < len(matches) else len(text)
        sections.append({
            "section_id": f"sec_{index:03d}",
            "level": SECTION_LEVELS[match.group(1)],
            "heading": clean_latex_text(match.group(2)) or f"Section {index}",
            "text": clean_latex_text(text[match.end():next_start]),
        })
    return sections


def parse_latex(flat_tex: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return sections, equations (linked to their section) and warnings."""
    with_placeholders, equations = extract_equations(strip_latex_comments(flat_tex))
    sections = extract_sections(with_placeholders)
    for equation in equations:
        placeholder = equation.pop("placeholder")
        equation["section_id"] = next(
            (s["section_id"] for s in sections if placeholder in s["text"]), None
        )
    warnings = []
    if not sections:
        warnings.append("no sections extracted")
    if not equations:
        warnings.append("no display equations extracted")
    return sections, equations, warnings


# Step entry point

def parse_paper(arxiv_id: str, paper_dir: Path, *, timeout_seconds: int) -> dict[str, Any]:
    """Download and parse one paper; failures are recorded in the record, not raised."""
    paper: dict[str, Any] = {
        "paper_id": paper_id_for(arxiv_id), "arxiv_id": arxiv_id, "title": "",
        "status": "failed", "sections_count": 0, "equations_count": 0,
        "errors": [], "warnings": [],
    }
    sections: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    paper_dir.mkdir(parents=True)
    try:
        paper.update(fetch_metadata(arxiv_id, timeout_seconds=timeout_seconds))
        source_root = download_source(arxiv_id, paper_dir / "source", timeout_seconds=timeout_seconds)
        main_tex = select_main_tex_file(source_root)
        if main_tex is None:
            raise SourceError("no .tex file in the e-print")
        flat_tex = flatten_latex_file(main_tex, source_root)
        (paper_dir / "flat.tex").write_text(flat_tex, encoding="utf-8")
        sections, equations, paper["warnings"] = parse_latex(flat_tex)
        if any(section["text"] for section in sections):
            paper["status"] = "partial" if paper["warnings"] else "success"
        else:
            paper["errors"].append("parsing produced no text")
    except (requests.RequestException, ET.ParseError, SourceError, OSError) as exc:
        paper["errors"].append(f"{type(exc).__name__}: {exc}")
    paper["sections_count"] = len(sections)
    paper["equations_count"] = len(equations)
    write_json(paper_dir / "sections.json", sections)
    write_json(paper_dir / "equations.json", equations)
    write_json(paper_dir / "paper.json", paper)
    return paper


def run_parsing(*, ids_file: Path, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    ids = read_id_list(ids_file)
    output_dir = create_output_dir(output_dir)
    papers = []
    for index, arxiv_id in enumerate(ids, start=1):
        paper = parse_paper(arxiv_id, output_dir / "papers" / paper_id_for(arxiv_id),
                            timeout_seconds=timeout_seconds)
        print(f"[{index}/{len(ids)}] {arxiv_id}: {paper['status']}", flush=True)
        papers.append(paper)
    write_jsonl(output_dir / "papers.jsonl", papers)
    summary = {
        "papers": len(papers),
        "status_counts": dict(Counter(paper["status"] for paper in papers)),
        "sections": sum(paper["sections_count"] for paper in papers),
        "equations": sum(paper["equations_count"] for paper in papers),
        "failed": {paper["arxiv_id"]: paper["errors"] for paper in papers if paper["status"] == "failed"},
    }
    write_manifest(output_dir, step="parse", params={"timeout_seconds": timeout_seconds},
                   inputs={"ids_file": ids_file}, summary=summary)
    return summary
