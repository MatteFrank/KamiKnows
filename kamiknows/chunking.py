"""Step 2: split parsed sections into retrieval chunks.

Three chunkers share one output format; only the boundary rule changes:

    word      fixed windows of whitespace words with overlap
    sentence  whole sentences packed up to a word budget, overlap of whole sentences
    semantic  like sentence, but cuts where the embedding similarity between
              neighbouring sentence windows drops (needs an embedding model)

Chunks never cross section boundaries. Sizes are always whitespace words.
Every chunk records why it ends (``cut_reason``): ``length_limit``,
``section_end`` or, for the semantic chunker, ``semantic_drop``; and the ids
of the equations of its section (``equation_ids``, see ``equations.jsonl``).
Output: ``chunks.jsonl``, ``equations.jsonl`` (+ ``section_exclusions.jsonl``
with --exclude-backmatter, + ``semantic_boundaries.jsonl`` and optional
``plots/`` for the semantic chunker).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from kamiknows.common import (
    create_output_dir,
    read_json,
    read_jsonl,
    read_manifest,
    sha256_file,
    write_jsonl,
    write_manifest,
)

Chunker = Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]


def chunk_id(paper_id: str, chunker: str, number: int) -> str:
    return f"{paper_id}::{chunker}::{number:04d}"


def word_count(text: str) -> int:
    return len(text.split())


# Back matter

EXCLUDED_HEADINGS = {
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "bibliography",
    "references",
    "reference",
    "reference list",
    "literature cited",
}
# Unexpanded bibliography commands left at the end of the last section by the parser.
TRAILING_MARKER = re.compile(r"(?<!\S)(?:refs|biblio\s+spphys)\s*$")


def filter_backmatter(sections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop acknowledgment/bibliography sections; return kept sections and an audit.

    Headings are matched exactly after normalization; bibliography that is not
    under its own heading is not guessed from the prose.
    """
    kept, audit = [], []
    for section in sections:
        section = dict(section)
        heading = re.sub(r"[^a-z]+", " ", str(section.get("heading") or "").casefold()).strip()
        identity = {"section_id": section.get("section_id"), "heading": section.get("heading")}
        if heading in EXCLUDED_HEADINGS:
            audit.append(
                {**identity, "action": "exclude_section", "removed_words": word_count(section["text"])}
            )
            continue
        marker = TRAILING_MARKER.search(section["text"])
        if marker:
            section["text"] = section["text"][: marker.start()].rstrip()
            audit.append({**identity, "action": "remove_trailing_marker", "removed_text": marker.group()})
        kept.append(section)
    return kept, audit


# Sentences

ABBREVIATIONS = re.compile(r"(?:\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Sec|Secs|Dr|Prof|vs)|\b[A-Z])\.$", re.I)


def split_sentences(text: str) -> list[str]:
    """Heuristic punctuation splitter that keeps common abbreviations together."""
    text = " ".join(text.split())
    boundaries = [
        match.end()
        for match in re.finditer(r"[.!?][\"')\]]*\s+", text)
        if not (text[: match.start() + 1].endswith("\\.") or ABBREVIATIONS.search(text[: match.start() + 1]))
    ]
    sentences, start = [], 0
    for end in boundaries + [len(text)]:
        if text[start:end].strip():
            sentences.append(text[start:end].strip())
        start = end
    return sentences


def overlap_start(lengths: list[int], start: int, end: int, overlap_words: int) -> int:
    """First sentence of the next chunk: the longest suffix of [start, end) within the
    overlap budget, never the whole previous chunk (so every chunk makes progress)."""
    next_start, overlap = end, 0
    while next_start > start + 1 and overlap + lengths[next_start - 1] <= overlap_words:
        next_start -= 1
        overlap += lengths[next_start]
    return next_start


def check_coverage(spans: list[tuple[int, int]], sentence_count: int, section_id: Any) -> None:
    """Every sentence of a section must end up in at least one chunk."""
    covered = 0
    for start, end in spans:
        if start > covered or end <= start:
            break
        covered = max(covered, end)
    if covered != sentence_count:
        raise RuntimeError(f"sentence coverage failed in section {section_id}")


# Chunkers


def make_word_chunker(*, target_words: int, overlap_words: int, min_tail_words: int = 80) -> Chunker:
    def chunk(paper: dict[str, Any], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks = []
        step = target_words - overlap_words
        for section in sections:
            words = section["text"].split()
            start = 0
            while start < len(words):
                end = min(len(words), start + target_words)
                if 0 < len(words) - end < min_tail_words:
                    end = len(words)  # absorb a short tail into the last window
                record = _chunk_record(paper, "word", len(chunks) + 1, section, " ".join(words[start:end]))
                record["cut_reason"] = "section_end" if end == len(words) else "length_limit"
                chunks.append(record)
                if end == len(words):
                    break
                start += step
        return chunks

    return chunk


def make_sentence_chunker(*, target_words: int, overlap_words: int) -> Chunker:
    def chunk(paper: dict[str, Any], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks = []
        for section in sections:
            sentences = split_sentences(section["text"])
            lengths = [word_count(s) for s in sentences]
            spans, start = [], 0
            while start < len(sentences):
                end, size = start + 1, lengths[start]
                while end < len(sentences) and size + lengths[end] <= target_words:
                    size += lengths[end]
                    end += 1
                spans.append((start, end))
                record = _chunk_record(
                    paper, "sentence", len(chunks) + 1, section, " ".join(sentences[start:end])
                )
                record["sentence_span"] = [start, end]
                record["cut_reason"] = "section_end" if end == len(sentences) else "length_limit"
                if lengths[start] > target_words:
                    record["flags"].append("oversize_sentence")
                chunks.append(record)
                if end == len(sentences):
                    break
                start = overlap_start(lengths, start, end, overlap_words)
            check_coverage(spans, len(sentences), section["section_id"])
        return chunks

    return chunk


class SemanticChunker:
    """Sentence packing with cuts at drops of embedding similarity.

    At every sentence boundary i the cosine between the mean embedding of the
    ``window_sentences`` sentences before and after i is computed. A boundary
    is a candidate when its similarity is below the section's
    ``threshold_quantile``; with ``cut_policy='first_drop'`` it must also be at
    least ``min_score_drop`` below the previous boundary. Chunks stay within
    [min_words, target_words] unless a single sentence or a section is shorter
    or longer. Every boundary score is kept in ``self.boundaries`` for auditing.
    """

    def __init__(
        self,
        encode: Callable[[list[str]], list[list[float]]],
        *,
        target_words: int,
        overlap_words: int,
        min_words: int,
        window_sentences: int,
        threshold_quantile: float,
        cut_policy: str,
        min_score_drop: float,
    ) -> None:
        if not 1 <= min_words <= target_words:
            raise ValueError("require 1 <= min_words <= target_words")
        if window_sentences < 1 or not 0 <= threshold_quantile <= 1 or not 0 <= min_score_drop <= 2:
            raise ValueError("invalid window_sentences, threshold_quantile or min_score_drop")
        if cut_policy not in {"lowest_similarity", "first_drop"}:
            raise ValueError(f"unknown cut policy: {cut_policy}")
        self.encode = encode
        self.target_words = target_words
        self.overlap_words = overlap_words
        self.min_words = min_words
        self.window = window_sentences
        self.quantile = threshold_quantile
        self.cut_policy = cut_policy
        self.min_score_drop = min_score_drop
        self.boundaries: list[dict[str, Any]] = []
        self.sections: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []  # word span and overlap of every chunk, for the plots

    def __call__(self, paper: dict[str, Any], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for section in sections:
            sentences = split_sentences(section["text"])
            if sentences:
                chunks.extend(self._chunk_section(paper, section, sentences, first_number=len(chunks) + 1))
        return chunks

    def _chunk_section(self, paper, section, sentences, *, first_number):
        vectors = self.encode(sentences)
        if len(vectors) != len(sentences) or any(not all(map(math.isfinite, v)) for v in vectors):
            raise ValueError("encoder returned invalid sentence vectors")
        lengths = [word_count(s) for s in sentences]
        positions = [0]
        for length in lengths:
            positions.append(positions[-1] + length)
        scores = {
            i: _cosine(
                _mean_vector(vectors[max(0, i - self.window) : i]), _mean_vector(vectors[i : i + self.window])
            )
            for i in range(1, len(sentences))
        }
        threshold = _quantile(list(scores.values()), self.quantile) if scores else None
        section_mean = mean(scores.values()) if scores else None
        drops = {i: scores.get(i - 1, section_mean) - score for i, score in scores.items()}
        eligible = {
            i: score < threshold
            and (self.cut_policy == "lowest_similarity" or drops[i] >= self.min_score_drop)
            for i, score in scores.items()
        }
        identity = {
            "paper_id": paper["paper_id"],
            "arxiv_id": paper["arxiv_id"],
            "section_id": section["section_id"],
            "section_heading": section.get("heading"),
        }
        rows = {
            i: {
                **identity,
                "boundary": i,
                "word_position": positions[i],
                "similarity": score,
                "threshold": threshold,
                "score_drop": drops[i],
                "candidate": eligible[i],
                "selected": False,
            }
            for i, score in scores.items()
        }

        def fill(start: int) -> int:
            end = start + 1
            while end < len(sentences) and positions[end + 1] - positions[start] <= self.target_words:
                end += 1
            return end

        chunks, spans = [], []
        start, previous_end = 0, 0
        while start < len(sentences):
            end = fill(start)
            # Drop overlap that would prevent the chunk from adding a new sentence.
            while end <= previous_end and start < previous_end:
                start += 1
                end = fill(start)
            candidates = [
                i
                for i in range(max(start + 1, previous_end + 1), end + 1)
                if i in scores
                and eligible[i]
                and positions[i] - positions[start] >= self.min_words
                and (self.cut_policy == "lowest_similarity" or positions[-1] - positions[i] >= self.min_words)
            ]
            if self.cut_policy == "first_drop" and candidates:
                end, reason = candidates[0], "semantic_drop"
            elif end == len(sentences):
                reason = "section_end"
            elif candidates:
                end, reason = min(candidates, key=lambda i: (scores[i], i)), "semantic_drop"
            else:
                reason = "length_limit"
                if self.cut_policy == "first_drop" and positions[-1] - positions[end] < self.min_words:
                    # Avoid a short final chunk when a boundary satisfies both limits.
                    balanced = [
                        i
                        for i in range(max(start + 1, previous_end + 1), end + 1)
                        if positions[i] - positions[start] >= self.min_words
                        and positions[-1] - positions[i] >= self.min_words
                    ]
                    if balanced:
                        end = balanced[-1]
            record = _chunk_record(
                paper, "semantic", first_number + len(chunks), section, " ".join(sentences[start:end])
            )
            record["sentence_span"] = [start, end]
            record["cut_reason"] = reason
            words = positions[end] - positions[start]
            if lengths[start] > self.target_words:
                record["flags"].append("oversize_sentence")
            if words < self.min_words:
                record["flags"].append("below_min_words")
            if words > self.target_words:
                record["flags"].append("above_max_words")
            chunks.append(record)
            spans.append((start, end))
            self.groups.append(
                {
                    **identity,
                    "chunk_id": record["chunk_id"],
                    "start_word": positions[start],
                    "end_word": positions[end],
                    "words": words,
                    "cut_reason": reason,
                    "overlap_words": max(0, positions[previous_end] - positions[start]),
                }
            )
            if end in rows:
                rows[end]["selected"] = True
            previous_end = end
            if end == len(sentences):
                break
            start = overlap_start(lengths, start, end, self.overlap_words)
        check_coverage(spans, len(sentences), section["section_id"])
        self.boundaries.extend(rows.values())
        self.sections.append(
            {
                **identity,
                "sentences": len(sentences),
                "words": positions[-1],
                "threshold": threshold,
                "mean_similarity": section_mean,
            }
        )
        return chunks

    def summary(self) -> dict[str, Any]:
        scores = [row["similarity"] for row in self.boundaries]
        return {"boundaries": len(scores), "mean_boundary_similarity": mean(scores) if scores else None}

    def write_diagnostics(self, output_dir: Path, lengths: dict[str, list[int]], *, plots: bool) -> None:
        write_jsonl(output_dir / "semantic_boundaries.jsonl", self.boundaries)
        if plots:
            plot_semantic_diagnostics(output_dir / "plots", self, lengths)


def _chunk_record(
    paper: dict[str, Any], chunker: str, number: int, section: dict[str, Any], text: str
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id(paper["paper_id"], chunker, number),
        "paper_id": paper["paper_id"],
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "section_id": section["section_id"],
        "section_heading": section.get("heading") or "",
        "text": text,
        "words": word_count(text),
        "flags": [],
    }


def _cosine(a: list[float], b: list[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if not denominator:
        raise ValueError("zero vector; cannot compute similarity")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / denominator))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    return [mean(column) for column in zip(*vectors)]


def _quantile(values: list[float], fraction: float) -> float:
    """Linearly interpolated quantile."""
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def plot_semantic_diagnostics(
    plots_dir: Path, chunker: SemanticChunker, lengths: dict[str, list[int]]
) -> None:
    """PNG and PDF per section plus an overview.

    Section figure, top: boundary cosine similarity, its decrease from the
    previous boundary (right axis, first_drop only), quantile threshold, section
    mean, unselected candidates and cuts coloured by reason. Bottom: chunk spans,
    black segments are overlap. Overview: similarity histogram, chunk lengths of
    the word/sentence/semantic chunkers on the same sections, cut reasons.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir()
    colors = {"semantic_drop": "tab:red", "length_limit": "tab:orange", "section_end": "tab:green"}

    def save(fig, stem: str) -> None:
        for suffix in ("png", "pdf"):
            fig.savefig(plots_dir / f"{stem}.{suffix}", bbox_inches="tight", dpi=160)
        plt.close(fig)

    for section in chunker.sections:
        key = (section["paper_id"], section["section_id"])
        rows = [r for r in chunker.boundaries if (r["paper_id"], r["section_id"]) == key]
        groups = [g for g in chunker.groups if (g["paper_id"], g["section_id"]) == key]
        fig, (ax, spans) = plt.subplots(
            2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax.plot(
            [r["word_position"] for r in rows],
            [r["similarity"] for r in rows],
            ".-",
            label="Cosine similarity",
        )
        if chunker.cut_policy == "first_drop" and rows:
            drop_axis = ax.twinx()
            drop_axis.plot(
                [r["word_position"] for r in rows],
                [r["score_drop"] for r in rows],
                color="teal",
                alpha=0.35,
                label="Cosine decrease",
            )
            drop_axis.axhline(
                chunker.min_score_drop,
                color="teal",
                linestyle=":",
                label=f"Min decrease {chunker.min_score_drop:g}",
            )
            drop_axis.set_ylabel("Previous cosine − current cosine", color="teal")
            drop_axis.legend(fontsize=6, loc="lower right")
        if section["threshold"] is not None:
            ax.axhline(
                section["threshold"], linestyle="--", color="purple", label="Section quantile threshold"
            )
            ax.axhline(section["mean_similarity"], linestyle=":", color="gray", label="Section mean")
        rejected = [r for r in rows if r["candidate"] and not r["selected"]]
        ax.scatter(
            [r["word_position"] for r in rejected],
            [r["similarity"] for r in rejected],
            marker="x",
            color="gray",
            label="Unselected semantic candidate",
        )
        used = set()
        for index, group in enumerate(groups):
            reason = group["cut_reason"]
            ax.axvline(
                group["end_word"],
                color=colors[reason],
                alpha=0.7,
                label=reason.replace("_", " ") if reason not in used else None,
            )
            used.add(reason)
            spans.plot(
                [group["start_word"], group["end_word"]],
                [index % 3] * 2,
                lw=5,
                color=colors[reason],
                alpha=0.65,
            )
            if group["overlap_words"]:
                spans.plot(
                    [group["start_word"], group["start_word"] + group["overlap_words"]],
                    [index % 3] * 2,
                    lw=8,
                    color="black",
                    alpha=0.35,
                )
        ax.set(
            ylabel="Cosine similarity",
            ylim=(-1.05, 1.05),
            title=f"{section['arxiv_id']} — {section['section_heading']} ({chunker.cut_policy})",
        )
        ax.legend(fontsize=7, loc="best")
        spans.set(
            xlabel="Word position within section",
            ylabel="Chunks",
            yticks=[],
            title="Chunk spans; black segments show overlap",
            xlim=(0, max(1, section["words"])),
        )
        save(fig, f"{section['paper_id']}_{section['section_id']}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    values = [r["similarity"] for r in chunker.boundaries]
    axes[0].hist(values, bins=25)
    if values:
        axes[0].axvline(mean(values), color="red", label="Global boundary mean")
        axes[0].legend(fontsize=8)
    axes[0].set(title="Boundary similarity distribution", xlabel="Cosine similarity", ylabel="Boundaries")
    axes[1].boxplot(list(lengths.values()), tick_labels=list(lengths))
    axes[1].set(title="Chunk lengths", ylabel="Whitespace words")
    reasons = Counter(g["cut_reason"] for g in chunker.groups)
    axes[2].bar(list(reasons), list(reasons.values()), color=[colors[r] for r in reasons])
    axes[2].set(title="Selected cut reasons", ylabel="Cuts")
    axes[2].tick_params(axis="x", labelrotation=25)
    save(fig, "semantic_overview")


# Step entry point


def _figure_layout(figure: dict, flat_tex: str) -> dict[str, str]:
    """Name source-image positions only when explicit TeX row breaks establish rows."""
    assets = figure.get("source_assets", [])
    if not assets or any(not a.get("flat_span") for a in assets):
        return {}
    rows = [[]]
    for index, asset in enumerate(assets):
        if index:
            between = flat_tex[assets[index - 1]["flat_span"][1]:asset["flat_span"][0]]
            between = re.sub(r"(?<!\\)%[^\n]*", "", between)
            depth, position, new_row = 0, 0, False
            while position < len(between):
                char = between[position]
                if char == "\\":
                    match = re.match(r"\\(?:\\|newline\b|linebreak\b|par\b)", between[position:])
                    if match and depth == 0:
                        new_row = True
                    position += 2 if position + 1 < len(between) and between[position + 1] in "\\{}%" else 1
                    continue
                depth += (char == "{") - (char == "}")
                position += 1
            if new_row:
                rows.append([])
        rows[-1].append(asset)
    if len(rows) < 2:
        return {}  # Automatic line wrapping does not establish a verified grid.
    result = {}
    for r, row in enumerate(rows):
        vertical = "top" if r == 0 else "bottom" if r == len(rows) - 1 else "middle" if len(rows) == 3 else f"row {r + 1}"
        for c, asset in enumerate(row):
            horizontal = "left" if c == 0 else "right" if c == len(row) - 1 else "middle" if len(row) == 3 else f"column {c + 1}"
            position_name = vertical if len(row) == 1 else f"{vertical}-{horizontal}"
            result[asset["asset_id"]] = f"{position_name} panel (explicit source layout)"
    return result


def _figure_name(figure_id: str) -> str:
    suffix = figure_id.rsplit("::", 1)[-1]
    return f"Figure {int(suffix)}" if suffix.isdigit() else "Source figure"


def _series_label(record: dict, series: dict, index: int) -> str:
    """Name a series by its legend label, else by its observed colour; legends are never guessed."""
    if series.get("label"):
        return series["label"]
    colour, kind = series.get("raster_color_rgb"), "curve"
    if colour is None:
        spec = next((s for s in record.get("calibration", {}).get("series", [])
                     if s.get("series_id") == series["series_id"]), {})
        indices = spec.get("path_indices", [])
        paths = record.get("raw_source", {}).get("paths", [])
        if indices and indices[0] < len(paths):
            # A filled shape carries its colour in "fill"; a stroked one in "color".
            path = paths[indices[0]]
            colour = path.get("color") or path.get("fill")
            kind = "curve" if path.get("color") else "filled area"
    name = _colour_name(colour)
    if name:
        return f"{name} {kind}"
    return "heatmap cells" if series.get("geometry") == "histogram_2d_cells" else f"series {index}"


HUES = [(15, "red"), (45, "orange"), (70, "yellow"), (170, "green"), (200, "cyan"), (260, "blue"),
        (300, "purple"), (345, "magenta"), (360, "red")]


def _colour_name(colour: Any) -> str | None:
    """Basic colour name from the hue, so that legends and captions can be matched by colour."""
    import colorsys

    if not (isinstance(colour, (list, tuple)) and len(colour) == 3):
        return None
    hue, saturation, value = colorsys.rgb_to_hsv(*colour)
    if value < 0.15:
        return "black"
    if saturation < 0.2:
        return "grey"
    return next(name for limit, name in HUES if hue * 360 < limit)


PLOT_KINDS = {
    "histogram_2d_cells": "heatmap (2D cells, value shown by colour)",
    "rendered_step_intervals": "1D histogram (drawn steps)",
    "rendered_curve_samples": "curve (sampled line)",
}
DISPLAYED_ONLY = ("Values estimate the displayed figure, not the original measurements; "
                  "original bins, counts and uncertainties are not available.")


def plot_chunks(directory: Path | None, papers: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    """One chunk per plotted image (a plot, or one panel of a multi-panel figure).

    The chunk describes the plot so that retrieval can find it: caption, the
    paper sentences that cite it, panel position, axes and units, and a few
    characteristic numbers (ranges, maximum) computed by kamiknows.compute.
    It never holds the extracted values: ``plot_link`` names the digitized
    record and the compute handles of its series, so a retrieved chunk leads
    to its data. Plots that were not digitized get a chunk saying why, so a
    question about them finds them instead of a wrong plot that has values.
    """
    from kamiknows.compute.sources.plots import DATA_FILE
    from kamiknows.plot_extractor.classification import reference_contexts

    if directory is None:
        return [], {}
    by_paper = {p["paper_id"]: p for p in papers}
    directory = Path(directory).resolve()
    manifest = read_manifest(directory, "extract_plots")
    data = directory / DATA_FILE
    data_sha = manifest["summary"]["data_sha256"]
    if manifest["params"].get("mode") != "digitize" or sha256_file(data) != data_sha:
        raise ValueError("expected an intact digitization run")
    source = manifest["inputs"]["classified_figures"]
    classified = Path(source["path"])
    if sha256_file(classified) != source["sha256"]:
        raise ValueError("classified figure hash mismatch")
    inputs: dict[str, Path] = {"digitization_manifest": directory / "manifest.json", "digitized_data": data,
                               "plot_figures": classified}
    figures = {f["figure_id"]: f for f in read_jsonl(classified)}
    flat_sources = {key[: -len(":flat")]: (Path(entry["path"]), entry["sha256"])
                    for key, entry in manifest.get("inputs", {}).items() if key.endswith(":flat")}
    records: dict[str, tuple[dict, dict]] = {}
    for record in read_jsonl(data):
        if record.get("paper_id") not in by_paper:
            raise ValueError("digitized paper is outside the parsed corpus")
        if record["extraction_status"] == "success" and record.get("quality", {}).get("structural_validity") != "passed":
            raise ValueError("successful plot record lacks structural checks")
        figure = figures[record["figure_id"]]
        provenance = record["provenance"]
        if not any(a["asset_id"] == provenance["asset_id"] and a.get("sha256") == provenance["asset_sha256"]
                   for a in figure["source_assets"]):
            raise ValueError("digitized asset does not match classified source")
        if provenance["asset_id"] in records:
            raise ValueError(f"plot asset {provenance['asset_id']} recorded more than once")
        records[provenance["asset_id"]] = (record, figure)

    discussions: dict[str, str] = {}
    layouts: dict[str, dict[str, str]] = {}
    chunks = []
    order = {p["paper_id"]: i for i, p in enumerate(papers)}
    for asset_id, (record, figure) in sorted(
        records.items(), key=lambda item: (order[item[1][0]["paper_id"]], item[1][0]["figure_id"], item[0])
    ):
        figure_id = record["figure_id"]
        if figure_id not in discussions:
            discussions[figure_id], layouts[figure_id] = "", {}
            if record["paper_id"] in flat_sources:
                flat_path, flat_sha = flat_sources[record["paper_id"]]
                if sha256_file(flat_path) != flat_sha:
                    raise ValueError("plot reference-context source hash mismatch")
                inputs["plot_context_" + record["paper_id"]] = flat_path
                flat_tex = flat_path.read_text()
                labels = [*figure.get("latex_labels", []), *([figure["latex_label"]] if figure.get("latex_label") else [])]
                window = " ".join(c["window"] for c in reference_contexts(flat_tex, labels)[:2])
                discussions[figure_id] = " ".join(window.split()[:100])
                layouts[figure_id] = _figure_layout(figure, flat_tex)
        chunks.append(_plot_chunk(record, figure, by_paper[record["paper_id"]], discussions[figure_id],
                                  layouts[figure_id], data_sha))
    # Every panel of a multi-panel figure links all its panels, so a planner that
    # retrieved one panel can use the one the question means (caption and layout tell which).
    panels = defaultdict(list)
    for chunk in chunks:
        panels[chunk["plot_link"]["figure_id"]].append(chunk)
    for group in panels.values():
        if len(group) > 1:
            summary = [{key: c["plot_link"][key] for key in ("image", "position", "asset_id", "extraction_status", "series")}
                       for c in group]
            for chunk in group:
                chunk["plot_link"]["figure_panels"] = summary
    return chunks, inputs


def _plot_chunk(record: dict, figure: dict, paper: dict, discussion: str,
                layout: dict[str, str], data_sha: str) -> dict:
    """The chunk of one plotted image; lines marked False are shown but not ranked on."""
    from kamiknows.compute.sources.plots import record_entries

    asset_id = record["provenance"]["asset_id"]
    assets = figure["source_assets"]
    number = next(i for i, a in enumerate(assets, 1) if a["asset_id"] == asset_id)
    figure_name = _figure_name(record["figure_id"])
    position = layout.get(asset_id)
    image = f"image {number} of {len(assets)}" if len(assets) > 1 else "single image"
    caption = figure.get("caption") or "Uncaptioned figure"
    lines: list[tuple[str, bool]] = [(f"Plot: {figure_name} of arXiv:{paper['arxiv_id']}, {image}.", True)]
    if position:
        lines.append((f"Panel position: {position}.", False))
        lines.append(("Figure layout (explicit source layout): " + "; ".join(
            f"image {i}: {layout[a['asset_id']].split(' (')[0]}" for i, a in enumerate(assets, 1)
            if a["asset_id"] in layout) + ".", False))
    lines.append((f"Caption: {caption}", True))
    if figure.get("caption_raw") and figure["caption_raw"] != caption:
        lines.append((f"Original caption: {figure['caption_raw']}", True))
    if discussion:
        lines.append((f"Paper discussion of this figure: {discussion}", True))
    raw_series = {s["series_id"]: s for s in record.get("series", [])}
    entries = record_entries(record)
    usable = [e for e in entries if e.series is not None]
    labels = {e.handle: _series_label(record, raw_series[e.source["series_id"]], i)
              for i, e in enumerate(entries, 1) if "series_id" in e.source}
    repeated = Counter(labels.values())
    labels = {handle: f"{label} {i}" if repeated[label] > 1 else label
              for i, (handle, label) in enumerate(labels.items(), 1)}
    if usable:
        lines += [(text, True) for text in _data_lines(entries, raw_series, labels)]
    else:
        lines.append((f"Data status: {entries[0].reason}; numerical questions about this plot cannot be "
                      "answered from extracted data.", True))
    text = "\n".join(line for line, _ in lines)
    heading = f"{figure_name}, {position.split(' (')[0] if position else image}"
    chunk = _chunk_record(paper, "plot", number, {"section_id": record["figure_id"], "heading": heading}, text)
    chunk.update(
        chunk_id=f"{paper['paper_id']}::plot::{hashlib.sha256(asset_id.encode()).hexdigest()[:20]}",
        evidence_type="plot",
        cut_reason="plot",
        equation_ids=[],
        flags=["extracted_values_scientific_review_pending" if usable else "values_not_extracted"],
        retrieval_text="\n".join(line for line, ranked in lines if ranked),
        retrieval_section_heading=f"{figure_name}, {image}",
        plot_link={
            "asset_id": asset_id,
            "figure_id": record["figure_id"],
            "image": f"{number} of {len(assets)}",
            "position": position.split(" (")[0] if position else None,
            "data_id": record.get("data_id"),
            "extraction_status": record["extraction_status"],
            "digitized_data_sha256": data_sha,
            "series": [{"handle": e.handle, "series_id": e.source["series_id"], "label": labels[e.handle],
                        "usable": e.series is not None} for e in entries if "series_id" in e.source],
        },
    )
    return chunk


def _data_lines(entries: list, raw_series: dict[str, dict], labels: dict[str, str]) -> list[str]:
    """Kind, axes and a few characteristic numbers of every series, computed with kamiknows.compute."""
    from kamiknows.compute import argextremum, describe

    fmt = lambda value: format(value, ".4g")
    kinds = sorted({PLOT_KINDS.get(raw_series[e.source["series_id"]].get("geometry"), "plot") for e in entries})
    lines = [f"Type: {', '.join(kinds)}; {len(entries)} series."]
    axes_lines: list[str] = []
    for entry in entries:
        label = labels[entry.handle]
        if entry.series is None:
            lines.append(f"- {label}: values not usable ({entry.reason}).")
            continue
        series, info = entry.series, describe(entry.series)
        axes = ", ".join(
            f"{name.upper()}{' (colour)' if axis['role'] == 'value' and series.dim == 2 else ''} in "
            f"{axis['unit'] if axis['unit'] != 'unknown' else 'an unestablished unit'} ({axis['scale']})"
            for name, axis in info["axes"].items()
        )
        if f"Axes: {axes}." not in axes_lines:
            axes_lines.append(f"Axes: {axes}.")
        elements = "cells" if series.dim == 2 else "bins" if series.has_support else "points"
        unit = lambda value: "" if str(value) in ("1", "unknown") else f" {value}"
        ranges = "; ".join(
            f"{name.upper()} from {fmt(axis['range'][0])} to {fmt(axis['range'][1])}{unit(axis['unit'])}"
            for name, axis in info["axes"].items() if axis["role"] == "domain" and "range" in axis
        )
        peaks = argextremum(series)
        where = ", ".join(f"{name.upper()} = {fmt(q.value)}{unit(q.unit)}" for name, q in peaks[0].coords.items())
        tied = f" ({len(peaks)} tied {elements}, first shown)" if len(peaks) > 1 else ""
        below = len(raw_series[entry.source["series_id"]].get("below_range_intervals") or [])
        hidden = f"; {below} more bins drawn below the plotted range (values not displayed)" if below else ""
        lines.append(f"- {label}: {series.size} {elements}; {ranges}; maximum "
                     f"{fmt(peaks[0].value.value)}{unit(peaks[0].value.unit)} at {where}{tied}{hidden}.")
    return [lines[0], *axes_lines, *lines[1:], DISPLAYED_ONLY]


def run_chunking(
    *,
    parsed_dir: Path,
    output_dir: Path,
    chunker_name: str,
    chunker: Chunker,
    exclude_backmatter: bool,
    params: dict[str, Any],
    plots: bool = False,
    digitized_dir: Path | None = None,
) -> dict[str, Any]:
    read_manifest(parsed_dir, "parse")
    papers = [p for p in read_jsonl(parsed_dir / "papers.jsonl") if p["status"] != "failed"]
    if not papers:
        raise ValueError(f"no parsed papers in {parsed_dir}")
    figure_chunks, plot_inputs = plot_chunks(digitized_dir, papers)
    output_dir = create_output_dir(output_dir)
    chunks: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    sections_by_paper = []
    for paper in papers:
        paper_dir = parsed_dir / "papers" / paper["paper_id"]
        sections = [s for s in read_json(paper_dir / "sections.json") if s["text"].strip()]
        if exclude_backmatter:
            sections, audit = filter_backmatter(sections)
            exclusions.extend({"paper_id": paper["paper_id"], **row} for row in audit)
        equations_by_section = defaultdict(list)
        for equation in read_json(paper_dir / "equations.json"):
            equation = {
                **equation,
                "equation_id": f"{paper['paper_id']}::{equation['equation_id']}",
                "paper_id": paper["paper_id"],
                "arxiv_id": paper["arxiv_id"],
            }
            equations.append(equation)
            equations_by_section[equation["section_id"]].append(equation["equation_id"])
        sections_by_paper.append((paper, sections))
        paper_chunks = chunker(paper, sections)
        for chunk in paper_chunks:
            chunk["equation_ids"] = equations_by_section[chunk["section_id"]]
        chunks.extend(paper_chunks)
        print(f"{paper['arxiv_id']}: {len(paper_chunks)} chunks", flush=True)

    chunks.extend(figure_chunks)
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise RuntimeError("duplicate chunk ids")
    text_counts = Counter(" ".join(c["text"].lower().split()) for c in chunks)
    for chunk in chunks:
        if text_counts[" ".join(chunk["text"].lower().split())] > 1:
            chunk["flags"].append("duplicate_text")

    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "equations.jsonl", equations)
    if exclude_backmatter:
        write_jsonl(output_dir / "section_exclusions.jsonl", exclusions)
    if isinstance(chunker, SemanticChunker):
        # Word and sentence chunks of the same sections, only for the length comparison plot.
        sizes = {"target_words": chunker.target_words, "overlap_words": chunker.overlap_words}
        lengths = {
            name: [c["words"] for paper, sections in sections_by_paper for c in baseline(paper, sections)]
            for name, baseline in (
                ("word", make_word_chunker(**sizes)),
                ("sentence", make_sentence_chunker(**sizes)),
            )
        }
        lengths["semantic"] = [c["words"] for c in chunks]
        chunker.write_diagnostics(output_dir, lengths, plots=plots)

    lengths = sorted(c["words"] for c in chunks)
    summary = {
        "papers": len(papers),
        "chunks": len(chunks),
        "plot_chunks": len(figure_chunks),
        "chunks_per_paper": dict(Counter(c["paper_id"] for c in chunks)),
        "words": {"min": lengths[0], "max": lengths[-1], "mean": round(mean(lengths), 1)},
        "cut_reasons": dict(Counter(c["cut_reason"] for c in chunks)),
        "flags": dict(Counter(flag for c in chunks for flag in c["flags"])),
        "equations": len(equations),
        "excluded_sections": sum(row["action"] == "exclude_section" for row in exclusions),
    }
    if isinstance(chunker, SemanticChunker):
        summary.update(chunker.summary())
    write_manifest(
        output_dir,
        step="chunk",
        params={
            "chunker": chunker_name,
            "exclude_backmatter": exclude_backmatter,
            "digitized_dir": digitized_dir,
            **params,
        },
        inputs={"papers": parsed_dir / "papers.jsonl", **plot_inputs},
        summary=summary,
    )
    return summary
