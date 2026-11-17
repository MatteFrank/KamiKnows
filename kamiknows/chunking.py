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


def _plot_references(record: dict, figure: dict, series: dict, series_index: int) -> dict:
    """Describe figures and observed series colours without exposing storage IDs or guessing legends."""
    assets = figure.get("source_assets", [])
    asset = record["provenance"]["asset_id"]
    image_index = next(i for i, a in enumerate(assets, 1) if a["asset_id"] == asset)
    suffix = figure["figure_id"].rsplit("::", 1)[-1]
    number = str(int(suffix)) if suffix.isdigit() else None
    reference = f"Figure {number}" if number else "Source figure"
    image = f"source image {image_index} of {len(assets)}"
    colour = series.get("raster_color_rgb")
    if colour is None:
        spec = next(
            (
                s
                for s in record.get("calibration", {}).get("series", [])
                if s.get("series_id") == series["series_id"]
            ),
            {},
        )
        indices = spec.get("path_indices", [])
        paths = record.get("raw_source", {}).get("paths", [])
        if indices and indices[0] < len(paths):
            colour = paths[indices[0]].get("color")
    label = series.get("label")
    if not label and isinstance(colour, (list, tuple)) and len(colour) == 3:
        r, g, b = colour[:3]
        # Dominant hue is observed geometry; caption-to-dataset assignment is not inferred here.
        dominant = sorted(zip((r, g, b), ("red", "green", "blue")), reverse=True)
        if dominant[0][0] > 0.25 and dominant[0][0] > 1.8 * max(dominant[1][0], 0.05):
            label = dominant[0][1] + " curve"
    label = label or (
        "heatmap cells" if series.get("geometry") == "histogram_2d_cells" else f"series {series_index}"
    )
    return {
        "figure": reference,
        "image": image,
        "series": label,
        "caption": figure.get("caption") or "Uncaptioned figure",
        "latex_label": figure.get("latex_label"),
        "number_basis": "source inventory order; paper numbering requires verification",
    }


def digitized_chunks(
    directories: list[Path], papers: list[dict[str, Any]], *, include_overviews: bool = True
) -> tuple[list[dict], dict]:
    """Publish successful displayed geometry as auditable evidence, never as original measurements."""
    import hashlib
    from kamiknows.plot_extractor.records import series_axis_metadata
    from kamiknows.plot_extractor.classification import reference_contexts

    chunks, inputs, seen = [], {}, set()
    discussions, overview_groups, layouts = {}, {}, {}
    by_paper = {p["paper_id"]: p for p in papers}
    for run_number, directory in enumerate(directories):
        directory = Path(directory).resolve()
        manifest = read_manifest(directory, "extract_plots")
        data = directory / "digitized_data.jsonl"
        if (
            manifest["params"].get("mode") != "digitize"
            or sha256_file(data) != manifest["summary"]["data_sha256"]
        ):
            raise ValueError("expected an intact digitization run")
        inputs[f"digitization_manifest_{run_number}"] = directory / "manifest.json"
        inputs[f"digitized_data_{run_number}"] = data
        source = manifest["inputs"]["classified_figures"]
        classified = Path(source["path"])
        if sha256_file(classified) != source["sha256"]:
            raise ValueError("classified figure hash mismatch")
        inputs[f"plot_figures_{run_number}"] = classified
        figures = {f["figure_id"]: f for f in read_jsonl(classified)}
        for record in read_jsonl(data):
            if record["extraction_status"] != "success":
                continue
            if record.get("quality", {}).get("structural_validity") != "passed":
                raise ValueError("successful plot record lacks structural checks")
            if record["paper_id"] not in by_paper:
                raise ValueError("digitized paper is outside the parsed corpus")
            provenance = record["provenance"]
            identity = (record["figure_id"], provenance["asset_id"])
            if identity in seen:
                raise ValueError("duplicate plot asset across digitization runs")
            seen.add(identity)
            figure = figures[record["figure_id"]]
            if record["figure_id"] not in discussions:
                flat = manifest.get("inputs", {}).get(record["paper_id"] + ":flat")
                discussion = ""
                if flat:
                    flat_path = Path(flat["path"])
                    if sha256_file(flat_path) != flat["sha256"]:
                        raise ValueError("plot reference-context source hash mismatch")
                    inputs["plot_context_" + record["paper_id"]] = flat_path
                    labels = list(figure.get("latex_labels", []))
                    if figure.get("latex_label"):
                        labels.append(figure["latex_label"])
                    flat_tex = flat_path.read_text()
                    contexts = reference_contexts(flat_tex, labels)
                    layouts[record["figure_id"]] = _figure_layout(figure, flat_tex)
                    discussion = " ".join(c["window"] for c in contexts[:2])
                discussions[record["figure_id"]] = " ".join(discussion.split()[:100])
            if not any(
                a["asset_id"] == provenance["asset_id"] and a.get("sha256") == provenance["asset_sha256"]
                for a in figure["source_assets"]
            ):
                raise ValueError("digitized asset does not match classified source")
            for series_index, series in enumerate(record["series"], 1):
                reference = _plot_references(record, figure, series, series_index)
                location = layouts.get(record["figure_id"], {}).get(provenance["asset_id"])
                if location:
                    reference["image"] = location + "; " + reference["image"]
                axes = ("x", "y", "z") if series["geometry"] == "histogram_2d_cells" else ("x", "y")
                values = {a: series[a] for a in axes}
                if not values["x"] or len({len(v) for v in values.values()}) != 1:
                    raise ValueError("empty or inconsistent plot coordinates")
                if any(not math.isfinite(v) for vs in values.values() for v in vs):
                    raise ValueError("nonfinite plot evidence")
                numeric_axis = axes[-1]
                peak = max(range(len(values[numeric_axis])), key=values[numeric_axis].__getitem__)
                fmt = lambda v: format(v, ".7g")
                header = [
                    f"Digitized plot: {reference['figure']}, {reference['image']}; {reference['series']}.",
                    f"Source file: {provenance['source_path']}. Caption: {figure.get('caption', '')}",
                    f"Original caption: {figure.get('caption_raw') or figure.get('caption', '')}",
                    f"Geometry: {series['geometry']}. Series appearance: {reference['series']}. Dataset association requires caption or legend evidence.",
                    "Values estimate displayed geometry, not original measurements. Scientific validation pending; original bin edges and experimental errors are unverified.",
                ]
                axis_metadata = series_axis_metadata(record, series)
                if discussions[record["figure_id"]]:
                    header.append("Paper discussion of this figure: " + discussions[record["figure_id"]])
                for axis, metadata in axis_metadata.items():
                    header.append(
                        f"{axis.upper()} scale: {metadata['scale']}; unit: {metadata['unit']}; unit status: {metadata['unit_status']}."
                    )
                header.append(
                    "Coordinates are physical axis values; display multipliers are already applied. Log10 describes axis spacing, not the unit of the values."
                )
                summary = [
                    "Numerical summary of extracted values: ranges and peak (maximum).",
                    f"Displayed elements: {len(values['x'])}.",
                ]
                summary.extend(
                    f"{axis.upper()} range: {fmt(min(vs))} to {fmt(max(vs))}." for axis, vs in values.items()
                )
                summary.append(
                    "Maximum "
                    + numeric_axis.upper()
                    + " (first tied maximum): "
                    + "; ".join(f"{a}={fmt(values[a][peak])}" for a in axes)
                    + "."
                )
                tables = [
                    "Extracted coordinates ("
                    + ", ".join(axes)
                    + "):\n"
                    + "\n".join(
                        ", ".join(fmt(values[a][i]) for a in axes)
                        for i in range(start, min(start + 24, len(values["x"])))
                    )
                    for start in range(0, len(values["x"]), 24)
                ]
                group = overview_groups.setdefault(
                    record["figure_id"],
                    {
                        "paper": by_paper[record["paper_id"]],
                        "figure": figure,
                        "reference": reference["figure"],
                        "items": [],
                    },
                )
                group["items"].append(
                    {
                        "text": f"{reference['image']}, {reference['series']} ({Path(provenance['source_path']).stem}): "
                        + "; ".join(
                            f"{a.upper()} {axis_metadata[a]['scale']} [{axis_metadata[a]['unit']}]"
                            for a in axes
                        )
                        + ". Peak (first tied maximum): "
                        + "; ".join(f"{a}={fmt(values[a][peak])}" for a in axes)
                        + ".",
                        "source": {
                            "asset_id": provenance["asset_id"],
                            "series_id": series["series_id"],
                            "data_id": record["data_id"],
                            "asset_sha256": provenance["asset_sha256"],
                            "axis_metadata": axis_metadata,
                            "digitized_data_sha256": manifest["summary"]["data_sha256"],
                        },
                    }
                )
                group["items"][-1]["retrieval_text"] = group["items"][-1]["text"].replace(location + "; ", "") if location else group["items"][-1]["text"]
                for part, body in enumerate(["\n".join(summary), *tables]):
                    text = "\n".join([*header, body])
                    key = hashlib.sha256(
                        f"{record['data_id']}:{series['series_id']}:{part}".encode()
                    ).hexdigest()[:20]
                    chunk = _chunk_record(
                        by_paper[record["paper_id"]],
                        "plot",
                        len(chunks) + 1,
                        {
                            "section_id": record["figure_id"],
                            "heading": f"{reference['figure']}, {reference['image']}, {reference['series']}: {'summary' if part == 0 else 'coordinates'}",
                        },
                        text,
                    )
                    chunk.update(
                        chunk_id=f"{record['paper_id']}::plot::{key}",
                        evidence_type="digitized_plot",
                        cut_reason="plot_summary" if part == 0 else "plot_table",
                        equation_ids=[],
                        flags=["extracted_values_scientific_review_pending"],
                        axis_metadata=axis_metadata,
                        plot_reference=reference,
                        retrieval_text=text.replace(location + "; ", "") if location else text,
                        retrieval_section_heading=chunk["section_heading"].replace(location + "; ", "") if location else chunk["section_heading"],
                        plot_provenance={
                            "data_id": record["data_id"],
                            "asset_id": provenance["asset_id"],
                            "asset_sha256": provenance["asset_sha256"],
                            "series_id": series["series_id"],
                            "digitized_data_sha256": manifest["summary"]["data_sha256"],
                        },
                    )
                    chunks.append(chunk)
    for figure_id, group in (overview_groups.items() if include_overviews else []):
        if len(group["items"]) < 2:
            continue
        for start in range(0, len(group["items"]), 6):
            items = group["items"][start : start + 6]
            content = (
                f"{group['reference']} numerical overview. Caption: {group['figure'].get('caption', '')}\n"
                f"Paper discussion: {discussions[figure_id]}\n"
                "Values estimate displayed geometry. Scientific validation pending; original counts, bins and experimental uncertainties are unavailable.\n"
                + "\n".join(item["text"] for item in items)
            )
            identity = hashlib.sha256(content.encode()).hexdigest()[:20]
            retrieval_content = content
            for item in items:
                retrieval_content = retrieval_content.replace(item["text"], item["retrieval_text"])
            chunk = _chunk_record(
                group["paper"],
                "plot",
                len(chunks) + 1,
                {"section_id": figure_id, "heading": group["reference"] + " numerical overview"},
                content,
            )
            chunk.update(
                chunk_id=f"{group['paper']['paper_id']}::plot::{identity}",
                evidence_type="digitized_plot",
                evidence_level="figure",
                cut_reason="plot_overview",
                equation_ids=[],
                flags=["extracted_values_scientific_review_pending"],
                plot_sources=[item["source"] for item in items],
                plot_reference={"figure": group["reference"], "caption": group["figure"].get("caption", "")},
                retrieval_text=retrieval_content,
            )
            chunks.append(chunk)
    return chunks, inputs


def run_chunking(
    *,
    parsed_dir: Path,
    output_dir: Path,
    chunker_name: str,
    chunker: Chunker,
    exclude_backmatter: bool,
    params: dict[str, Any],
    plots: bool = False,
    digitized_dirs: list[Path] | None = None,
    plot_overviews: bool = True,
) -> dict[str, Any]:
    read_manifest(parsed_dir, "parse")
    papers = [p for p in read_jsonl(parsed_dir / "papers.jsonl") if p["status"] != "failed"]
    if not papers:
        raise ValueError(f"no parsed papers in {parsed_dir}")
    plot_chunks, plot_inputs = digitized_chunks(
        digitized_dirs or [], papers, include_overviews=plot_overviews
    )
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

    chunks.extend(plot_chunks)
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
        "plot_chunks": len(plot_chunks),
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
            "digitized_dirs": digitized_dirs or [],
            "plot_overviews": plot_overviews,
            **params,
        },
        inputs={"papers": parsed_dir / "papers.jsonl", **plot_inputs},
        summary=summary,
    )
    return summary
