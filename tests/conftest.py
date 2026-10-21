import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kamiknows.common import write_json, write_jsonl, write_manifest  # noqa: E402

PAPERS = [
    {
        "paper_id": "1111_00001v1", "arxiv_id": "1111.00001v1", "title": "Calorimeter GAN",
        "sections": [
            {"section_id": "sec_001", "level": 1, "heading": "Introduction",
             "text": "We simulate calorimeter showers with a generative adversarial network. "
                     "The GAN is trained on Geant4 showers [EQUATION_001]. It is fast."},
            {"section_id": "sec_002", "level": 1, "heading": "Acknowledgments",
             "text": "We thank the funding agency for support."},
        ],
        "equations": [{"equation_id": "eq_001", "section_id": "sec_001", "raw_latex": "E=mc^2",
                       "context_before": "", "context_after": ""}],
    },
    {
        "paper_id": "2222_00002v1", "arxiv_id": "2222.00002v1", "title": "Normalizing flows for jets",
        "sections": [
            {"section_id": "sec_001", "level": 1, "heading": "Method",
             "text": "Normalizing flows model the jet energy distribution. "
                     "The flow is invertible and gives exact likelihoods."},
        ],
        "equations": [],
    },
]


@pytest.fixture
def parsed_dir(tmp_path: Path) -> Path:
    """A parse run with two tiny papers, written without network access."""
    run = tmp_path / "parsed"
    records = []
    for paper in PAPERS:
        record = {key: paper[key] for key in ("paper_id", "arxiv_id", "title")}
        record.update(status="success", sections_count=len(paper["sections"]),
                      equations_count=len(paper["equations"]), errors=[], warnings=[])
        write_json(run / "papers" / paper["paper_id"] / "sections.json", paper["sections"])
        write_json(run / "papers" / paper["paper_id"] / "equations.json", paper["equations"])
        records.append(record)
    write_jsonl(run / "papers.jsonl", records)
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(p["arxiv_id"] for p in PAPERS) + "\n")
    write_manifest(run, step="parse", params={}, inputs={"ids_file": ids_file}, summary={})
    return run


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "questions.jsonl"
    write_jsonl(path, [
        {"question_id": "q1", "question": "Which generative adversarial network simulates calorimeter showers?",
         "expected_behavior": "answerable", "expected_arxiv_ids": ["1111.00001v1"],
         "reference_answer": "A GAN.", "required_answer_points": ["Name the GAN."],
         "reference_evidence": [{"arxiv_id": "1111.00001v1", "section_heading": "Introduction",
                                 "quote": "We simulate calorimeter showers"}]},
        {"question_id": "q2", "question": "In arXiv:2222.00002v1, why are flows invertible?",
         "expected_behavior": "answerable", "expected_arxiv_ids": ["2222.00002v1"]},
        {"question_id": "q3", "question": "What does arXiv:9999.99999 report?",
         "expected_behavior": "insufficient_evidence"},
    ])
    return path
