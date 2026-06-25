"""Basic checks for tutorial documentation files."""

from __future__ import annotations

from pathlib import Path


def test_run_record_schema_doc_exists_and_mentions_core_blocks() -> None:
    path = Path("docs/run_record_schema.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows run record schema" in text
    assert "source" in text
    assert "extraction" in text
    assert "run" in text
    assert "prompt_version" in text
    assert "abstract_to_json_v0" in text


def test_model_mini_benchmark_doc_exists_and_mentions_separation() -> None:
    path = Path("docs/model_mini_benchmark.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows model mini benchmark" in text
    assert "metadata ingestion" in text
    assert "model interpretation" in text
    assert "download_arxiv_metadata_batch.py" in text
    assert "run_model_mini_benchmark.py" in text
    assert "formal" in text


def test_prompt_versioning_doc_exists_and_mentions_hash() -> None:
    path = Path("docs/prompt_versioning.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows prompt versioning" in text
    assert "prompt_template_sha256" in text
    assert "extraction_schema_version" in text
    assert "prompt_registry.py" in text


def test_dataset_manifest_doc_exists_and_mentions_hashes() -> None:
    path = Path("docs/dataset_manifest.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows dataset manifest" in text
    assert "dataset_manifest_v0" in text
    assert "sha256" in text
    assert "record_count" in text
    assert "create_dataset_manifest.py" in text

def test_manual_quality_checklist_doc_exists_and_mentions_fidelity() -> None:
    path = Path("docs/manual_quality_checklist.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows manual quality checklist" in text
    assert "scientific fidelity" in text
    assert "create_manual_quality_checklist.py" in text
    assert "main_claim" in text
    assert "method" in text
    assert "limitations" in text



def test_manual_quality_review_summary_doc_exists_and_mentions_outcomes() -> None:
    path = Path("docs/manual_quality_review_summary.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows manual quality review summary" in text
    assert "summarize_manual_quality_review.py" in text
    assert "pass" in text
    assert "revise" in text
    assert "reject" in text
    assert "unclear" in text

def test_quality_gate_doc_exists_and_mentions_decisions() -> None:
    path = Path("docs/quality_gate.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows quality gate" in text
    assert "run_quality_gate.py" in text
    assert "ACCEPT" in text
    assert "REVISE" in text
    assert "REJECT" in text
    assert "dataset_manifest.json" in text
    assert "manual_review_summary" in text




def test_hep_pilot_doc_exists_and_mentions_controlled_query() -> None:
    path = Path("docs/hep_pilot.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows HEP pilot" in text
    assert "run_hep_pilot.py" in text
    assert "cat:hep-ex AND calorimeter" in text
    assert "10-20" in text
    assert "manual quality checklist" in text


def test_post_pilot_analysis_doc_exists_and_mentions_recommendations() -> None:
    path = Path("docs/post_pilot_analysis.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows post-pilot analysis" in text
    assert "summarize_hep_pilot_run.py" in text
    assert "READY_FOR_NEXT_PILOT_CYCLE" in text
    assert "REVISE_BEFORE_SCALING" in text
    assert "quality_gate_report.json" in text


def test_phase0_closure_doc_exists_and_marks_completion() -> None:
    path = Path("docs/phase0_closure.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows Fase 0 closure" in text
    assert "Fase 0 is complete" in text
    assert "ModelPlugin" in text
    assert "Quality gate" in text or "quality gate" in text


def test_phase1_handoff_doc_exists_and_mentions_codex() -> None:
    path = Path("docs/phase1_handoff.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows Fase 1 handoff for Codex" in text
    assert "Codex" in text
    assert "run_hep_pilot.py" in text
    assert "ids-file" in text


def test_project_status_exists_and_mentions_phase_boundary() -> None:
    path = Path("PROJECT_STATUS.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows project status" in text
    assert "Fase 0 is complete" in text
    assert "Fase 1 next" in text


def test_changelog_exists_and_mentions_phase0_closure() -> None:
    path = Path("CHANGELOG.md")
    text = path.read_text(encoding="utf-8")

    assert "# KamiKnows changelog" in text
    assert "Fase 0 closure" in text
    assert "--ids-file" in text
