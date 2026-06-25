"""Tiny integrated test: fake model -> extraction JSON -> JSONL."""

from __future__ import annotations

from kamiknows.extraction.abstract_to_json import abstract_to_json
from kamiknows.models.base import ModelPlugin
from kamiknows.storage.jsonl import append_extraction_jsonl, read_jsonl_records


class FakeExtractionModel(ModelPlugin):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return '''{
          "title": "Fast calorimeter simulation for HEP",
          "field": "High Energy Physics / detector simulation",
          "main_claim": "A parameterized method can reduce simulation time within validated regions.",
          "method": "Parameterized shower shapes calibrated on reference samples.",
          "limitations": "Requires further validation on full detector geometries.",
          "confidence": "medium"
        }'''


def test_fake_model_extraction_can_be_saved_to_jsonl(tmp_path) -> None:
    extraction = abstract_to_json(
        model=FakeExtractionModel(),
        title="Fast calorimeter simulation for HEP",
        abstract="We present a lightweight method for fast detector simulation.",
    )
    output_path = tmp_path / "outputs" / "extractions.jsonl"

    append_extraction_jsonl(extraction, output_path)
    records = read_jsonl_records(output_path)

    assert len(records) == 1
    assert records[0]["main_claim"].startswith("A parameterized method")
