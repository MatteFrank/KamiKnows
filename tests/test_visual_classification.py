"""Offline visual pilot checks: no Ollama server or models."""

import json
from pathlib import Path

import pytest

from kamiknows.plot_extractor.classification import (
    render_sheet,
    validate_prediction,
    run_visual_classification,
)


def prediction():
    return {
        "analysis_type": "histogram_2d",
        "composition": "single_panel",
        "regions": [
            {
                "asset_number": 1,
                "description": "Cells with intensity encoding",
                "analysis_type": "histogram_2d",
                "axes": {
                    name: {"presence": "present", "numeric_ticks": "readable", "unit_label": "absent"}
                    for name in ("x", "y", "colorbar")
                },
            }
        ],
        "reason": "Binned grid with colour scale",
        "uncertainties": [],
    }


def wire_axes():
    return {
        name: {**state, "numeric_labels": "present"}
        for name, state in prediction()["regions"][0]["axes"].items()
    }


def test_visual_predictions_validate_asset_links():
    assert validate_prediction(prediction(), 1)["analysis_type"] == "histogram_2d"
    value = prediction()
    value["regions"][0]["asset_number"] = 2
    with pytest.raises(ValueError):
        validate_prediction(value, 1)
    value = prediction()
    value["analysis_type"] = "plot"
    with pytest.raises(ValueError):
        validate_prediction(value, 1)


def test_contact_sheet_preserves_original(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    image = tmp_path / "original.png"
    Image.new("RGB", (100, 80), "blue").save(image)
    original = image.read_bytes()
    output = render_sheet([image], tmp_path)
    assert output.exists() and image.read_bytes() == original


def test_visual_pilot_offline_provenance_and_errors(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    from kamiknows.common import write_manifest, write_json, write_jsonl
    from kamiknows.plot_extractor import run_inventory, run_classification, run_recovery
    from kamiknows import parsing

    parsed = tmp_path / "parsed"
    root = parsed / "papers" / "p1" / "source"
    root.mkdir(parents=True)
    tex = r"\section{Results}\begin{figure}\includegraphics{a.png}\caption{Histogram}\end{figure}"
    Image.new("RGB", (100, 80), "blue").save(root / "a.png")
    (root / "main.tex").write_text(tex)
    paper = root.parent
    (paper / "flat.tex").write_text(tex)
    sections, _, _ = parsing.parse_latex(tex)
    write_json(paper / "sections.json", sections)
    write_jsonl(parsed / "papers.jsonl", [{"paper_id": "p1", "status": "success"}])
    write_manifest(parsed, step="parse", params={}, inputs={}, summary={})
    inv = tmp_path / "inventory"
    run_inventory(parsed_dir=parsed, output_dir=inv)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    calls = []

    def post(url, json, timeout):
        calls.append((url, json))
        r = Response()
        r.value = (
            {"capabilities": ["vision"]}
            if url.endswith("/show")
            else {
                "response": __import__("json").dumps(
                    {
                        "asset_role": "plot",
                        "analysis_type": "histogram_2d",
                        "composition": "single_panel",
                        "axis_breaks": {"x": "absent", "y": "absent"},
                        "axes": wire_axes(),
                        "reason": "Binned grid with colour scale",
                    }
                ),
                "done_reason": "stop",
            }
        )
        return r

    monkeypatch.setattr("kamiknows.plot_extractor.classification.requests.post", post)
    args = dict(
        inventory_dir=inv,
        parsed_dir=parsed,
        figure_ids=["p1::figure::0001"],
        model="fake:vision",
        base_url="http://test",
    )
    out = tmp_path / "visual"
    summary = run_classification(output_dir=out, classifier="vision", **args)
    assert summary["status_counts"] == {"predicted": 1}
    assert calls[1][1]["images"] and calls[1][1]["options"]["temperature"] == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert any("asset" in role for role in manifest["inputs"])
    assert manifest["params"]["mode"] == "classify"
    assert manifest["params"]["classifier"] == "vision"
    figure = json.loads((out / "figures.jsonl").read_text())
    assert figure["analysis_type"] == "histogram_2d" and figure["figure_type"] == "plot"
    recovered = run_recovery(classified_dir=out, parsed_dir=parsed, output_dir=tmp_path / "recovered")
    assert recovered["paper_results"][0]["status"] == "sources_verified"

    from kamiknows.common import sha256_file

    prepared = tmp_path / "prepared"
    run_classification(
        inventory_dir=inv, parsed_dir=parsed, output_dir=prepared, classifier="vision", prepare_only=True
    )
    item = json.loads((prepared / "images.jsonl").read_text())
    sheet = prepared / item["assets"][0]["image"]
    monkeypatch.setattr(
        "kamiknows.plot_extractor.classification.render_sheet",
        lambda *args: pytest.fail("must reuse prepared images"),
    )
    summary = run_visual_classification(output_dir=tmp_path / "reused", prepared_images_dir=prepared, **args)
    assert summary["status_counts"] == {"predicted": 1}
    sheet.write_bytes(b"changed")
    summary = run_visual_classification(
        output_dir=tmp_path / "invalid_prepared", prepared_images_dir=prepared, **args
    )
    assert summary["status_counts"] == {"error": 1}
    (root / "a.png").write_bytes(b"changed")
    summary = run_visual_classification(output_dir=tmp_path / "bad", **args)
    assert summary["status_counts"] == {"error": 1}
    assert summary["prediction_type_counts"] == {}
    assert summary["analysis_type_counts"] == {"unknown": 1}


def test_pdf_rendering_needs_no_system_pdf_command(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "source.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=100, height=80)
        page.draw_line((10, 10), (90, 70))
        document.save(path)
    original = path.read_bytes()
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("no system renderer"))
    sheet = render_sheet([path], tmp_path)
    assert sheet.is_file() and path.read_bytes() == original


def test_axes_requirements_do_not_invent_units_or_change_class():
    value = prediction()
    value["regions"][0]["axes"]["x"]["numeric_ticks"] = "absent"
    result = validate_prediction(value, 1)
    assert result["analysis_type"] == "histogram_2d"
    requirements = result["regions"][0]["calibration_requirements"]
    assert requirements["status"] == "needs_review" and requirements["missing"] == ["x.numeric_ticks"]
    assert (
        validate_prediction(prediction(), 1)["regions"][0]["calibration_requirements"]["status"]
        == "visually_supported"
    )
    value = prediction()
    del value["regions"][0]["axes"]
    with pytest.raises(ValueError, match="axes"):
        validate_prediction(value, 1)
    value = prediction()
    value["regions"][0]["axes"]["x"]["presence"] = "absent"
    with pytest.raises(ValueError, match="contradicts"):
        validate_prediction(value, 1)


def test_mixed_regions_and_asset_coverage_require_review():
    value = prediction()
    other = __import__("copy").deepcopy(value["regions"][0])
    other["analysis_type"] = "diagram"
    value["regions"].append(other)
    with pytest.raises(ValueError, match="class conflicts"):
        validate_prediction(value, 1)
    value["analysis_type"] = "mixed"
    result = validate_prediction(value, 1)
    assert result["regions"][1]["calibration_requirements"]["status"] == "excluded"
    with pytest.raises(ValueError, match="cover all assets"):
        validate_prediction(prediction(), 2)


def test_categorical_bar_axes_do_not_require_numeric_x_ticks():
    value = prediction()
    value["analysis_type"] = "bar_chart"
    region = value["regions"][0]
    region["analysis_type"] = "bar_chart"
    region["axes"]["x"]["numeric_ticks"] = "absent"
    region["axes"]["colorbar"] = {"presence": "absent", "numeric_ticks": "absent", "unit_label": "absent"}
    assert (
        validate_prediction(value, 1)["regions"][0]["calibration_requirements"]["status"]
        == "visually_supported"
    )


@pytest.mark.parametrize("description_outcome", [None, "success", "failure"])
def test_asset_requests_cache_and_downstream_classes_are_independent(
    tmp_path, monkeypatch, description_outcome
):
    from PIL import Image
    from kamiknows import parsing
    from kamiknows.common import write_manifest, write_json, write_jsonl
    from kamiknows.plot_extractor import run_inventory, run_classification
    from kamiknows.plot_extractor.calibration import run_calibration

    parsed = tmp_path / "parsed"
    root = parsed / "papers" / "p" / "source"
    root.mkdir(parents=True)
    Image.new("RGB", (50, 50), "blue").save(root / "a.png")
    (root / "b.png").write_bytes((root / "a.png").read_bytes())
    Image.new("RGB", (50, 50), "red").save(root / "c.png")
    tex = r"\section{Results}\begin{figure}\includegraphics{a.png}\includegraphics{b.png}\includegraphics{c.png}\caption{Results}\end{figure}"
    (root / "main.tex").write_text(tex)
    (root.parent / "flat.tex").write_text(tex)
    sections, _, _ = parsing.parse_latex(tex)
    write_json(root.parent / "sections.json", sections)
    write_jsonl(parsed / "papers.jsonl", [{"paper_id": "p", "status": "success"}])
    write_manifest(parsed, step="parse", params={}, inputs={}, summary={})
    inv = tmp_path / "inventory"
    run_inventory(parsed_dir=parsed, output_dir=inv)
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    def post(url, json, timeout):
        response = Response()
        if url.endswith("/show"):
            response.value = {"capabilities": ["vision"]}
        else:
            calls.append(json)
            assert len(json["images"]) == 1 and "Caption" not in json["prompt"]
            response.value = {
                "response": __import__("json").dumps(
                    {
                        "asset_role": "plot" if len(calls) == 1 else "diagram",
                        "analysis_type": "histogram_2d" if len(calls) == 1 else "diagram",
                        "composition": "single_panel",
                        "axis_breaks": {"x": "absent", "y": "absent"},
                        "axes": wire_axes(),
                        "reason": "Visible evidence",
                    }
                ),
                "done_reason": "stop",
            }
        return response

    monkeypatch.setattr("kamiknows.plot_extractor.classification.requests.post", post)

    def describe(*args):
        if description_outcome == "failure":
            raise ValueError("description failed")
        return {
            "content": {
                "visible_content": "Boxes connected by arrows.",
                "visible_labels": [],
                "limitations": [],
            },
            "source_kind": "model_visual_description",
            "review_status": "pending",
            "retrieval_eligible": False,
        }

    monkeypatch.setattr("kamiknows.plot_extractor.classification.describe_visual_asset", describe)
    args = dict(
        inventory_dir=inv,
        parsed_dir=parsed,
        classifier="vision",
        model="fake:vision",
        base_url="http://test",
        describe_skipped=description_outcome is not None,
    )
    out = tmp_path / "visual"
    summary = run_classification(output_dir=out, **args)
    assert len(calls) == 2 and summary["cache_hits"] == 1 and summary["inference_requests"] == 2
    assert calls[0]["prompt"] == (out / "prompt.txt").read_text()
    assert json.loads((out / "manifest.json").read_text())["params"]["prompt_version"] == "vision_review"
    assert list(calls[0]["format"]["properties"])[0] != "reason"
    assert summary["descriptions"] == (1 if description_outcome == "success" else 0)
    assert summary["description_errors"] == (1 if description_outcome == "failure" else 0)
    descriptions = [json.loads(l) for l in (out / "visual_descriptions.jsonl").read_text().splitlines()]
    if descriptions:
        assert descriptions[0]["paper_id"] == "p" and descriptions[0]["asset_id"].endswith("03")
        assert descriptions[0]["retrieval_eligible"] is False
    assets = [json.loads(l) for l in (out / "asset_predictions.jsonl").read_text().splitlines()]
    assert assets[1]["cache_hit"] and assets[1]["prediction"]["regions"][0]["asset_number"] == 2
    figure = json.loads((out / "figures.jsonl").read_text())
    assert figure["analysis_type"] == "mixed"
    cal = run_calibration(classified_dir=out, parsed_dir=parsed, output_dir=tmp_path / "cal")
    # PNG assets have no native text: automatic calibration runs OCR and leaves candidates unverified.
    assert cal["status_counts"] == {"needs_review_ocr": 3}
    text = tmp_path / "text"
    run_classification(inventory_dir=inv, output_dir=text)
    sidecar = run_calibration(
        classified_dir=text,
        parsed_dir=parsed,
        output_dir=tmp_path / "sidecar_cal",
        visual_predictions=out / "visual_predictions.jsonl",
    )
    assert sidecar["status_counts"] == cal["status_counts"]
    # A broken third asset must not erase the first two valid asset predictions.
    (root / "c.png").write_bytes(b"changed")
    calls.clear()
    out2 = tmp_path / "partial"
    summary = run_classification(output_dir=out2, **args)
    assert summary["status_counts"] == {"partial": 1} and summary["asset_status_counts"] == {
        "predicted": 2,
        "error": 1,
    }
    figure = json.loads((out2 / "figures.jsonl").read_text())
    assert figure["analysis_type"] == "unknown"
    assert len(figure["classification_evidence"]["asset_predictions"]) == 3


def test_asset_presence_is_distinct_from_transcription_and_legend_role():
    from kamiknows.plot_extractor.classification import validate_asset_prediction
    from kamiknows.plot_extractor.calibration import visual_numeric_gate

    value = {
        "asset_role": "plot",
        "analysis_type": "histogram_2d",
        "composition": "single_panel",
        "axis_breaks": {"x": "absent", "y": "absent"},
        "axes": wire_axes(),
        "reason": "Numeric labels visible",
    }
    value["axes"]["x"]["numeric_ticks"] = "unclear"
    result = validate_asset_prediction(value)
    assert result["regions"][0]["axes"]["x"]["numeric_labels"] == "present"
    assert visual_numeric_gate(result["regions"], "histogram_2d") == (None, [])
    value = {
        "asset_role": "legend",
        "analysis_type": "unknown",
        "composition": "single_panel",
        "axis_breaks": {"x": "absent", "y": "absent"},
        "reason": "Only labelled color swatches",
        "axes": {
            a: {
                "presence": "absent",
                "numeric_ticks": "absent",
                "numeric_labels": "absent",
                "unit_label": "absent",
            }
            for a in ("x", "y", "colorbar")
        },
    }
    assert validate_asset_prediction(value)["asset_role"] == "legend"
    value["analysis_type"] = "histogram_1d"
    with pytest.raises(ValueError, match="legend"):
        validate_asset_prediction(value)


@pytest.mark.parametrize("valid", [True, False])
def test_visual_description_is_unreviewed_and_bounded(tmp_path, monkeypatch, valid):
    from kamiknows.plot_extractor.classification import describe_visual_asset

    sheet = tmp_path / "image.png"
    sheet.write_bytes(b"fake image")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": json.dumps(
                    {
                        "visible_content": "Boxes connected by arrows." if valid else "word " * 61,
                        "visible_labels": ["Input", "Output"],
                        "limitations": ["Small labels are unreadable."],
                    }
                ),
                "done_reason": "stop",
            }

    monkeypatch.setattr("kamiknows.plot_extractor.classification.requests.post", lambda *a, **k: Response())
    if not valid:
        with pytest.raises(ValueError):
            describe_visual_asset(sheet, "fake", "http://test")
    else:
        result = describe_visual_asset(sheet, "fake", "http://test")
        assert result["review_status"] == "pending" and result["retrieval_eligible"] is False
        assert result["source_kind"] == "model_visual_description"


def test_axis_break_observations_are_explicit_and_preserve_plot_class():
    from kamiknows.plot_extractor.classification import validate_asset_prediction, asset_prediction_schema

    value = {
        "asset_role": "plot",
        "analysis_type": "line_plot",
        "composition": "multi_panel",
        "axes": wire_axes(),
        "axis_breaks": {"x": "absent", "y": "present"},
        "reason": "Two Y intervals with visible break marks",
    }
    result = validate_asset_prediction(value)
    assert result["analysis_type"] == "line_plot" and result["axis_breaks"]["y"] == "present"
    assert "axis_breaks" in asset_prediction_schema()["required"]
    del value["axis_breaks"]["x"]
    with pytest.raises(ValueError, match="axis-break"):
        validate_asset_prediction(value)
