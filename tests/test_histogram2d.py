"""Known 2D cell colors verify calibration, Z inversion, abstention and automatic review."""

import json
from pathlib import Path

import pytest

from kamiknows.plot_extractor.calibration import propose_pdf, run_calibration
from kamiknows.plot_extractor.digitization import histogram2d_series
from kamiknows.plot_extractor.validation import structural_check, compare_series


def make_histogram(path, log=False, colorbar=True, zunit=True, colorbar_frame=True):
    pdf = pytest.importorskip("pymupdf")
    with pdf.open() as doc:
        page = doc.new_page(width=440, height=300)
        roi = [80, 50, 260, 230]
        bar = [285, 50, 300, 230]
        for rect, t in [
            ([80, 50, 170, 140], 0.25),
            ([170, 50, 260, 140], 0.75),
            ([80, 140, 170, 230], 0.5),
            ([170, 140, 260, 230], 0.1),
        ]:
            page.draw_rect(pdf.Rect(rect), color=None, fill=(1 - t, 0, t))
        if colorbar:
            for i in range(256):
                t = (i + 0.5) / 256
                page.draw_rect(
                    pdf.Rect(285, 50 + i / 256 * 180, 300, 50 + (i + 1) / 256 * 180),
                    color=None,
                    fill=(1 - t, 0, t),
                )
            if colorbar_frame:
                page.draw_rect(pdf.Rect(bar), color=(0, 0, 0))
        page.draw_rect(pdf.Rect(roi), color=(0, 0, 0))
        for x, value in zip([80, 170, 260], ["0", "1", "2"]):
            page.draw_line((x, 230), (x, 234), color=(0, 0, 0))
            page.insert_text((x - pdf.get_text_length(value, fontsize=10) / 2, 246), value, fontsize=10)
        for y, value in zip([50, 140, 230], ["2", "1", "0"]):
            page.draw_line((80, y), (76, y), color=(0, 0, 0))
            page.insert_text((69 - pdf.get_text_length(value, fontsize=10), y + 3), value, fontsize=10)
        if colorbar:
            for y, value in zip([50, 140, 230], ["1e2", "1e1", "1e0"] if log else ["10", "5", "0"]):
                page.draw_line((300, y), (304, y), color=(0, 0, 0))
                page.insert_text((308, y + 3), value, fontsize=10)
        page.insert_text((150, 275), "(m)", fontsize=10)
        page.insert_text((10, 130), "(s)", fontsize=10)
        if zunit:
            page.insert_text((350, 130), "(counts)", fontsize=10)
        doc.save(path)


@pytest.mark.parametrize("log", [False, True])
def test_histogram2d_axis_and_cells(tmp_path, log):
    path = tmp_path / "known.pdf"
    make_histogram(path, log=log)
    before = path.read_bytes()
    config, evidence = propose_pdf(path, "histogram_2d")
    assert config is not None, evidence
    panel = config["histogram2d_panels"][0]
    assert panel["x"]["scale"] == "linear" and panel["y"]["scale"] == "linear"
    assert panel["z"]["scale"] == ("log10" if log else "linear")
    series = histogram2d_series(path, config)[0]
    structural_check(series)
    assert len(series["z"]) == 4
    assert series["units"] == {"x": "m", "y": "s", "z": "counts"}
    assert series["cell_bounds"][0]["x"] == pytest.approx([0.0, 1.0])
    assert series["cell_bounds"][0]["y"] == pytest.approx([1.0, 2.0])
    expected = 10**1.5 if log else 7.5
    assert series["z"][0] == pytest.approx(expected, rel=0.02)
    assert series["z_ranges"][0][0] <= series["z"][0] <= series["z_ranges"][0][1]
    assert not series["original_bin_edges_verified"]
    assert compare_series(series, series, {})["status"] == "not_comparable"
    assert path.read_bytes() == before


@pytest.mark.parametrize("kwargs", [{"colorbar": False}])
def test_missing_required_2d_information(tmp_path, kwargs):
    path = tmp_path / "missing.pdf"
    make_histogram(path, **kwargs)
    config, evidence = propose_pdf(path, "histogram_2d")
    assert config is None and evidence["missing_requirements"]


@pytest.mark.parametrize("second_format", ["pdf", "png"])
@pytest.mark.parametrize("second_type", ["histogram_2d", "diagram", "image", "mixed"])
def test_gemma_routing_mismatch_section_and_automatic_html(tmp_path, second_type, second_format, monkeypatch):
    from kamiknows import parsing
    from kamiknows.common import write_manifest, write_json, write_jsonl
    from kamiknows.plot_extractor import run_inventory, run_classification, run_digitization

    monkeypatch.setattr(
        "kamiknows.plot_extractor.calibration.ocr_axis_candidates",
        lambda path: {
            "limitation": "Unverified OCR candidate.",
            "observations": [{"box": [1, 2, 3, 4], "text": "0.1", "score": 0.99, "verified": False}],
        },
    )
    monkeypatch.setattr("kamiknows.plot_extractor.calibration._ocr_model_inputs", lambda: {})
    parsed = tmp_path / "parsed"
    root = parsed / "papers" / "synthetic" / "source"
    root.mkdir(parents=True)
    make_histogram(root / "ok.pdf")
    make_histogram(root / "missing.pdf", colorbar=False)
    tex = r"\section{Results}\begin{figure}\includegraphics{ok.pdf}\caption{Results.}\end{figure}\begin{figure}\includegraphics{missing.pdf}\caption{Other results.}\end{figure}"
    if second_format == "png":
        from PIL import Image

        Image.new("RGB", (30, 30), "white").save(root / "missing.png")
        tex = tex.replace("missing.pdf", "missing.png")
    (root / "main.tex").write_text(tex)
    (root.parent / "flat.tex").write_text(tex)
    sections, _, _ = parsing.parse_latex(tex)
    write_json(root.parent / "sections.json", sections)
    write_jsonl(parsed / "papers.jsonl", [{"paper_id": "synthetic", "status": "success"}])
    write_manifest(parsed, step="parse", params={}, inputs={}, summary={})
    inv = tmp_path / "inv"
    classified = tmp_path / "classified"
    run_inventory(parsed_dir=parsed, output_dir=inv)
    run_classification(inventory_dir=inv, output_dir=classified)
    hints = tmp_path / "gemma.jsonl"
    write_jsonl(
        hints,
        [
            {
                "figure_id": f"synthetic::figure::{i:04d}",
                "prediction": {
                    "analysis_type": "histogram_2d" if i == 1 else second_type,
                    "regions": [
                        {
                            "asset_number": 1,
                            "analysis_type": "histogram_2d" if i == 1 else second_type,
                            "axes": {
                                a: {
                                    "presence": "present",
                                    "numeric_ticks": "readable",
                                    "unit_label": "absent",
                                }
                                for a in ("x", "y", "colorbar")
                            },
                        }
                    ],
                },
            }
            for i in [1, 2]
        ],
    )
    output = tmp_path / "calibration"
    summary = run_calibration(
        classified_dir=classified, parsed_dir=parsed, output_dir=output, visual_predictions=hints
    )
    second_status = (
        "needs_review_classification"
        if second_type == "mixed" and second_format == "png"
        else (
            "needs_raster_calibration"
            if second_format == "png"
            else "needs_review_axes" if second_type == "histogram_2d" else "proposed_class_conflict"
        )
    )
    if second_status == "needs_raster_calibration":
        second_status = "needs_review_ocr"
    assert summary["status_counts"] == {"proposed": 1, second_status: 1}
    page = (output / "review.html").read_text()
    assert 'id="hint-mismatches"' in page and "synthetic::figure::0002" in page
    assert "axis Z" in page
    if second_format == "png" and second_type != "mixed":
        assert "OCR observations — unverified" in page
        assert len(json.loads((output / "calibration_proposals.json").read_text())["plots"]) == 1
    assert ("Missing requirements" in page) == (
        second_type != "mixed" and (second_format == "png" or second_type == "histogram_2d")
    )
    assert json.loads((output / "manifest.json").read_text())["params"]["mode"] == "calibrate"
    result = run_digitization(
        classified_dir=classified,
        parsed_dir=parsed,
        output_dir=tmp_path / "digitized",
        backend="pymupdf",
        calibration_file=output / "calibration_proposals.json",
        figure_ids=None,
    )
    if second_type == "mixed" and second_format == "pdf":
        proposals = json.loads((output / "calibration_proposals.json").read_text())["plots"]
        assert proposals[1]["classification_review_required"] is True
    second_extraction = (
        ("needs_calibration" if second_format == "pdf" else "unsupported")
        if second_type == "histogram_2d"
        else "skipped_generic"
    )
    assert result["extraction_status_counts"] == {"success": 1, second_extraction: 1}


def test_embedded_uniform_grid_and_missing_color_inverse(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    from PIL import Image
    import io

    path = tmp_path / "vector.pdf"
    make_histogram(path)
    # Replace only the data rectangles with a uniform-block embedded image.
    with pdf.open(path) as doc:
        page = doc[0]
        page.add_redact_annot(pdf.Rect(80, 50, 260, 230), fill=None)
        page.apply_redactions(images=0, graphics=1, text=0)
        image = Image.new("RGB", (20, 20))
        colors = [(191, 0, 64), (64, 0, 191), (128, 0, 128), (230, 0, 25)]
        for y in range(20):
            for x in range(20):
                image.putpixel((x, y), colors[(y // 10) * 2 + x // 10])
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        page.insert_image(pdf.Rect(80, 50, 260, 230), stream=buffer.getvalue())
        raster = tmp_path / "embedded.pdf"
        doc.save(raster)
    # Explicit geometry is used here to isolate pixel-block extraction from frame discovery.
    config, _ = propose_pdf(path, "histogram_2d")
    series = histogram2d_series(raster, config)[0]
    assert len(series["z"]) == 4 and series["cell_source"]["kind"] == "rendered_raster_blocks"
    assert series["z"][0] == pytest.approx(7.5, abs=0.08)
    # A green cell is not represented on the red/blue colorbar and must not get a fabricated Z.
    with pdf.open(path) as doc:
        doc[0].draw_rect(pdf.Rect(80, 50, 170, 140), color=None, fill=(0, 1, 0))
        bad = tmp_path / "overlap.pdf"
        doc.save(bad)
    with pytest.raises(ValueError):
        histogram2d_series(bad, config)


@pytest.mark.parametrize("alpha", [0, 128, 255])
def test_raster_alpha_policy(tmp_path, alpha):
    import io
    from PIL import Image

    pdf = pytest.importorskip("pymupdf")
    original = tmp_path / "base.pdf"
    make_histogram(original)
    config, _ = propose_pdf(original, "histogram_2d")
    with pdf.open(original) as doc:
        page = doc[0]
        page.add_redact_annot(pdf.Rect(80, 50, 260, 230), fill=None)
        page.apply_redactions(images=0, graphics=1, text=0)
        image = Image.new("RGBA", (20, 20))
        colors = [(191, 0, 64), (64, 0, 191), (128, 0, 128), (230, 0, 25)]
        for y in range(20):
            for x in range(20):
                index = (y // 10) * 2 + x // 10
                image.putpixel((x, y), (*colors[index], alpha if index == 0 else 255))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        page.insert_image(pdf.Rect(80, 50, 260, 230), stream=stream.getvalue())
        path = tmp_path / "alpha.pdf"
        doc.save(path)
    if alpha == 128:
        with pytest.raises(ValueError, match="partially_transparent"):
            histogram2d_series(path, config)
    else:
        result = histogram2d_series(path, config)[0]
        assert len(result["z"]) == (3 if alpha == 0 else 4)
        assert result["cell_source"]["masked_blocks_omitted"] == (1 if alpha == 0 else 0)


def test_six_panels_with_shared_colorbar(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    original = tmp_path / "base.pdf"
    make_histogram(original, colorbar=False, zunit=False)
    path = tmp_path / "six.pdf"
    with pdf.open(original) as source, pdf.open() as doc:
        page = doc.new_page(width=1280, height=620)
        for y in (0, 300):
            for x in (0, 420, 840):
                page.show_pdf_page(pdf.Rect(x, y, x + 280, y + 300), source, 0, clip=pdf.Rect(0, 0, 280, 300))
        for i in range(256):
            t = (i + 0.5) / 256
            page.draw_rect(
                pdf.Rect(1125, 50 + i / 256 * 480, 1140, 50 + (i + 1) / 256 * 480),
                color=None,
                fill=(1 - t, 0, t),
            )
        page.draw_rect(pdf.Rect(1125, 50, 1140, 530), color=(0, 0, 0))
        for y, value in zip((50, 290, 530), ("10", "5", "0")):
            page.draw_line((1140, y), (1144, y), color=(0, 0, 0))
            page.insert_text((1148, y + 3), value, fontsize=10)
        page.insert_text((1190, 290), "(counts)", fontsize=10)
        doc.save(path)
    config, evidence = propose_pdf(path, "histogram_2d")
    assert config is not None, evidence
    assert len(config["histogram2d_panels"]) == 6
    alternative, details = propose_pdf(path, "scatter_plot")
    assert alternative is not None and details["calibration_type"] == "histogram_2d"
    assert details["suggested_type"] == "scatter_plot"
    unknown, unknown_details = propose_pdf(path, "unknown")
    assert unknown is not None and len(unknown["histogram2d_panels"]) == 6
    assert unknown_details["calibration_type"] == "histogram_2d"
    assert all(p["colorbar_shared"] for p in evidence["panels"])
    result = histogram2d_series(path, config)
    assert len(result) == 6 and all(len(s["z"]) == 4 for s in result)


def test_2d_axis_calibration_survives_unsupported_cell_geometry(tmp_path, monkeypatch):
    path = tmp_path / "known.pdf"
    make_histogram(path)

    def unsupported(*args):
        raise ValueError("unsupported grid geometry")

    monkeypatch.setattr("kamiknows.plot_extractor.digitization.pdf_cell_geometry", unsupported)
    config, evidence = propose_pdf(path, "histogram_2d")
    assert config is not None and config["digitization_readiness"] == "geometry_review_required"
    assert evidence["reason"] == "axes_calibrated_geometry_review_required"
    assert evidence["panels"][0]["geometry_requirements"] == ["unsupported grid geometry"]


def test_open_colorbar_is_verified_from_contiguous_palette_and_numeric_ticks(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    path = tmp_path / "open_bar.pdf"
    make_histogram(path, colorbar_frame=False)
    target = tmp_path / "open_bar_overprint.pdf"
    with pdf.open(path) as source, pdf.open() as document:
        page = document.new_page(width=440, height=300)
        # Repeated source layers also duplicate the numeric text.
        for _ in range(2):
            page.show_pdf_page(page.rect, source, 0)
        document.save(target)
    proposal, evidence = propose_pdf(target, "histogram_2d")
    assert proposal is not None, evidence
    from kamiknows.plot_extractor.calibration import _palette_strip_candidates, infer_axis, _axis_words

    with pdf.open(target) as document:
        page = document[0]
        paths = page.get_drawings()
        candidates = _palette_strip_candidates(paths)
        assert len(candidates) == 1
        config, evidence = infer_axis(paths, _axis_words(page), candidates[0], "z")
        assert config is not None, evidence
        assert config["scale"] == "linear"


def test_raster_ocr_colorbar_is_a_z_axis_not_an_extra_panel(tmp_path, monkeypatch):
    from kamiknows.plot_extractor import calibration

    pdf = pytest.importorskip("pymupdf")
    pytest.importorskip("cv2")
    source = tmp_path / "source.pdf"
    make_histogram(source)
    image = tmp_path / "plot.png"
    with pdf.open(source) as document:
        page = document[0]
        page.get_pixmap(matrix=pdf.Matrix(2, 2), alpha=False).save(image)
        words = page.get_text("words")
    observations = []
    for x0, y0, x1, y1, text, *rest in words:
        observations.append(
            ([[2 * x0, 2 * y0], [2 * x1, 2 * y0], [2 * x1, 2 * y1], [2 * x0, 2 * y1]], text, 0.99)
        )
    monkeypatch.setattr(
        calibration,
        "_ocr_engine",
        lambda: lambda pixels: (observations if pixels.shape[1] == 880 else [], None),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    result = calibration.ocr_axis_candidates(image)
    assert len(result["panels"]) == 1
    panel = result["panels"][0]
    assert panel["z"]["scale"] == "linear" and len(panel["z"]["anchors"]) == 3
    assert result["numeric_verified"] is False
