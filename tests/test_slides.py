"""The slide decks build offline from repository files and every text box fits."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_slides", ROOT / "docs/slides/build_slides.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("deck,pages", [("overview", 15), ("plot_extraction", 17)])
def test_slide_decks_build_and_never_name_local_runs(tmp_path, deck, pages):
    pymupdf = pytest.importorskip("pymupdf")
    builder = load_builder()
    path = tmp_path / f"{deck}.pdf"
    assert getattr(builder, deck)().save(path) == pages
    with pymupdf.open(path) as document:
        text = "\n".join(page.get_text() for page in document)
    assert "extract_plots.py calibrate" in text
    # Versioned slides describe configurations, not local run directories or their purpose.
    assert "talk" not in text.lower() and "layout_evidence" not in text and "plots_v2" not in text
