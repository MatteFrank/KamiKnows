"""Offline checks for the public commands, scheduler portability and slide source."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_plot_help_is_scoped_and_experimental_flags_are_removed():
    script = ROOT / "scripts/extract_plots.py"
    for command in [
        "inventory",
        "classify",
        "calibrate",
        "digitize",
        "recover",
        "benchmark",
        "calibration-review",
        "digitization-review",
    ]:
        result = subprocess.run(
            [sys.executable, str(script), command, "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        if command != "digitize":
            assert "--raster-dpi" not in result.stdout
        assert "--text-method" not in result.stdout and "--vision-prompt" not in result.stdout
    result = subprocess.run(
        [sys.executable, str(script), "calibrate", "--help"], capture_output=True, text=True
    )
    assert "--model" not in result.stdout and "--human-review-file" not in result.stdout


def test_scheduler_uses_checkout_paths_and_explicit_environment(tmp_path):
    script = ROOT / "scripts/run_ollama.slurm"
    assert "/home/" not in script.read_text()
    subprocess.run(["bash", "-n", str(script)], check=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, source in {
        "ollama": '#!/bin/bash\nif [ "$1" = serve ]; then exec sleep 5; fi\n',
        "curl": "#!/bin/bash\nexit 0\n",
        "python": '#!/bin/bash\nprevious=""\nfor arg in "$@"; do\nif [ "$previous" = --output-dir ]; then mkdir -p "$arg"; fi\nprevious="$arg"\ndone\necho "fake generation complete"\n',
    }.items():
        p = binaries / name
        p.write_text(source)
        p.chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "OLLAMA_BIN": str(binaries / "ollama"),
        "OLLAMA_MODELS": str(models),
        "PYTHON": str(binaries / "python"),
        "SLURM_JOB_ID": "43",
        "SLURM_JOB_NAME": "test",
        "SLURM_SUBMIT_DIR": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(script), "scripts/generate.py", "--output-dir", "outputs/test"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs/test/ollama.log").exists()
    assert (tmp_path / "outputs/test/slurm.log").exists()


def test_plot_questions_use_human_language_and_keep_annotations_separate():
    path = ROOT / "data/questions/fastcalo_plots.jsonl"
    plots = [json.loads(line) for line in path.read_text().splitlines()]
    from kamiknows.retrieval import extract_arxiv_ids

    assert len(plots) == 12 and {q["question_type"] for q in plots} == {"plot_data"}
    # Annotations name paper assets, never local run directories (outputs/ is not versioned).
    assert "plots_fastcalo" not in path.read_text() and "outputs/" not in path.read_text()
    assert all("::" not in q["question"] and "path_" not in q["question"] for q in plots)
    natural = [q for q in plots if q["question_style"] == "natural"]
    assert len(natural) == 6
    assert all(
        not extract_arxiv_ids(q["question"]) and "figure" not in q["question"].lower() for q in natural
    )
    assert all(q["expected_plot_sources"] for q in plots)
