"""Figure evidence: inventory, classification, calibration, digitization and recovery."""

from .calibration import run_calibration
from .inventory import FigureExtractor, PanelDetector
from .classification import FigureClassifier, run_classification, run_visual_preparation
from .pipeline import run_benchmark, run_digitization, run_inventory, run_recovery
from .records import (
    CLASSIFIED_SCHEMA_VERSION,
    NUMERICAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    FigureRecord,
    PanelRecord,
    PlotDataRecord,
)

__all__ = [
    "SCHEMA_VERSION",
    "CLASSIFIED_SCHEMA_VERSION",
    "FigureExtractor",
    "PanelDetector",
    "FigureClassifier",
    "NUMERICAL_SCHEMA_VERSION",
    "FigureRecord",
    "PanelRecord",
    "PlotDataRecord",
    "run_inventory",
    "run_calibration",
    "run_visual_preparation",
    "run_classification",
    "run_recovery",
    "run_digitization",
    "run_benchmark",
]
