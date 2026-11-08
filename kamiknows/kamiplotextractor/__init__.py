"""Figure inventory, metadata classification and explicit numerical source recovery."""

from .inventory import FigureClassifier, FigureExtractor, PanelDetector
from .pipeline import run_classification, run_inventory, run_recovery
from .records import (CLASSIFIED_SCHEMA_VERSION, NUMERICAL_SCHEMA_VERSION, SCHEMA_VERSION,
                      FigureRecord, PanelRecord, PlotDataRecord)

__all__ = ["SCHEMA_VERSION", "CLASSIFIED_SCHEMA_VERSION", "FigureExtractor", "PanelDetector", "FigureClassifier",
           "NUMERICAL_SCHEMA_VERSION", "FigureRecord", "PanelRecord", "PlotDataRecord",
           "run_inventory", "run_classification", "run_recovery"]
