"""Local figure inventory and conservative metadata classification, without digitization."""

from .inventory import FigureClassifier, FigureExtractor, PanelDetector
from .pipeline import run_classification, run_inventory
from .records import CLASSIFIED_SCHEMA_VERSION, SCHEMA_VERSION, FigureRecord, PanelRecord

__all__ = ["SCHEMA_VERSION", "CLASSIFIED_SCHEMA_VERSION", "FigureExtractor", "PanelDetector", "FigureClassifier",
           "FigureRecord", "PanelRecord", "run_inventory", "run_classification"]
