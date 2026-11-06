"""KamiPlotExtractor Task 01: local LaTeX figure inventory, without digitization."""

from .inventory import FigureExtractor, PanelDetector
from .pipeline import run_inventory
from .records import SCHEMA_VERSION, FigureRecord, PanelRecord

__all__ = ["SCHEMA_VERSION", "FigureExtractor", "PanelDetector", "FigureRecord", "PanelRecord", "run_inventory"]
