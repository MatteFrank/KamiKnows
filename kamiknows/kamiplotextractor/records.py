"""Versioned inventory records; numerical extraction is deliberately absent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class FigureRecord:
    paper_id: str
    figure_id: str
    caption: str | None
    caption_raw: str | None
    latex_label: str | None
    section_id: str | None
    source_assets: list[dict[str, Any]]
    source_hashes: dict[str, str]
    panels: list[str]
    provenance: dict[str, Any]
    latex_labels: list[str] = field(default_factory=list)
    captions: list[dict[str, Any]] = field(default_factory=list)
    figure_type: str = "unknown"
    classification_confidence: float | None = None
    classification_method: str = "not_classified"
    warnings: list[dict[str, Any]] = field(default_factory=list)
    status: str = "success"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PanelRecord:
    panel_id: str
    figure_id: str
    paper_id: str
    panel_label: str | None
    source_region: dict[str, Any]
    source_asset_ids: list[str]
    caption: str | None
    caption_raw: str | None
    latex_labels: list[str]
    panel_type: str = "unknown"
    plot_type: str | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
