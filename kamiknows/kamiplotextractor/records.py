"""Versioned figure metadata and numerical records with explicit provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"
CLASSIFIED_SCHEMA_VERSION = "1.1"
NUMERICAL_SCHEMA_VERSION = "1.2"
DIGITIZED_SCHEMA_VERSION = "1.3"


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
    classification_evidence: dict[str, Any] = field(default_factory=dict)
    classification_review_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if self.schema_version == SCHEMA_VERSION:
            for key in ("classification_evidence", "classification_review_status"):
                record.pop(key)
        return record


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
    classification_confidence: float | None = None
    classification_method: str = "not_classified"
    classification_evidence: dict[str, Any] = field(default_factory=dict)
    classification_review_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if self.schema_version == SCHEMA_VERSION:
            for key in ("classification_confidence", "classification_method",
                        "classification_evidence", "classification_review_status"):
                record.pop(key)
        return record


@dataclass
class PlotDataRecord:
    paper_id: str | None
    figure_id: str | None
    panel_id: str | None
    data_id: str
    source_method: str
    association: dict[str, Any]
    classification: dict[str, Any]
    provenance: dict[str, Any]
    extraction_status: str = "unavailable"
    backend: str = "native"
    backend_version: str = "1"
    axes: dict[str, Any] = field(default_factory=dict)
    series: list[dict[str, Any]] = field(default_factory=list)
    raw_source: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    validation_status: str = "not_validated"
    review_status: str = "pending"
    schema_version: str = NUMERICAL_SCHEMA_VERSION
    calibration: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if self.schema_version == NUMERICAL_SCHEMA_VERSION:
            record.pop("calibration")
            record.pop("quality")
        return record
