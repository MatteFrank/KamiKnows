"""Named series available to a computation, and why any of them is missing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

from kamiknows.compute.errors import NotComputable
from kamiknows.compute.types import Series


@dataclass(frozen=True)
class Entry:
    """A handle with either its series or the reason there is none."""

    handle: str
    series: Series | None
    reason: str | None = None
    source: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.series is None) == (self.reason is None):
            raise ValueError(f"{self.handle}: an entry has exactly one of series and reason")


class Catalog:
    """Read-only lookup of entries by handle."""

    def __init__(self, entries: Iterable[Entry]) -> None:
        self._entries: dict[str, Entry] = {}
        for entry in entries:
            if entry.handle in self._entries:
                raise ValueError(f"duplicate handle {entry.handle!r}")
            self._entries[entry.handle] = entry

    def get(self, handle: str) -> Series:
        """Return the series, or raise NotComputable saying why it is unavailable."""
        entry = self._entries.get(handle)
        if entry is None:
            raise NotComputable(f"no extracted data named {handle!r}")
        if entry.series is None:
            raise NotComputable(f"{handle}: {entry.reason}")
        return entry.series

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, handle: object) -> bool:
        return handle in self._entries
