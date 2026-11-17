"""Small layout helpers shared by the KamiKnows slide decks.

Slides are 960x540 pt PDF pages drawn with PyMuPDF. Text goes through HTML
boxes so that wrapping, bold and inline code work; every box must fit, so a
too-long text fails the build instead of being silently clipped.
"""

from __future__ import annotations

import html
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 960, 540
LEFT, RIGHT = 43, 917

INK = "#15212c"
MUTED = "#586775"
ACCENT = "#b86a12"
ACCENT_DARK = "#7a4510"
CARD = "#f8ede0"
CODE_BG = "#f7f4ef"
BORDER = "#e6ddd0"

CSS = f"""
@font-face {{font-family: Cabin; src: url(fonts/Cabin-Regular.ttf);}}
@font-face {{font-family: Cabin; font-weight: bold; src: url(fonts/Cabin-Bold.ttf);}}
@font-face {{font-family: Jost; src: url(fonts/Jost-Medium.ttf);}}
@font-face {{font-family: Jost; font-weight: bold; src: url(fonts/Jost-Bold.ttf);}}
@font-face {{font-family: Mono; src: url(fonts/DejaVuSansMono.ttf);}}
* {{margin: 0; padding: 0;}}
body {{font-family: Cabin; font-size: 12.5pt; color: {INK}; line-height: 1.32;}}
p {{margin-bottom: 6pt;}}
b {{font-weight: bold;}}
code {{font-family: Mono; font-size: 0.86em; color: {ACCENT_DARK};}}
.pre {{font-family: Mono; font-size: 9.6pt; color: {INK}; line-height: 1.38; white-space: pre-wrap;}}
.pre .c {{color: {MUTED};}}
ul {{margin-left: 0; padding-left: 0;}}
li {{margin-left: 12pt; margin-bottom: 5pt;}}
.muted {{color: {MUTED};}}
.small {{font-size: 11pt;}}
.tiny {{font-size: 9.5pt;}}
.label {{font-weight: bold; font-size: 10.5pt; color: {ACCENT};}}
.h {{font-family: Jost; font-size: 15pt; color: {ACCENT}; margin-bottom: 5pt;}}
.th {{font-family: Jost; font-size: 11pt; color: {ACCENT};}}
.td {{font-size: 10.5pt; line-height: 1.25;}}
"""


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def code_html(text: str) -> str:
    """Monospace block; lines starting with '#' are rendered as comments."""
    lines = []
    for line in text.strip("\n").splitlines():
        escaped = esc(line)
        lines.append(f'<span class="c">{escaped}</span>' if line.lstrip().startswith("#") else escaped)
    return '<div class="pre">' + "\n".join(lines) + "</div>"


def _rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))


class Slide:
    def __init__(self, deck: "Deck", page: pymupdf.Page):
        self.deck, self.page = deck, page

    def html(self, rect, content: str, css: str = "") -> None:
        # The bottom margin of a last paragraph is empty space: allow it to hang below the box.
        r = pymupdf.Rect(rect)
        spare, _ = self.page.insert_htmlbox(
            pymupdf.Rect(r.x0, r.y0, r.x1, r.y1 + 8), content, css=CSS + css, archive=self.deck.archive,
            scale_low=1,
        )
        if spare < 0:  # PyMuPDF reports -1 when the content needs more room than the rectangle
            r = pymupdf.Rect(rect)
            needed = self._height(r.width, content, css)
            raise ValueError(f"slide {self.page.number + 1}: text needs {needed:.0f} pt, box {r} has "
                             f"{r.height:.0f}: {content[:60]!r}")

    def box(self, rect, fill: str = CARD, border: str | None = None, radius: float = 0.035) -> None:
        shape = self.page.new_shape()
        shape.draw_rect(pymupdf.Rect(rect), radius=radius or None)
        shape.finish(fill=_rgb(fill), color=_rgb(border) if border else None, width=0.7)
        shape.commit()

    def line(self, start, end, color: str = ACCENT, width: float = 1.2) -> None:
        self.page.draw_line(start, end, color=_rgb(color), width=width)

    def arrow(self, start, end, color: str = ACCENT) -> None:
        """Straight arrow with a small filled head at ``end``."""
        (x0, y0), (x1, y1) = start, end
        length = max(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 1e-6)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        size = 5
        self.line(start, (x1 - size * ux, y1 - size * uy), color)
        left = (x1 - size * ux + size * 0.6 * uy, y1 - size * uy - size * 0.6 * ux)
        right = (x1 - size * ux - size * 0.6 * uy, y1 - size * uy + size * 0.6 * ux)
        shape = self.page.new_shape()
        shape.draw_polyline([end, left, right, end])
        shape.finish(fill=_rgb(color), color=_rgb(color), closePath=True)
        shape.commit()

    def label(self, x0: float, y0: float, x1: float, text: str) -> None:
        self.html((x0, y0, x1, y0 + 22), f'<div class="label">{esc(text)}</div>')

    def code(self, rect, text: str, label: str | None = None, size: float | None = None) -> float:
        """Code block that shrinks to its content within ``rect``; returns its bottom y."""
        x0, y0, x1, y1 = rect
        if label:
            self.label(x0, y0, x1, label)
            y0 += 20
        css = f".pre {{font-size: {size}pt;}}" if size else ""
        content = code_html(text)
        y1 = min(y1, y0 + self._height(x1 - x0 - 18, content, css) + 16)
        self.box((x0, y0, x1, y1), CODE_BG, BORDER, radius=0.04)
        self.html((x0 + 10, y0 + 8, x1 - 8, y1 - 4), content, css)
        return y1

    def card(self, rect, title: str, body: str, foot: str | None = None) -> None:
        x0, y0, x1, y1 = rect
        self.box(rect)
        self.html((x0 + 13, y0 + 11, x1 - 11, y1 - (26 if foot else 6)),
                  f'<div class="h">{esc(title)}</div><div class="small">{body}</div>')
        if foot:
            self.html((x0 + 13, y1 - 25, x1 - 8, y1 - 2),
                      f'<div class="pre muted" style="font-size:8.6pt">{esc(foot)}</div>')

    def _height(self, width: float, content: str, css: str = "") -> float:
        """Height some HTML needs at this width, measured on a scratch page."""
        scratch = pymupdf.open()
        page = scratch.new_page(width=width + 10, height=2000)
        spare, _ = page.insert_htmlbox(pymupdf.Rect(0, 0, width, 2000), content, css=CSS + css,
                                       archive=self.deck.archive, scale_low=1)
        return 2000 - spare

    def table(self, rect, header: list[str], rows: list[list[str]], widths: list[float]) -> float:
        """Grid with fixed column fractions; returns the y below the last row."""
        x0, y, x1, y1 = rect
        lefts = [x0]
        for fraction in widths[:-1]:
            lefts.append(lefts[-1] + fraction * (x1 - x0))
        rights = lefts[1:] + [x1]
        for index, row in enumerate([header, *rows]):
            cls = "th" if index == 0 else "td"
            cells = [f'<div class="{cls}">{c}</div>' for c in row]
            height = max(self._height(r - l - 8, c) for l, r, c in zip(lefts, rights, cells)) + 8
            if y + height > y1:
                raise ValueError(f"table does not fit on slide {self.page.number + 1}")
            for l, r, c in zip(lefts, rights, cells):
                self.html((l, y + 4, r - 8, y + height), c)
            y += height
            self.line((x0, y), (x1, y), ACCENT if index == 0 else BORDER, 0.9 if index == 0 else 0.7)
        return y

    def image(self, rect, path: Path) -> None:
        self.page.insert_image(pymupdf.Rect(rect), filename=str(path), keep_proportion=True)

    def note(self, text: str, y: float = 466) -> None:
        """One closing sentence in the muted style, full width."""
        self.html((LEFT, y, RIGHT, y + 36), f'<div class="muted">{text}</div>')


class Deck:
    def __init__(self, footer: str):
        self.doc = pymupdf.open()
        self.footer = footer
        self.archive = pymupdf.Archive(str(HERE))

    def cover(self, title_dark: str, title_accent: str, subtitle: str, line: str, authors: str,
              band: str, logo: Path) -> None:
        slide = Slide(self, self.doc.new_page(width=WIDTH, height=HEIGHT))
        slide.image((540, 30, 930, 450), logo)
        slide.html((LEFT, 175, 600, 265),
                   f'<div style="font-family:Jost;font-weight:bold;font-size:60pt;line-height:1;color:{INK}">'
                   f'{esc(title_dark)}<span style="color:{ACCENT_DARK}">{esc(title_accent)}</span></div>')
        slide.html((LEFT, 268, 600, 372),
                   f'<p style="font-size:17pt;color:{MUTED};line-height:1.3">{subtitle}</p>'
                   f'<div class="muted" style="font-size:11.5pt">{line}</div>')
        slide.html((LEFT, 378, 560, 410),
                   f'<div class="pre" style="font-size:10.5pt;color:{ACCENT_DARK}">{esc(authors)}</div>')
        slide.box((0, 473, WIDTH, HEIGHT), ACCENT, radius=0)
        slide.html((LEFT, 494, 600, 530), f'<div style="color:#ffffff;font-size:11.5pt">{esc(band)}</div>')

    def slide(self, title: str) -> Slide:
        slide = Slide(self, self.doc.new_page(width=WIDTH, height=HEIGHT))
        slide.html((LEFT, 48, RIGHT, 94),
                   f'<div style="font-family:Jost;font-size:29pt;color:{ACCENT};line-height:1">{esc(title)}</div>')
        slide.line((LEFT, 95), (RIGHT, 95), "#b6772f", 0.9)
        slide.html((LEFT, 504, 600, 538), f'<div class="muted small">{esc(self.footer)}</div>')
        slide.html((860, 506, RIGHT + 10, 538),
                   f'<div class="muted tiny" style="text-align:right">{self.doc.page_count}</div>')
        return slide

    def save(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.subset_fonts()
        self.doc.save(path, garbage=4, deflate=True)
        return self.doc.page_count
