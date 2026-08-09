"""Turn source documents into positioned text blocks.

Extraction is routed per document family because the families encode meaning in
position differently, not because they differ in column count. The NATA position
statements are two-column but write their evidence categories inline, so they
need no special handling; the JOSPT guidelines are two-column and set their
grades as separate glyphs, so reading order alone loses the association between
a recommendation and the strength of evidence behind it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from ingest.models import Block

# Journal furniture that appears mid-page and would otherwise land between a
# recommendation and the text that continues it.
BOILERPLATE = re.compile(
    r"^(downloaded from|copyright ©|for personal use only|"
    r"journal of orthopaedic\s*&\s*sports physical therapy|"
    r"protected by copyright|br j sports med|cpg\d+\s*\||"
    r"a \.gov website belongs to|a lock \(|\d+\s*$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LayoutConfig:
    """How to read one family of documents."""

    columns: int
    grade_scale: str | None = None


LAYOUTS: dict[str, LayoutConfig] = {
    "Mass General Brigham": LayoutConfig(columns=1),
    "JOSPT / APTA Academy of Orthopaedic Physical Therapy": LayoutConfig(
        columns=2, grade_scale="JOSPT grade of recommendation"
    ),
    "NATA": LayoutConfig(columns=2, grade_scale="NATA evidence category"),
    "BJSM": LayoutConfig(columns=2),
    "CDC HEADS UP": LayoutConfig(columns=1),
}

DEFAULT_LAYOUT = LayoutConfig(columns=1)


def layout_for(org: str) -> LayoutConfig:
    """Select the reading strategy for a publisher."""
    return LAYOUTS.get(org, DEFAULT_LAYOUT)


def _column_of(x0: float, page_width: float, columns: int) -> int:
    """Assign a block to a column by its left edge."""
    if columns < 2:
        return 0
    return min(int(x0 / (page_width / columns)), columns - 1)


def _clean(text: str) -> str:
    """Collapse whitespace and rejoin words split by line-break hyphenation.

    The journal PDFs hyphenate with a soft hyphen (U+00AD) at the break. Spans
    are joined with a space, so the soft hyphen and the space that follows it
    must be removed together, otherwise "in-clude" becomes "in clude".
    """
    text = re.sub("­\\s*", "", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf(path: Path, layout: LayoutConfig) -> list[Block]:
    """Read a PDF into blocks ordered by page, then column, then vertical position.

    PyMuPDF groups a JOSPT grade glyph into the same block as the recommendation
    it qualifies, so sorting blocks within their column is enough to keep the two
    together. Sorting the raw block stream instead would not be.
    """
    blocks: list[Block] = []
    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc):
            width = page.rect.width
            page_blocks: list[Block] = []
            for raw in page.get_text("dict")["blocks"]:
                if raw.get("type") != 0:
                    continue
                text = _clean(
                    " ".join(span["text"] for line in raw["lines"] for span in line["spans"])
                )
                if not text or BOILERPLATE.match(text):
                    continue
                x0, y0 = raw["bbox"][0], raw["bbox"][1]
                page_blocks.append(
                    Block(
                        text=text,
                        page=page_index + 1,
                        column=_column_of(x0, width, layout.columns),
                        x0=x0,
                        y0=y0,
                    )
                )
            page_blocks.sort(key=lambda b: (b.column, b.y0, b.x0))
            blocks.extend(page_blocks)
    return blocks


def extract_html(path: Path) -> list[Block]:
    """Read an HTML page into blocks, discarding navigation and scripts."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    blocks: list[Block] = []
    for order, element in enumerate(soup.find_all(["h1", "h2", "h3", "h4", "li", "p"])):
        text = _clean(element.get_text(" "))
        if not text or BOILERPLATE.match(text):
            continue
        blocks.append(Block(text=text, page=1, column=0, x0=0.0, y0=float(order)))
    return blocks


def extract(path: Path, org: str, doc_type: str) -> list[Block]:
    """Extract one source document into ordered blocks."""
    if doc_type == "html":
        return extract_html(path)
    return extract_pdf(path, layout_for(org))
