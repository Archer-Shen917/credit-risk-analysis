"""Rebuild concise Chinese PDF reports from saved results and JSON page templates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from credit_risk.report_context import build_context


PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - 80


def register_chinese_font() -> str:
    """Choose an installed CJK TrueType font without distributing font files.

    Args:
        None; CJK_FONT_PATH may specify a locally installed TrueType font.

    Returns:
        Registered ReportLab font name, with a CID fallback when necessary.
    """
    candidates = [
        os.environ.get("CJK_FONT_PATH", ""),
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                pdfmetrics.registerFont(TTFont("ReportCN", candidate))
                return "ReportCN"
            except (ValueError, OSError, TTFError):
                continue
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def make_styles(font: str) -> dict:
    """Create a restrained, readable page hierarchy for Chinese reports.

    Args:
        font: Registered Chinese font family name.

    Returns:
        Paragraph styles for page titles, text, captions, and table cells.
    """
    common = {"fontName": font, "wordWrap": "CJK", "splitLongWords": True}
    return {
        "title": ParagraphStyle("Title", **common, fontSize=19, leading=25,
                                textColor=colors.HexColor("#17384D"), spaceAfter=12),
        "h": ParagraphStyle("Heading", **common, fontSize=12, leading=17,
                            textColor=colors.HexColor("#1C6077"), spaceBefore=5,
                                spaceAfter=6),
        "p": ParagraphStyle("Body", **common, fontSize=10, leading=15,
                            spaceAfter=8, alignment=TA_LEFT),
        "note": ParagraphStyle("Note", **common, fontSize=8, leading=11.5,
                               textColor=colors.HexColor("#53616B"), spaceAfter=5),
        "cell": ParagraphStyle("Cell", **common, fontSize=8.3, leading=11.5),
        "header": ParagraphStyle("HeaderCell", **common, fontSize=8.3, leading=11.5,
                                 textColor=colors.white),
        "caption": ParagraphStyle("Caption", **common, fontSize=8, leading=11,
                                  alignment=TA_CENTER, spaceAfter=7),
    }


def render_table(rows: list[list[str]], styles: dict,
                 ratios: list[float] | None = None) -> Table:
    """Wrap table cells and apply consistent non-overflowing column widths.

    Args:
        rows: Header plus formatted data rows.
        styles: Registered paragraph styles.
        ratios: Relative column widths; equal widths when omitted.

    Returns:
        ReportLab table ready for page flow.
    """
    proportions = ratios or [1.0] * len(rows[0])
    widths = [CONTENT_WIDTH * value / sum(proportions) for value in proportions]
    cells = [
        [Paragraph(escape(str(value)), styles["header" if number == 0 else "cell"])
         for value in row]
        for number, row in enumerate(rows)
    ]
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17384D")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#17384D")),
    ]))
    return table


def build_report(root: Path, specification: dict, context: dict,
                 font: str, styles: dict) -> None:
    """Build one report with explicitly planned page breaks.

    Args:
        root: Repository root containing charts and output PDF directories.
        specification: Report title, filename, and per-page section definitions.
        context: Verified scalar values and table rows generated from outputs.
        font: Installed Chinese font registered with ReportLab.
        styles: Paragraph and table style mapping.

    Returns:
        None; a readable, searchable PDF is saved to reports/.
    """
    destination = root / specification["file"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination), pagesize=A4, rightMargin=40, leftMargin=40,
        topMargin=48, bottomMargin=43,
        title=specification["title"], author="Archer-Shen917",
        subject="Home Credit公开数据研究；非银行生产业务实证",
    )
    total = len(specification["pages"])

    def page_frame(canvas: object, doc: object) -> None:
        """Draw stable headers and actual page numbering.

        Args:
            canvas: ReportLab drawing canvas.
            doc: Active ReportLab document template.

        Returns:
            None; headers and footers are added to each page.
        """
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CCD8DE"))
        canvas.line(40, PAGE_HEIGHT - 31, PAGE_WIDTH - 40, PAGE_HEIGHT - 31)
        canvas.setFillColor(colors.HexColor("#53616B"))
        canvas.setFont(font, 8)
        canvas.drawString(40, PAGE_HEIGHT - 22, "CREDIT RISK RESEARCH  |  2026.10.26")
        canvas.drawRightString(PAGE_WIDTH - 40, PAGE_HEIGHT - 22,
            specification["short_title"])
        canvas.line(40, 34, PAGE_WIDTH - 40, 34)
        canvas.drawString(40, 21, "公开数据离线研究 | 原始数据不随仓库分发 | Archer-Shen917")
        canvas.drawRightString(PAGE_WIDTH - 40, 21, f"{doc.page} / {total}")
        canvas.restoreState()

    story = []
    for number, page in enumerate(specification["pages"]):
        if number:
            story.append(PageBreak())
        story.append(Paragraph(escape(page["title"].format_map(context)),
            styles["title"]))
        for section in page["sections"]:
            kind = section["kind"]
            if kind in {"p", "h", "note"}:
                text = section["text"].format_map(context)
                story.append(Paragraph(escape(text).replace("\n", "<br/>"),
                    styles[kind]))
            elif kind == "table":
                rows = context["tables"][section[
                    "key"]] if "key" in section else section["rows"]
                story.append(render_table(rows, styles, section.get("ratios")))
                story.append(Spacer(1, 9))
            elif kind == "image":
                path = root / section["path"]
                with PillowImage.open(path) as image:
                    width, height = image.size
                factor = min(CONTENT_WIDTH / width, section.get("height", 200) / height)
                story.append(Image(str(path), width=width * factor,
                    height=height * factor))
                if "caption" in section:
                    story.append(Paragraph(escape(section["caption"].format_map(
                        context)), styles["caption"]))
            elif kind == "spacer":
                story.append(Spacer(1, section["height"]))
            else:
                raise ValueError(f"Unknown report section: {kind}")
    document.build(story, onFirstPage=page_frame, onLaterPages=page_frame)


def generate_reports(root: Path) -> None:
    """Create all stage PDFs and the five-page closing summary.

    Args:
        root: Repository with completed numerical outputs and page templates.

    Returns:
        None; every configured report is rebuilt from current saved results.
    """
    templates = json.loads((root / "configs/report_templates.json").read_text(
        encoding="utf-8"))
    context = build_context(root)
    font = register_chinese_font()
    styles = make_styles(font)
    for specification in templates:
        build_report(root, specification, context, font, styles)


def main() -> None:
    """Read CLI settings and rebuild PDFs.

    Args:
        None; --root is read from the command line.

    Returns:
        None; PDF reports are written under the project root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    generate_reports(arguments.root)


if __name__ == "__main__":
    main()
