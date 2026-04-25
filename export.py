"""
export.py
PDF report generation for Robin investigations using reportlab Platypus.
"""

import io
import re
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)

# Colour palette
_RED    = colors.HexColor("#CC2222")
_DARK   = colors.HexColor("#111111")
_GRAY   = colors.HexColor("#555555")
_LGRAY  = colors.HexColor("#AAAAAA")
_WHITE  = colors.white
_OFFWHT = colors.HexColor("#F7F7F7")


def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "RobinTitle", parent=base["Title"],
            fontSize=22, textColor=_RED, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "RobinSubtitle", parent=base["Normal"],
            fontSize=9, textColor=_GRAY, spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "RobinSection", parent=base["Heading2"],
            fontSize=13, textColor=_RED, spaceBefore=14, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "RobinBody", parent=base["Normal"],
            fontSize=9, textColor=_DARK, leading=14, spaceAfter=3,
        ),
        "source_link": ParagraphStyle(
            "RobinLink", parent=base["Normal"],
            fontSize=8, textColor=_GRAY, leading=11,
        ),
    }


# ---------------------------------------------------------------------------
# Lightweight Markdown → reportlab flowables
# ---------------------------------------------------------------------------

def _inline_md(text: str) -> str:
    """Convert inline Markdown to reportlab XML. Escapes HTML entities first."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_",       r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', text)
    return text


def _md_to_flowables(text: str, styles: dict) -> list:
    """Convert a Markdown string to a list of reportlab flowables."""
    flowables = []
    for line in text.splitlines():
        s = line.strip()

        if not s:
            flowables.append(Spacer(1, 4))
            continue

        if s.startswith("### "):
            flowables.append(Paragraph(f"<b>{_inline_md(s[4:])}</b>", styles["section"]))
        elif s.startswith("## "):
            flowables.append(Paragraph(f"<b>{_inline_md(s[3:])}</b>", styles["section"]))
        elif s.startswith("# "):
            flowables.append(Paragraph(f"<b>{_inline_md(s[2:])}</b>", styles["section"]))
        elif s.startswith(("- ", "* ", "+ ")):
            flowables.append(Paragraph(f"&bull;&nbsp;&nbsp;{_inline_md(s[2:])}", styles["body"]))
        elif re.match(r"^\d+\.\s+", s):
            m = re.match(r"^(\d+)\.\s+(.*)", s)
            flowables.append(Paragraph(f"{m.group(1)}.&nbsp;&nbsp;{_inline_md(m.group(2))}", styles["body"]))
        elif s in ("---", "***", "___"):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=_LGRAY))
        else:
            flowables.append(Paragraph(_inline_md(s), styles["body"]))

    return flowables


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf(investigation: dict) -> bytes:
    """
    Build a formatted PDF report for one investigation dict.
    Returns raw PDF bytes suitable for st.download_button.
    """
    styles = _build_styles()
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title=f"Robin — {investigation.get('query', '')}",
        author="Robin: AI-Powered Dark Web OSINT Tool",
    )

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Robin Investigation Report", styles["title"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_RED, spaceAfter=8))

    # Timestamp formatting
    ts_raw = investigation.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d  %H:%M:%S")
    except Exception:
        ts = ts_raw

    # Meta table
    meta_rows = [
        ["Query",         investigation.get("query", "—")],
        ["Refined Query", investigation.get("refined_query", "—")],
        ["Model",         investigation.get("model", "—")],
        ["Domain",        investigation.get("preset", "—")],
        ["Status",        investigation.get("status", "active").capitalize()],
        ["Tags",          investigation.get("tags", "") or "—"],
        ["Generated",     ts],
    ]
    meta_table = Table(meta_rows, colWidths=[38 * mm, None], hAlign="LEFT")
    meta_table.setStyle(TableStyle([
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",      (0, 0), (0, -1), _GRAY),
        ("TEXTCOLOR",      (1, 0), (1, -1), _DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_OFFWHT, _WHITE]),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("GRID",           (0, 0), (-1, -1), 0.25, _LGRAY),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── Sources ───────────────────────────────────────────────────────────────
    sources = investigation.get("sources", [])
    if sources:
        story.append(Paragraph("Sources", styles["section"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=_LGRAY, spaceAfter=4))
        for i, src in enumerate(sources, 1):
            title = _inline_md(src.get("title", "Untitled"))
            link  = src.get("link", "")
            display = f"{i}.&nbsp;&nbsp;<b>{title}</b>"
            if link:
                short = link[:90] + ("…" if len(link) > 90 else "")
                display += f"<br/><font name='Courier' size='7' color='#999999'>{short}</font>"
            story.append(Paragraph(display, styles["source_link"]))
            story.append(Spacer(1, 2))
        story.append(Spacer(1, 6))

    # ── Findings (new page) ───────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Findings", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_LGRAY, spaceAfter=6))
    summary = investigation.get("summary", "No summary available.")
    story.extend(_md_to_flowables(summary, styles))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_LGRAY))
    story.append(Paragraph(
        "Generated by Robin — AI-Powered Dark Web OSINT Tool. "
        "For lawful investigative purposes only.",
        styles["subtitle"],
    ))

    doc.build(story)
    return buf.getvalue()