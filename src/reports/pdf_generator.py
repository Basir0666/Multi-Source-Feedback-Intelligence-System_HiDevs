"""
PDF Report Generator
Creates professional weekly insight reports using ReportLab
"""
import io
import logging
from datetime import datetime
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
C_PRIMARY   = colors.HexColor("#4F46E5")
C_ACCENT    = colors.HexColor("#7C3AED")
C_SUCCESS   = colors.HexColor("#10B981")
C_WARNING   = colors.HexColor("#F59E0B")
C_DANGER    = colors.HexColor("#EF4444")
C_LIGHT     = colors.HexColor("#F3F4F6")
C_MID_GRAY  = colors.HexColor("#9CA3AF")
C_DARK      = colors.HexColor("#1F2937")
C_WHITE     = colors.white


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_sentiment_pie(df: pd.DataFrame) -> io.BytesIO:
    counts = df["sentiment"].value_counts()
    palette = {"Positive": "#10B981", "Neutral": "#F59E0B", "Negative": "#EF4444"}
    clrs = [palette.get(s, "#9CA3AF") for s in counts.index]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(counts.values, labels=counts.index, colors=clrs,
           autopct="%1.1f%%", startangle=90, pctdistance=0.78)
    ax.set_title("Sentiment Distribution", fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    return _fig_to_buf(fig)


def _chart_trend(df: pd.DataFrame) -> io.BytesIO:
    from src.intelligence.trend_detector import compute_rolling_sentiment
    daily = compute_rolling_sentiment(df)

    fig, ax = plt.subplots(figsize=(6.5, 3))
    if not daily.empty:
        ax.plot(daily["date_only"], daily["avg_sentiment"],
                color="#9CA3AF", lw=1, alpha=0.5, label="Daily avg")
        ax.plot(daily["date_only"], daily["rolling_sentiment"],
                color="#4F46E5", lw=2.5, label="7-day rolling")
        ax.axhline(0, color="#EF4444", ls="--", lw=0.8, alpha=0.4)
        ax.fill_between(daily["date_only"], daily["rolling_sentiment"], 0,
                        where=(daily["rolling_sentiment"] >= 0),
                        alpha=0.12, color="#10B981")
        ax.fill_between(daily["date_only"], daily["rolling_sentiment"], 0,
                        where=(daily["rolling_sentiment"] < 0),
                        alpha=0.12, color="#EF4444")
        ax.set_ylabel("Sentiment Score", fontsize=9)
        ax.set_title("Sentiment Trend Over Time", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        plt.xticks(rotation=25, fontsize=7)
    plt.tight_layout()
    return _fig_to_buf(fig)


def _chart_categories(df: pd.DataFrame) -> io.BytesIO:
    cat = df["category"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.barh(cat.index[::-1], cat.values[::-1], color="#4F46E5", alpha=0.82)
    ax.set_xlabel("Number of Reviews", fontsize=9)
    ax.set_title("Feedback by Category", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    for bar, val in zip(bars, cat.values[::-1]):
        ax.text(bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)
    plt.tight_layout()
    return _fig_to_buf(fig)


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_pdf_report(
    df: pd.DataFrame,
    trend_info: Dict,
    trending_issues: List[Dict],
    date_range: str = "",
    company_name: str = "Your App",
) -> bytes:
    """
    Build a professional A4 PDF insight report.

    Args:
        df              : Processed & filtered feedback DataFrame
        trend_info      : Output from detect_sentiment_trend()
        trending_issues : Output from get_trending_issues()
        date_range      : Human-readable date string for the header
        company_name    : App / company name

    Returns:
        Raw PDF bytes
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "rTitle", parent=styles["Title"],
        fontSize=22, textColor=C_PRIMARY, fontName="Helvetica-Bold", spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "rH2", parent=styles["Heading2"],
        fontSize=13, textColor=C_DARK, fontName="Helvetica-Bold",
        spaceBefore=10, spaceAfter=4,
    )
    body = ParagraphStyle(
        "rBody", parent=styles["Normal"],
        fontSize=9.5, textColor=C_DARK, spaceAfter=3,
    )
    caption = ParagraphStyle(
        "rCaption", parent=styles["Normal"],
        fontSize=8.5, textColor=C_MID_GRAY,
    )

    story = []
    total = len(df)
    pos   = (df["sentiment"] == "Positive").sum()
    neg   = (df["sentiment"] == "Negative").sum()
    neu   = total - pos - neg
    avg_s = round(df["score"].mean(), 3) if "score" in df.columns else 0.0
    srcs  = df["source"].nunique()

    # ── Cover / Header ──────────────────────────────────────────────────────
    story.append(Paragraph("📊 Feedback Intelligence Report", title_style))
    story.append(Paragraph(
        f"<b>{company_name}</b> &nbsp;·&nbsp; {date_range or datetime.now().strftime('%B %Y')}",
        body,
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        caption,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=C_PRIMARY, spaceAfter=8))

    # ── Executive Summary ────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", h2_style))

    summary_rows = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Feedback",    f"{total:,}",          "Avg Sentiment Score", f"{avg_s:.3f}"],
        ["Positive",          f"{pos:,} ({round(pos/total*100,1) if total else 0}%)",
         "Negative",          f"{neg:,} ({round(neg/total*100,1) if total else 0}%)"],
        ["Neutral",           f"{neu:,} ({round(neu/total*100,1) if total else 0}%)",
         "Data Sources",      str(srcs)],
        ["Trend Direction",   trend_info.get("direction", "N/A").title(),
         "Period Change",     f"{trend_info.get('change', 0):+.3f}"],
    ]
    col_w = [4.5*cm, 3.5*cm, 4.5*cm, 3.5*cm]
    tbl = Table(summary_rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1,  0), C_PRIMARY),
        ("TEXTCOLOR",   (0, 0), (-1,  0), C_WHITE),
        ("FONTNAME",    (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",        (0, 0), (-1, -1), 0.5, C_WHITE),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"<b>Trend:</b> {trend_info.get('description', 'N/A')}", body,
    ))
    story.append(Spacer(1, 0.3*cm))

    # ── Sentiment Charts ─────────────────────────────────────────────────────
    story.append(Paragraph("Sentiment Analysis", h2_style))
    try:
        pie_buf   = _chart_sentiment_pie(df)
        trend_buf = _chart_trend(df)
        chart_tbl = Table(
            [[Image(pie_buf, width=7*cm, height=5.5*cm),
              Image(trend_buf, width=10*cm, height=5.5*cm)]],
            colWidths=[7.5*cm, 10.5*cm],
        )
        chart_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(chart_tbl)
    except Exception as exc:
        logger.warning("Sentiment charts failed: %s", exc)
        story.append(Paragraph("(Charts unavailable)", caption))

    story.append(Spacer(1, 0.3*cm))

    # ── Category Chart ────────────────────────────────────────────────────────
    story.append(Paragraph("Feedback Categories", h2_style))
    try:
        cat_buf = _chart_categories(df)
        story.append(Image(cat_buf, width=16*cm, height=7*cm))
    except Exception as exc:
        logger.warning("Category chart failed: %s", exc)

    story.append(Spacer(1, 0.3*cm))

    # ── Trending Issues ───────────────────────────────────────────────────────
    story.append(Paragraph("Top Trending Negative Issues", h2_style))
    if trending_issues:
        issue_rows = [["#", "Category", "Volume", "Avg Sentiment", "Priority"]]
        for i, iss in enumerate(trending_issues, 1):
            issue_rows.append([
                str(i), iss["category"], str(iss["count"]),
                f"{iss['avg_sentiment']:.3f}", f"{iss['avg_priority']:.0f}/100",
            ])
        itbl = Table(issue_rows, colWidths=[1*cm, 5.5*cm, 2.5*cm, 3.5*cm, 3.5*cm])
        itbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1,  0), C_DANGER),
            ("TEXTCOLOR",  (0, 0), (-1,  0), C_WHITE),
            ("FONTNAME",   (0, 0), (-1,  0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
            ("GRID",       (0, 0), (-1, -1), 0.5, C_WHITE),
            ("ALIGN",      (2, 0), (-1, -1), "CENTER"),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(itbl)
    else:
        story.append(Paragraph("No significant trending issues detected.", body))

    story.append(Spacer(1, 0.3*cm))

    # ── Critical Feedback Samples ─────────────────────────────────────────────
    story.append(Paragraph("Critical Feedback Samples", h2_style))
    critical = df[df["priority_label"].str.contains("Critical", na=False)].head(5)
    if critical.empty:
        critical = df[df["sentiment"] == "Negative"].nsmallest(5, "score")

    for _, row in critical.iterrows():
        snippet = str(row["text"])[:220] + ("…" if len(str(row["text"])) > 220 else "")
        src     = row.get("source", "Unknown")
        dt      = str(row.get("date", ""))[:10]
        story.append(Paragraph(
            f'<font color="#EF4444">●</font> <b>[{src}]</b> '
            f'<font color="#9CA3AF">{dt}</font> — {snippet}',
            body,
        ))
        story.append(Spacer(1, 0.08*cm))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_MID_GRAY))
    story.append(Paragraph(
        f"Generated by Feedback Intelligence System · {datetime.now().year}",
        caption,
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
