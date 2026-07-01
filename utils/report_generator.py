"""
=============================================================================
Report Generator Module
=============================================================================
Uses ReportLab to create a professional PDF report from the ATS analysis
dictionary returned by rag_pipeline.py.
=============================================================================
"""
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)


def generate_pdf_report(analysis_dict):
    """
    Generates a PDF report from the ATS analysis results.

    Args:
        analysis_dict (dict): The final analysis dictionary from rag_pipeline.py.

    Returns:
        bytes: Binary content of the generated PDF file.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             rightMargin=inch * 0.75,
                             leftMargin=inch * 0.75,
                             topMargin=inch * 0.75,
                             bottomMargin=inch * 0.75)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('title', parent=styles['Title'],
                                  fontSize=22, textColor=colors.HexColor('#1e293b'),
                                  spaceAfter=6)
    h2_style = ParagraphStyle('h2', parent=styles['Heading2'],
                               fontSize=14, textColor=colors.HexColor('#334155'),
                               spaceBefore=12, spaceAfter=4)
    normal = ParagraphStyle('normal', parent=styles['Normal'],
                             fontSize=10, leading=16)

    elements = []

    # ── Title ──────────────────────────────────────────────────────────────
    elements.append(Paragraph("AI Resume Analyzer — ATS Report", title_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#94a3b8')))
    elements.append(Spacer(1, 12))

    # ── ATS Score ──────────────────────────────────────────────────────────
    ats_score = analysis_dict.get("ats_score", 0)
    recommendation = analysis_dict.get("hiring_recommendation", "N/A")
    elements.append(Paragraph(f"ATS Match Score: {ats_score}%", h2_style))
    elements.append(Paragraph(f"Hiring Recommendation: {recommendation}", normal))
    elements.append(Spacer(1, 6))

    suitability = analysis_dict.get("candidate_suitability", "")
    if suitability:
        elements.append(Paragraph(f"Candidate Suitability: {suitability}", normal))
    elements.append(Spacer(1, 12))

    # ── Skills Table ───────────────────────────────────────────────────────
    matching = ", ".join(analysis_dict.get("matching_skills", []))
    missing  = ", ".join(analysis_dict.get("missing_skills",  []))
    extra    = ", ".join(analysis_dict.get("extra_skills",    []))
    to_learn = ", ".join(analysis_dict.get("recommended_skills_to_learn", []))

    data = [
        ["Category", "Skills"],
        ["✅ Matching Skills",  matching  or "None"],
        ["❌ Missing Skills",   missing   or "None"],
        ["⭐ Extra Skills",     extra     or "None"],
        ["📚 Recommended to Learn", to_learn or "None"],
    ]

    t = Table(data, colWidths=[150, 340])
    t.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (1, 0), colors.HexColor('#334155')),
        ('TEXTCOLOR',    (0, 0), (1, 0), colors.whitesmoke),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 10),
        ('ALIGN',        (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING',   (0, 0), (-1, -1), 8),
        ('LEFTPADDING',  (0, 0), (-1, -1), 10),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 16))

    # ── Strengths ──────────────────────────────────────────────────────────
    elements.append(Paragraph("Strengths", h2_style))
    for s in analysis_dict.get('strengths', []):
        elements.append(Paragraph(f"• {s}", normal))
    elements.append(Spacer(1, 10))

    # ── Weaknesses ─────────────────────────────────────────────────────────
    elements.append(Paragraph("Weaknesses", h2_style))
    for w in analysis_dict.get('weaknesses', []):
        elements.append(Paragraph(f"• {w}", normal))
    elements.append(Spacer(1, 10))

    # ── Improvement Suggestions ────────────────────────────────────────────
    elements.append(Paragraph("Improvement Suggestions", h2_style))
    for suggestion in analysis_dict.get('resume_improvement_suggestions', []):
        elements.append(Paragraph(f"👉 {suggestion}", normal))
    elements.append(Spacer(1, 10))

    # ── Overall Feedback ───────────────────────────────────────────────────
    elements.append(Paragraph("Overall Feedback", h2_style))
    overall = analysis_dict.get('overall_feedback', 'No feedback available.')
    elements.append(Paragraph(overall, normal))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
