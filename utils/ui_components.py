"""
=============================================================================
UI Components Module
=============================================================================
Contains reusable Streamlit HTML/CSS rendering functions for skills badges,
metric cards, and recommendation displays.
=============================================================================
"""
import streamlit as st


def render_skill_badges(skills, color):
    """
    Renders a list of skills as colored pill-shaped HTML badges.

    Args:
        skills (list): List of skill strings to render.
        color (str):   CSS hex color for the badge background.
    """
    if not skills:
        st.markdown("*None found.*")
        return

    badges_html = "".join([
        f'<span style="display:inline-block; background-color:{color}; color:white; '
        f'padding:4px 12px; border-radius:20px; margin:4px 3px; font-size:13px; '
        f'font-weight:500;">{skill}</span>'
        for skill in sorted(skills)
    ])
    st.markdown(f'<div style="line-height:2.2;">{badges_html}</div>', unsafe_allow_html=True)


def render_score_card(ats_score):
    """
    Renders the ATS score as a large styled metric card.

    Args:
        ats_score (int): ATS score between 0 and 100.
    """
    if ats_score >= 70:
        color = "#22c55e"   # green
        label = "🟢 Excellent Match"
    elif ats_score >= 50:
        color = "#f59e0b"   # amber
        label = "🟡 Moderate Match"
    else:
        color = "#ef4444"   # red
        label = "🔴 Low Match"

    st.markdown(f"""
    <div style="text-align:center; padding:20px; background: linear-gradient(135deg, #1e293b, #334155);
                border-radius:16px; border: 1px solid #475569; margin-bottom:20px;">
        <p style="color:#94a3b8; font-size:14px; margin:0;">ATS Match Score</p>
        <h1 style="color:{color}; font-size:64px; margin:10px 0;">{ats_score}%</h1>
        <p style="color:{color}; font-size:16px; margin:0;">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def render_recommendation(recommendation):
    """
    Renders the hiring recommendation as a styled banner.

    Args:
        recommendation (str): One of the four hiring recommendation strings.
    """
    colors = {
        "Strongly Recommended": ("#22c55e", "🌟"),
        "Recommended":          ("#3b82f6", "👍"),
        "Consider":             ("#f59e0b", "🤔"),
        "Not Recommended":      ("#ef4444", "❌"),
    }
    color, icon = colors.get(recommendation, ("#64748b", "ℹ️"))

    st.markdown(f"""
    <div style="padding:14px 20px; background-color:{color}22; border-left:4px solid {color};
                border-radius:8px; margin:10px 0;">
        <span style="color:{color}; font-size:18px; font-weight:700;">{icon} {recommendation}</span>
    </div>
    """, unsafe_allow_html=True)
