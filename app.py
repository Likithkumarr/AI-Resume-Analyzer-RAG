"""
=============================================================================
AI Resume Analyzer - Main Streamlit Application
=============================================================================
This is the frontend orchestrator. It renders the full SaaS-like dashboard,
handles user inputs (PDF upload + Job Description), triggers the backend
two-stage pipeline, and displays the final analysis results.
=============================================================================
"""
import streamlit as st
from utils.pdf_loader import process_uploaded_pdf
from utils.rag_pipeline import analyze_resume_with_rag
from utils.ui_components import render_skill_badges, render_score_card, render_recommendation
from utils.report_generator import generate_pdf_report

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide sidebar scrollbar */
    [data-testid="stSidebar"] { overflow-y: hidden !important; }
    [data-testid="stSidebar"] > div:first-child { overflow-y: hidden !important; }

    /* Style the analyze button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        transform: translateY(-1px);
    }

    /* Style headers */
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 24px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Project Logo
    try:
        st.image("assets/logo.png", width=200)
    except Exception:
        st.markdown("## 🎯 AI Resume Analyzer")

    st.markdown("---")
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
1. **Upload** your resume PDF
2. **Paste** the job description
3. **Click** Analyze Resume
4. **Get** your ATS score instantly
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Powered By")
    tech_html = "".join([
        f'<span style="display:inline-block; background:#1e293b; color:#94a3b8; '
        f'padding:3px 10px; border-radius:12px; margin:3px 2px; font-size:12px; '
        f'border:1px solid #334155;">{t}</span>'
        for t in ["Ollama", "Llama 3.2", "LangChain", "RapidFuzz", "Streamlit", "ReportLab"]
    ])
    st.markdown(f'<div style="line-height:2;">{tech_html}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔒 100% Local & Private")
    st.caption("Your resume is never sent to any cloud service.")

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("# 🎯 AI Resume Analyzer")
st.markdown("##### *Deterministic ATS Matching powered by Llama 3.2 + RapidFuzz*")
st.markdown("---")

# Input columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

with col2:
    st.markdown("### 📋 Job Description")
    job_description = st.text_area(
        "Paste the full job description here",
        height=250,
        placeholder="Paste the job description here...",
        label_visibility="collapsed"
    )

st.markdown("---")

# ── Analyze Button ────────────────────────────────────────────────────────────
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    analyze_clicked = st.button("🚀 Analyze Resume", use_container_width=True)

# ── Pipeline Execution ────────────────────────────────────────────────────────
if analyze_clicked:
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF resume.")
    elif not job_description.strip():
        st.warning("⚠️ Please paste a job description.")
    else:
        with st.spinner("🔍 Analyzing resume against job description..."):
            my_bar = st.progress(0, text="📄 Reading PDF...")
            docs = process_uploaded_pdf(uploaded_file)

            # Combine all pages into one string — bypasses lossy RAG chunking
            full_resume_text = "\n".join([doc.page_content for doc in docs])

            my_bar.progress(30, text="🤖 Stage 1: Extracting skills with Llama 3.2...")
            result = analyze_resume_with_rag(full_resume_text, job_description)

            my_bar.progress(85, text="📊 Generating PDF report...")
            if "error" not in result:
                pdf_bytes = generate_pdf_report(result)

            my_bar.progress(100, text="✅ Analysis complete!")
            my_bar.empty()

        # ── Results Dashboard ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        if "error" in result:
            st.error(result["error"])
        else:
            # ATS Score Card
            render_score_card(result.get("ats_score", 0))

            # Hiring Recommendation
            st.markdown("#### Hiring Recommendation")
            render_recommendation(result.get("hiring_recommendation", "N/A"))

            # Candidate Suitability
            suitability = result.get("candidate_suitability", "")
            if suitability:
                st.markdown(f"**📋 Candidate Suitability:** {suitability}")

            st.markdown("---")

            # Skill Counters
            matching = result.get("matching_skills", [])
            missing  = result.get("missing_skills",  [])
            extra    = result.get("extra_skills",    [])

            cnt1, cnt2, cnt3 = st.columns(3)
            cnt1.metric("✅ Matching Skills", len(matching))
            cnt2.metric("❌ Missing Skills",  len(missing))
            cnt3.metric("⭐ Extra Skills",    len(extra))

            # Skills Analysis
            st.markdown("### 🛠️ Skills Analysis")

            st.markdown(f"**✅ Matching Skills** &nbsp;&nbsp;<span style='color:#94a3b8; font-size:13px;'>({len(matching)} skills found in both resume & JD)</span>", unsafe_allow_html=True)
            render_skill_badges(matching, "#22c55e")

            st.markdown(f"**❌ Missing Skills** &nbsp;&nbsp;<span style='color:#94a3b8; font-size:13px;'>({len(missing)} skills required but not found in resume)</span>", unsafe_allow_html=True)
            render_skill_badges(missing, "#ef4444")

            st.markdown(f"**⭐ Extra Skills** &nbsp;&nbsp;<span style='color:#94a3b8; font-size:13px;'>({len(extra)} bonus skills in resume not required by JD)</span>", unsafe_allow_html=True)
            render_skill_badges(extra, "#3b82f6")

            st.markdown("---")

            # Strengths & Weaknesses
            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("### 🌟 Resume Strengths")
                for s in result.get("strengths", []):
                    st.markdown(f"✔ {s}")

            with col_w:
                st.markdown("### ⚠️ Resume Weaknesses")
                for w in result.get("weaknesses", []):
                    st.markdown(f"• {w}")

            # Improvement Suggestions
            st.markdown("### 📈 Improvement Suggestions")
            for suggestion in result.get("resume_improvement_suggestions", []):
                st.markdown(f"👉 {suggestion}")

            # Recommended Skills
            to_learn = result.get("recommended_skills_to_learn", [])
            if to_learn:
                st.markdown("### 📚 Recommended Skills to Learn")
                render_skill_badges(to_learn, "#8b5cf6")

            # Overall Feedback
            st.markdown("### 📝 Overall Feedback")
            st.info(result.get("overall_feedback", "No feedback available."))

            # Download Button
            st.markdown("---")
            st.download_button(
                label="⬇️ Download Full PDF Report",
                data=pdf_bytes,
                file_name="resume_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )