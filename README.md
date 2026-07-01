<div align="center">

# 🎯 AI Resume Analyzer

### *Intelligent ATS Scoring powered by Local LLMs + Deterministic Fuzzy Matching*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> A fully local, privacy-first ATS (Applicant Tracking System) that uses a **Two-Stage Hybrid Architecture** to deliver **100% accurate skill matching** without hallucinations.

</div>

---

## 🌟 Why This Project Is Different

Traditional ATS systems rely on **rigid keyword matching** and miss semantic equivalents (e.g., rejecting a candidate who knows `"MS SQL"` because the JD asked for `"Microsoft SQL Server"`).

Pure LLM-based systems **hallucinate** — they put skills in both "Matching" and "Missing" simultaneously, or completely miss skills from the resume.

This project solves both problems with a **Two-Stage Hybrid Pipeline**:

| Stage | Who Does It | What It Does |
|-------|-------------|--------------|
| **1. Extraction** | 🤖 Llama 3.2 | Reads resume & JD, outputs raw skill lists |
| **2. Matching** | 🐍 Python (RapidFuzz) | Deterministic fuzzy matching, alias normalization, ATS score |
| **3. Evaluation** | 🤖 Llama 3.2 | Writes qualitative HR feedback based on the hard math |

**The LLM never touches the matching math. Python does the math. The LLM does the writing.**

---

## ✨ Features

- 🔒 **100% Local & Private** — No API keys, no cloud, no data leakage. Everything runs on your machine via Ollama.
- 🎯 **Deterministic ATS Scoring** — RapidFuzz `token_set_ratio` gives you an exact, reproducible score every time.
- 🔤 **Smart Alias Normalization** — Understands `JS` → `JavaScript`, `EDA` → `Exploratory Data Analysis`, `MS SQL` → `Microsoft SQL Server`, and more.
- 🔍 **Fuzzy Skill Matching** — Matches `"Data Cleaning & Visualization"` against separate JD skills `"Data Cleaning"` and `"Data Visualization"` correctly.
- 🎨 **SaaS-like Dashboard** — Beautiful Streamlit UI with colored skill badges, metric cards, and animated progress.
- 📄 **PDF Report Export** — Download a professional PDF report of the full analysis with one click.

---

## 🏗️ Architecture

```
User uploads PDF + Job Description
              ↓
     [pdf_loader.py] — Extract raw text
              ↓
     [rag_pipeline.py] — Orchestrator
              ↓
  ┌─────── STAGE 1 ──────────┐
  │  Llama 3.2 via Ollama    │
  │  Extracts skill lists    │
  │  from resume & JD text   │
  └──────────────────────────┘
              ↓
  ┌─────── STAGE 2 ──────────┐
  │  skill_matcher.py        │
  │  • Normalize strings     │
  │  • Apply alias map       │
  │  • RapidFuzz fuzzy match │
  │  • Calculate ATS score   │
  └──────────────────────────┘
              ↓
  ┌─────── STAGE 3 ──────────┐
  │  Llama 3.2 via Ollama    │
  │  Generates: Strengths,   │
  │  Weaknesses, Feedback,   │
  │  Hiring Recommendation   │
  └──────────────────────────┘
              ↓
     [app.py] — Render Streamlit Dashboard
              ↓
     [report_generator.py] — Export PDF
```

---

## 📂 Project Structure

```
ResumeAnalyzer/
│
├── app.py                      # 🎨 Main Streamlit UI & orchestrator
├── requirement.txt             # 📦 Python dependencies
├── README.md                   # 📖 This file
├── .gitignore                  # 🚫 Git ignore rules
│
├── assets/                     # 🖼️ Images and logo
│   └── logo.png
│
└── utils/                      # ⚙️ Backend modules
    ├── pdf_loader.py           # PDF → text extraction
    ├── prompt.py               # LangChain prompt templates (2-stage)
    ├── rag_pipeline.py         # Two-stage pipeline orchestrator
    ├── skill_matcher.py        # RapidFuzz deterministic matching
    ├── ui_components.py        # Reusable Streamlit HTML/CSS widgets
    ├── report_generator.py     # ReportLab PDF report generator
    └── embeddings.py           # ChromaDB embeddings (legacy/optional)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed on your machine

### 1. Clone the Repository
```bash
git clone https://github.com/Likithkumarr/AI-Resume-Analyzer-RAG.git
cd AI-Resume-Analyzer-RAG
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv myenv
.\myenv\Scripts\activate

# Mac / Linux
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

### 4. Pull the Ollama Model
Make sure Ollama is running, then pull Llama 3.2:
```bash
ollama pull llama3.2
```

### 5. Run the Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📖 How to Use

1. **Upload Resume** — Drag and drop your resume PDF in the left panel.
2. **Paste Job Description** — Copy the full JD text into the right panel.
3. **Click Analyze** — Hit the `🚀 Analyze Resume` button.
4. **View Results** — See your ATS score, matching/missing skills, strengths, weaknesses, and hiring recommendation.
5. **Download Report** — Click `⬇️ Download Full PDF Report` to save the analysis.

---

## 📊 Output

| Section | Description |
|---------|-------------|
| **ATS Match Score** | Exact percentage `(Matched / Required) × 100` |
| **Hiring Recommendation** | Strongly Recommended / Recommended / Consider / Not Recommended |
| **Candidate Suitability** | One-paragraph HR summary of fit |
| **✅ Matching Skills** | Skills present in both resume and JD |
| **❌ Missing Skills** | Skills required by JD but absent in resume |
| **⭐ Extra Skills** | Bonus skills in resume not asked by JD |
| **Strengths & Weaknesses** | AI-generated qualitative analysis |
| **Improvement Suggestions** | Actionable resume enhancement tips |
| **Overall Feedback** | Full HR evaluation paragraph |

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Streamlit** | Web UI framework |
| **LangChain** | LLM prompt orchestration |
| **Ollama + Llama 3.2** | Local LLM execution engine |
| **RapidFuzz** | High-performance fuzzy string matching |
| **PyPDF** | PDF text extraction |
| **ReportLab** | PDF report generation |
| **json-repair** | LLM output sanitization |
| **ChromaDB** | Vector database (optional/legacy) |

---

## 🔧 Configuration

### Change the LLM
Open `utils/rag_pipeline.py` and change:
```python
llm = OllamaLLM(model="llama3.2")  # Change to "mistral", "gemma", etc.
```

### Adjust Matching Sensitivity
Open `utils/skill_matcher.py` and change the threshold:
```python
if score >= 85:  # Lower = more lenient, Higher = more strict
```

### Add Custom Skill Aliases
Open `utils/skill_matcher.py` → `normalize_skill()` → `aliases` dict:
```python
aliases = {
    "k8s": "kubernetes",
    "tf": "terraform",
    # Add your own...
}
```

---

## 🔮 Future Enhancements

- [ ] Support for `.docx` and `.txt` resume formats
- [ ] Batch processing — analyze multiple resumes at once
- [ ] LinkedIn JD URL scraping
- [ ] Interactive resume improvement editor
- [ ] FastAPI + Next.js version for production deployment

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Likith Kumar](https://github.com/Likithkumarr)

⭐ **Star this repo if you found it useful!** ⭐

</div>
