import streamlit as st
import tempfile
import re
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="Resume Analyzer",page_icon="")
st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
job_description = st.text_area("Enter Job Description")

if st.button("Analyze Resume"):

    if uploaded_file is None or job_description == "":
        st.warning("Please upload resume and enter job description")
        st.stop()

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load resume
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Vector DB
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings
    )

    retriever = vectordb.as_retriever()

    # LLM
    llm = OllamaLLM(model="tinyllama")

    # RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    query = f"""
You are an ATS resume analyzer.

Compare the resume with the job description.

Job Description:
{job_description}

Return ONLY in this format:

ATS Score: <number> %

Matching Skills:
- skill
- skill

Missing Skills:
- skill
- skill

Skills to Learn:
- skill
- skill
"""

    result = qa_chain.invoke({"query": query})
    response = result["result"]

    st.subheader("📊 ATS Resume Analysis")

    # Extract ATS score
    match = re.search(r'ATS Score:\s*(\d+)', response)

    if match:
        score = int(match.group(1))
        st.subheader("ATS Score")
        st.progress(score)
        st.write(f"{score}% Match")

    # Parse skills
    matching = []
    missing = []
    learn = []

    section = None

    for line in response.split("\n"):

        if "Matching Skills" in line:
            section = "matching"

        elif "Missing Skills" in line:
            section = "missing"

        elif "Skills to Learn" in line:
            section = "learn"

        elif line.strip().startswith("-"):
            skill = line.replace("-", "").strip()

            if section == "matching":
                matching.append(skill)

            elif section == "missing":
                missing.append(skill)

            elif section == "learn":
                learn.append(skill)

    # Create dataframe
    data = {
        "Matching Skills": matching,
        "Missing Skills": missing,
        "Skills to Learn": learn
    }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

    st.subheader("📋 Skill Analysis")
    st.dataframe(df)

    # Download report
    report = f"""
Resume ATS Analysis Report

{response}
"""

    st.download_button(
        label="📥 Download Resume Report",
        data=report,
        file_name="resume_analysis.txt",
        mime="text/plain"
    )