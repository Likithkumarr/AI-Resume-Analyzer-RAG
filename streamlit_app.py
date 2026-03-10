import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
import tempfile

st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

job_description = st.text_area("Enter Job Description")

if st.button("Analyze Resume"):

    if uploaded_file and job_description:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

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
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

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

            Return ONLY this format:

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

        # st.subheader("Resume Analysis Result")
        # st.write(result["result"])

        # response = result["result"]
        # st.subheader("📊 ATS Analysis")
        # st.text(response)

        response = result["result"]
        st.subheader("📊 ATS Resume Analysis")
        lines = response.split("\n")
        for line in lines:
            if "ATS Score" in line:
                st.success(line)
            elif "Matching Skills" in line:
                st.subheader("✅ Matching Skills")
            elif "Missing Skills" in line:
                st.subheader("❌ Missing Skills")
            elif "Skills to Learn" in line:
                st.subheader("📚 Skills to Learn")
            else:
                st.write(line)

    else:
        st.warning("Please upload resume and enter job description.")