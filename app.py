import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

# 1 Load Resume
loader = PyPDFLoader("resume.pdf")
documents = loader.load()

# 2 Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# 3 Create embeddings using Ollama
# embeddings = OllamaEmbeddings(model="tinyllama")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4 Store in vector DB
vectordb = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectordb.as_retriever()

# 5 Load TinyLlama model
llm = OllamaLLM(model="tinyllama")

# 6 Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 7 Job description input
job_description = input("Enter Job Description: ")
query = f"""
You are an ATS (Applicant Tracking System).

Analyze the candidate resume against the job description.

Job Description:
{job_description}

From the resume context identify:

1. ATS Score (0–100)
2. Matching Skills
3. Missing Skills
4. Skills the candidate should learn for this role

Return the answer in the following format:

ATS Score: <number> %

Matching Skills:
- skill1
- skill2

Missing Skills:
- skill1
- skill2

Recommended Skills to Learn:
- skill1
- skill2
"""
result = qa_chain.invoke({"query": query})

print("\nResume Analysis Result:\n")
print(result["result"])