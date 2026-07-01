"""
=============================================================================
Embeddings Module (Optional / Legacy)
=============================================================================
Handles ChromaDB vector store creation using Ollama's nomic-embed-text model.
NOTE: This module is kept for reference but is NOT used in the current
production pipeline. The current version uses Direct Context Injection
(feeding the full resume text directly to the LLM) to avoid lossy
vector retrieval that caused skills to be hidden from the AI.
=============================================================================
"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_vector_db(documents):
    """
    Chunks the documents, generates embeddings, and stores them in ChromaDB.

    Args:
        documents (list): List of LangChain Document objects.

    Returns:
        Chroma: A ChromaDB vector store object.
    """
    # Split documents into smaller chunks for the vector DB
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # Generate embeddings using Ollama's local embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Store embeddings in ChromaDB
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="database/chroma_db"
    )

    return vectordb
