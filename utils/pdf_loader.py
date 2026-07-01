"""
=============================================================================
PDF Loader Module
=============================================================================
Handles PDF file uploading and text extraction using LangChain's PyPDFLoader.
Returns raw Document objects containing full resume text.
=============================================================================
"""
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader


def process_uploaded_pdf(uploaded_file):
    """
    Takes a Streamlit UploadedFile object, saves it temporarily to disk,
    and loads it using PyPDFLoader.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        List of LangChain Document objects containing the resume text.
    """
    # Save the uploaded file to a temporary location on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load the PDF and extract text page by page
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        # Always clean up the temp file
        os.unlink(tmp_path)

    return documents
