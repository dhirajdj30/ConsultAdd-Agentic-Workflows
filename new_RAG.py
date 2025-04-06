import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# === 1. Utility to load and split PDFs ===
def process_pdf(file_path, chunk_size=500, chunk_overlap=50):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)

def embed_and_save_to_vectorstore(pdf_path: str, index_path: str):
    docs = process_pdf(pdf_path)
    vector_db = FAISS.from_documents(docs, embedding_model)
    vector_db.save_local(index_path)
    return f"✅ Embedding saved to: {index_path}"


print("✅ FAISS vector DBs for Company Data and RFPs created successfully.")
