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

# company_docs = process_pdf("./Data/company_data.pdf")  # use your correct path if different
# company_db = FAISS.from_documents(company_docs, embedding_model)
# company_db.save_local("outputs/faiss_company_index")

# === 3. Create vector DB for RFP (eligible) ===
rfp_docs = process_pdf("./Data/eligible1.pdf")  # this is the uploaded eligible RFP
rfp_db = FAISS.from_documents(rfp_docs, embedding_model)
rfp_db.save_local("outputs/faiss_rfp_index_el1")

print("✅ FAISS vector DBs for Company Data and RFPs created successfully.")
