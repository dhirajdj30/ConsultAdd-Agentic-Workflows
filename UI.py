import streamlit as st
import requests

from new_RAG import embed_and_save_to_vectorstore

st.title("📄 RFP Automation System - ConsultAdd")

st.sidebar.header("Upload Section")

rfp_file = st.sidebar.file_uploader("Upload RFP PDF", type=["pdf"])
company_file = st.sidebar.file_uploader("Upload Company PDF", type=["pdf"])

if st.sidebar.button("Process and Embed PDFs"):
    if rfp_file:
        with open("./Downloaded/uploaded_rfp.pdf", "wb") as f:
            f.write(rfp_file.read())
    if company_file:
        with open("./Downloaded/company_data.pdf", "wb") as f:
            f.write(company_file.read())
    st.success("Files saved locally. Now run `RAG.py` to generate FAISS embeddings.")

if st.sidebar.button("Generate FAISS Vector DBs"):
    if rfp_file:
        embed_and_save_to_vectorstore("./Downloaded/uploaded_rfp.pdf", "./VectorDB/RPF_Uploadded")
    if company_file:
        embed_and_save_to_vectorstore("./Downloaded/company_data.pdf", "./VectorDB/Company_Uploaded")
    st.success("FAISS Vector DBs created successfully.")
# import os

# def safe_delete(filepath):
#     if os.path.exists(filepath):
#         os.remove(filepath)
#         print(f"🧹 Deleted: {filepath}")
#     else:
#         print(f"⚠️ File not found for deletion: {filepath}")

# # After saving FAISS vector stores
# print("✅ FAISS vector DBs for Company Data and RFPs created successfully.")

# # Clean up PDFs after embedding
# safe_delete("./Downloaded/uploaded_rfp.pdf")
# safe_delete("./Downloaded/company_data.pdf") 

st.header("Run Agents")

if st.button("🧠 Run Eligibility Check"):
    res = requests.post("http://localhost:8000/eligibility")
    st.write(res.json())

if st.button("📋 Generate Submission Checklist"):
    res = requests.post("http://localhost:8000/checklist")
    st.write(res.json())

if st.button("⚠️ Analyze Contract Risks"):
    res = requests.post("http://localhost:8000/risks")
    st.write(res.json())
