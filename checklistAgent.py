from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
import os
import re
# Setup: Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()


# 1. Load embedding model
def checklistAgent():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rfp_db = FAISS.load_local("./VectorDB/RPF_Uploadded", embedding_model, allow_dangerous_deserialization=True)

    # 2. Load LLM
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.7
    )

    # 3. Create multi-query retriever
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=rfp_db.as_retriever(search_kwargs={"k": 8}),
        llm=llm
    )

    # 4. Define prompt template (human-readable checklist format)
    prompt_template = PromptTemplate.from_template("""
    You are an AI assistant tasked with reviewing a government RFP document.

    From the RFP context below, generate a **submission checklist** with clearly described requirements in human-readable bullet points.

    Focus on:
    - Document formatting rules (font, spacing, page limits, etc.)
    - Required forms, resumes, certifications, etc.
    - Structure or layout expectations (TOC, section naming, etc.)
    - Submission method (email, portal, physical mail), deadlines

    Use bullet points. Do **not** assume anything. If something is not mentioned, skip it.

    RFP Context:
    {rfp_context}
                                                   
    all should be in html format
    """)

    # 5. Perform multi-query retrieval
    query = "What are the submission checklist requirements in the RFP?"
    docs = multi_retriever.invoke(query)
    rfp_context = "\n\n".join([doc.page_content for doc in docs])

    # 6. Run prompt with retrieved context
    chain = prompt_template | llm
    response = chain.invoke({"rfp_context": rfp_context}).content
    # response = response["result"]
    cleaned_submission_checklist = remove_think_tags(response)
    return cleaned_submission_checklist