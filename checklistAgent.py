from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
import os
import re


def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()


# 1. Load embedding model
def checklistAgent(path: str):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rfp_db = FAISS.load_local( path, embedding_model, allow_dangerous_deserialization=True)

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
        You are an AI assistant using Generative AI and Retrieval-Augmented Generation (RAG) to analyze a government RFP document. Your task is to automate and simplify the RFP analysis process by generating a detailed submission checklist in HTML format.

        **Instructions:**
        1. Review the RFP context provided and extract all submission requirements.
        2. Organize the information into a structured HTML checklist.
        3. Focus on the following key areas:
        - **Document Formatting Rules:** Include font type, size, spacing, margins, page limits, and any specific layout instructions.
        - **Required Attachments:** List all necessary forms, resumes, certifications, and other documents.
        - **Structure/Layout Expectations:** Detail the required sections, Table of Contents (TOC), section naming conventions, and any specific formatting for each section.
        - **Submission Method and Deadlines:** Specify the method of submission (email, online portal, physical mail) and the exact deadline date and time.
        4. Use bullet points for clarity and ensure each requirement is clearly described.
        5. Do not assume any information not explicitly mentioned in the RFP context. If a detail is unclear or missing, note it as "Not specified."
        6. Format the final checklist in HTML, ensuring it is clean and easy to read.

        **RFP Context:**
        {rfp_context}

        **Output Requirements:**
        - Provide the checklist in HTML format.
        - Ensure all extracted requirements are accurately represented.
        - Highlight any potential ambiguities or missing information that could impact submission.
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