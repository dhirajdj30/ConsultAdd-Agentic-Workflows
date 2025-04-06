from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re

# 1. Load embedding model and vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("./VectorDB/faiss_rfp_index_el1", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 7})

# 2. LLM Setup (ChatGroq)
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.3)
def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()


# 3. Risk Analysis Prompt
def risk_analysis_agent():
    prompt_template = PromptTemplate.from_template("""
    You are a legal risk advisor analyzing U.S. Government RFPs for the company ConsultAdd.

    Based on the content below, your task is to:

    1. Identify any clauses or language that could pose legal or business risks to ConsultAdd.
    2. For each issue found, determine its **Severity**: High / Moderate / Low / No Issues.
    3. Suggest how to **rephrase or modify** the clause to make it more balanced.

    Respond in this format:

    [Issue 1]
    Clause: <quote the risky clause>
    Severity: <High/Moderate/Low/No Issues>
    Suggestion: <suggested changes>

    [Issue 2]
    ...

    --- Start of RFP Context ---
    {rfp_context}
    --- End of RFP Context ---

    Return only analysis based on above content.
    """)

    # 4. Build Chain
    chain = prompt_template | llm

    # 5. Run
    query = "What are the risky clauses or biased language in this RFP that could hurt ConsultAdd?"

    docs = retriever.get_relevant_documents(query)
    rfp_context = "\n\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"rfp_context": rfp_context}).content
    cleaned_response = remove_think_tags(response)
    return cleaned_response
