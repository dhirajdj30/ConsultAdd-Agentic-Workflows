from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re

def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()


# 3. Risk Analysis Prompt
def risk_analysis_agent(path: str):
    # 1. Load embedding model and vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 7})

    # 2. LLM Setup (ChatGroq)
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.3)
    prompt_template = PromptTemplate.from_template("""
    You are a legal risk advisor analyzing U.S. Government RFPs on behalf of ConsultAdd. Your task is to identify potential legal or business risks, assess their severity, and suggest balanced modifications to mitigate these risks.
        **Instructions:**

        1. **Identify Risks:**
        - Examine the RFP for clauses that may pose legal, financial, operational, or reputational risks to ConsultAdd.
        - Consider risks such as ambiguous terms, one-sided liabilities, payment terms, intellectual property issues, and termination clauses.

        2. **Assess Severity:**
        - Classify each identified risk as High, Moderate, or Low based on its potential impact on ConsultAdd.
        - **High Severity:** Risks that could significantly impact ConsultAdd's operations or finances.
        - **Moderate Severity:** Risks that pose manageable challenges but require attention.
        - **Low Severity:** Minor risks with minimal impact.
        - **No Issues:** Clauses that do not pose any risk.

        3. **Suggest Modifications:**
        - Propose specific rephrasing or modifications to balance the clauses and mitigate identified risks.
        - Ensure suggestions align with legal standards and industry best practices.

        4. **Format Response:**
        - Use the following format for each issue:
            ```
            [Issue X]
            Clause: [Quote the specific clause]
            Severity: [High/Moderate/Low/No Issues]
            Suggestion: [Propose detailed alternative language or modifications]
            ```
        - Include a **Summary Section** at the end to provide an overview of key findings for quick reference.

        5. **Analyze Only Provided Content:**
        - Base your analysis solely on the RFP context provided.
        - Do not assume information not explicitly stated in the context.

        6. **Ensure Clarity and Organization:**
        - Use bullet points or numbered lists for clarity.
        - Maintain a logical flow in the presentation of findings.

        **RFP Context:**
        {rfp_context}

        **Return:**
        Provide your analysis in the specified format, focusing only on the content provided.

        ---

        **Example Response:**

        [Issue 1]
        Clause: "The contractor shall indemnify and hold harmless the Government against any claims arising from the performance of this contract."
        Severity: High
        Suggestion: Revise to include mutual indemnification, limiting liability to direct damages and excluding indirect or consequential damages.

        [Issue 2]
        Clause: "The Government reserves the right to terminate the contract at any time for its convenience."
        Severity: Moderate
        Suggestion: Propose a notice period and conditions for termination, ensuring fair compensation for work completed.

        **Summary:**
        - High Severity: Indemnification clause requires mutual terms and liability limitations.
        - Moderate Severity: Termination clause needs a notice period and compensation terms.

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
