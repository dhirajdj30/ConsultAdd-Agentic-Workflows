from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
import re
def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# === Load Embeddings ===
def eligibilityAgent(eligibility_criteria):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # === Load RFP Vector DB ===
    rfp_db = FAISS.load_local("./VectorDB/RPF_Uploadded", embedding_model, allow_dangerous_deserialization=True)

    # === Load Company Data Vector DB ===
    company_db = FAISS.load_local("./VectorDB/Company_Uploaded", embedding_model, allow_dangerous_deserialization=True)



    # === Load LLM ===
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.7
    )

    # === Create MultiQuery Retriever for RFP DB ===
    multi_retriever_rfp = MultiQueryRetriever.from_llm(
        retriever=rfp_db.as_retriever(search_kwargs={"k": 4}),
        llm=llm
    )

    # === Create MultiQuery Retriever for Company DB ===
    multi_retriever_company = MultiQueryRetriever.from_llm(
        retriever=company_db.as_retriever(search_kwargs={"k": 4}),
        llm=llm
    )

    # === Fetch Documents from both DBs ===

    query = "Is ConsultAdd eligible to apply for this RFP?"
    rfp_docs = multi_retriever_rfp.get_relevant_documents(query)
    company_docs = multi_retriever_company.get_relevant_documents(query)

    # === Combine Context ===
    rfp_context = "\n".join([doc.page_content for doc in rfp_docs])
    company_context = "\n".join([doc.page_content for doc in company_docs])

    # === Print Contexts for Debugging ===
    print("🔍 RFP Context:\n", rfp_context[:1000])
    print("\n🏢 Company Context:\n", company_context[:1000])

    # === Prompt Template ===
    prompt_template = PromptTemplate.from_template(f"""
        You are an expert compliance agent responsible for evaluating whether ConsultAdd is eligible to apply for a government RFP. Your task is to automate and simplify the RFP analysis process using Generative AI and Retrieval-Augmented Generation (RAG) to ensure accuracy and efficiency.
        **Eligibility Criteria (from the RFP):**
        {eligibility_criteria}

        **Company Profile:**
        {company_context}

        **Your Task:**

        1. **Verify Legal Eligibility:**
        - Evaluate each criterion against ConsultAdd's company profile.
        - Use RFP context and company data to support your reasoning.
        - Mark each criterion as ✅ (met) or ❌ (unmet).
        - If information is unclear, assume "likely met" if relevant.

        2. **Extract Mandatory Eligibility Criteria:**
        - Scan and summarize must-have qualifications, certifications, and experience required to bid.
        - Flag missing requirements to prevent wasted effort on ineligible proposals.

        3. **Generate Submission Checklist:**
        - Extract and structure RFP submission requirements, including document format, attachments, and forms.

        4. **Analyze Contract Risks:**
        - Identify biased clauses that could disadvantage ConsultAdd.
        - Suggest modifications to balance contract terms.

        **Eligibility Threshold:**
        - If 80-85% or more of the criteria are met, mark as eligible.

        **RFP Context:**
        {{rfp_context}}

        **Company Profile Context:**
        {{company_context}}

        **Your Answer:**
        1. List each criterion with its status (✅ or ❌) and brief justification.
        2. Calculate the percentage of met criteria.
        3. Final Eligibility: Yes or No.
        4. Provide a concise summary explaining the eligibility decision.
    """)

    # === Create Custom QA Chain ===
    from langchain.chains.llm import LLMChain
    qa_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    # === Run Chain with Formatted Inputs ===
    response = qa_chain.invoke({
        "rfp_context": rfp_context,
        "company_context": company_context
    })

    print("\n🤖 EligibilityAgent Response:\n", response["text"])
    cleaned_response = remove_think_tags(response["text"])
    return cleaned_response