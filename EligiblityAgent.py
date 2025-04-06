from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document

# === Load Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Load RFP Vector DB ===
rfp_db = FAISS.load_local("./outputs/faiss_rfp_index_not_el", embedding_model, allow_dangerous_deserialization=True)

# === Load Company Data Vector DB ===
company_db = FAISS.load_local("./outputs/faiss_company_index", embedding_model, allow_dangerous_deserialization=True)



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


import re
def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()



    # === Fetch Documents from both DBs ===
def eligibilityAgent(eligibility_criteria):
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
    You are an expert compliance agent evaluating whether ConsultAdd is eligible to apply for a government RFP.

    Below are the **Eligibility Criteria** extracted from the RFP:
    {eligibility_criteria}

    Your task:
    1. Evaluate the criteria against the company profile.
    2. Use the **RFP context** and **Company Profile** to support your reasoning.
    3. Mark each criterion as ✅ (met) or ❌ (unmet).
    4. If some information is not clearly available, assume "likely met" if the context is relevant.

    RFP Context:
    {{rfp_context}}

    Company Profile Context:
    {{company_context}}

    Your Answer (Start with 'Yes' or 'No', then give reasoning):
    Is ConsultAdd eligible to apply for this RFP?
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