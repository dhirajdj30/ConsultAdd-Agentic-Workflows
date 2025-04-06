from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import re
def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()




def extract_eligibility_criteria():
    # Load and split RFP PDF
    loader = PyPDFLoader("Downloaded/uploaded_rfp.pdf")
    docs = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(loader.load())

    # Create vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Define LLM & Prompt
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.)
    prompt = PromptTemplate.from_template("""
   You are a compliance expert using Generative AI and Retrieval-Augmented Generation (RAG) to analyze an RFP and extract mandatory eligibility criteria. Your task is to automate and simplify the RFP analysis process, ensuring accuracy and efficiency.
    **Extract the mandatory eligibility criteria** from the context, including:
    - **Required certifications** (e.g., ISO, CMMI, 8(a), etc.)
    - **Past performance or years of experience** (e.g., "5+ years of experience in IT consulting")
    - **Technical or staffing requirements** (e.g., "Must have a team of 10 certified developers")
    - **Disqualifiers** (e.g., "Cannot have any unresolved legal disputes")

    **Output Requirements:**
    1. Return the criteria as a clean, organized bullet-point list.
    2. Highlight any ambiguous or unclear criteria that require further clarification.
    3. Identify potential gaps between the criteria and ConsultAdd's company profile (if provided).
    4. Provide a brief summary of the findings to support decision-making.

    **Context:**
    {context}

    **Question:** What are the mandatory eligibility criteria for applying to this RFP?
    """)

    # Run RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    result = qa_chain.invoke({"query": "What are the mandatory eligibility criteria?"})
    cleaned_eligblity_criteria = remove_think_tags(result["result"])
    return cleaned_eligblity_criteria
