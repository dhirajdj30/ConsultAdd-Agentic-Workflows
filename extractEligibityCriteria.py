from langchain.document_loaders import PyPDFLoader
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




def extract_eligibility_criteria(rfp_pdf_path: str):
    # Load and split RFP PDF
    loader = PyPDFLoader(rfp_pdf_path)
    docs = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(loader.load())

    # Create vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Define LLM & Prompt
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.2)
    prompt = PromptTemplate.from_template("""
    You are a compliance expert reviewing an RFP.

    Extract the **mandatory eligibility criteria** from the context, including:
    - Required certifications
    - Past performance or years of experience
    - Technical or staffing requirements
    - Disqualifiers
    Return the result as a clean bullet-point list.

    Context:
    {context}

    Question: What are the mandatory eligibility criteria for applying to this RFP?
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
