from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 
# Load retriever
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local("./outputs/faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# Load model from HuggingFace
llm = llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0.7
)

# Custom prompt template
prompt_template = PromptTemplate.from_template("""
You are a compliance assistant reviewing a U.S. government RFP for a company called ConsultAdd.

Based on the retrieved content below, determine whether the company is **eligible** to apply for the RFP. 

Focus on:
- Required certifications or registrations (e.g. SAM, state registration)
- Past performance requirements
- Any disqualifying clauses
- Any missing mandatory qualifications

Give a direct Yes/No answer and justify it clearly.

Context:
{context}

Question:
Is ConsultAdd eligible to apply for this RFP? Why or why not?
""")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# Run the agent
query = "Is ConsultAdd eligible to apply for this RFP?"
# result = qa_chain.run(query)
result = qa_chain.invoke({"query": query})

print("🤖 EligibilityAgent Response:\n", result["result"])
