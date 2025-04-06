from langchain_community.tools import DuckDuckGoSearchRun

# Web search tool
web_search_tool = DuckDuckGoSearchRun()

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re
from langchain.agents import AgentExecutor

def remove_think_tags(text):
    # This pattern matches anything between <think> and </think>, including newlines
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def market_search_agent():
    # === LLM ===
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rfp_db = FAISS.load_local("./VectorDB/RPF_Uploadded", embedding_model, allow_dangerous_deserialization=True)

    # === Load Company Data Vector DB ===
    company_db = FAISS.load_local("./VectorDB/Company_Uploaded", embedding_model, allow_dangerous_deserialization=True)
    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.7)
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

    query = "Generate a market analysis report for ConsultAdd based on the RFP and company profile."
    rfp_docs = multi_retriever_rfp.get_relevant_documents(query)
    company_docs = multi_retriever_company.get_relevant_documents(query)

    # === Combine Context ===
    rfp_context = "\n".join([doc.page_content for doc in rfp_docs])
    company_context = "\n".join([doc.page_content for doc in company_docs])

    # === Prompt Template for Report Generation ===
    prompt_template = PromptTemplate.from_template(f"""
        You are a Market Research Analyst. Your job is to analyze the current market for ConsultAdd based on the RFP and company profile.
        
        Use the provided context and internet search results to create a structured report.

        **RFP Context:** {{rfp_context}}

        **Company Profile:** {{company_context}}

        **Internet Findings:** {{web_context}}

        **Your Task:**
        - Identify industry trends
        - Analyze competition
        - Identify potential partners
        - Determine risks and opportunities
        - Suggest how ConsultAdd can position itself for this RFP

        Output your answer as a structured report with bullet points and clear headings.
    """)

    # === Define a tool for web search ===
    def search_wrapper(q):
        return web_search_tool.invoke(q)

    tools = [
        Tool(name="RPF Web Search", func=search_wrapper, description="Useful for real-time market analysis for RFPs Analysis."),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    llm.temperature = 0.4

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent.agent,  # underlying agent logic
        tools=tools,
        handle_parsing_errors=True,
        verbose=True
    )

    web_results = agent_executor.invoke({
    "input": f"Search for IT consulting trends in government RFPs for 2024.",
    })

    # === Generate Final Report ===
    report_chain = LLMChain(llm=llm, prompt=prompt_template)
    report = report_chain.invoke({
        "rfp_context": rfp_context,
        "company_context": company_context,
        "web_context": web_results
    })
    cleaned_report = remove_think_tags(report["text"])
    return cleaned_report


print(market_search_agent())
