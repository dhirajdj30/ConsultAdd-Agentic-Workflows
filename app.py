from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Import your agent modules
from eligiblityAgent import eligibilityAgent
from checklistAgent import checklistAgent
from riskAgent import risk_analysis_agent
from extractEligibityCriteria import extract_eligibility_criteria
from marketSearchAgent import market_search_agent
app = FastAPI(
    title="RFP Automation Backend",
    description="ConsultAdd's GenAI Assistant for RFP Analysis",
)

# Input schema for all endpoints
class RFPRequest(BaseModel):
    query: str = "analyze this RFP for the task"


# 1. Eligibility Check Endpoint
@app.post("/eligibility")
def eligibility_check():
    try:
        eligibility_criteria = extract_eligibility_criteria()
        result = eligibilityAgent(eligibility_criteria)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Checklist Generation Endpoint
@app.post("/checklist")
def checklist_gen():
    try:
        path = "./VectorDB/RPF_Uploadded"
        result = checklistAgent(path)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Contract Risk Analysis Endpoint
@app.post("/risks")
def risk_analysis():
    try:
        path = "./VectorDB/RPF_Uploadded"
        result = risk_analysis_agent(path)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-market-report")
async def generate_market_report():
    try:
        report = market_search_agent()
        return {"status": "success", "report": report}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to ConsultAdd's RFP Analysis API!"}
