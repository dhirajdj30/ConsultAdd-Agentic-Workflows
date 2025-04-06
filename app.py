from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Import your agent modules
from eligiblityAgent import eligibilityAgent
from checklistAgent import checklistAgent
from riskAgent import risk_analysis_agent
from extractEligibityCriteria import extract_eligibility_criteria
app = FastAPI(
    title="RFP Automation Backend",
    description="ConsultAdd's GenAI Assistant for RFP Analysis",
    version="1.0"
)

# Input schema for all endpoints
class RFPRequest(BaseModel):
    query: str = "analyze this RFP for the task"

# 1. Eligibility Check Endpoint
@app.post("/eligibility")
def eligibility_check():
    try:
        eligibility_criteria = extract_eligibility_criteria("./Data/ineligible.pdf")
        result = eligibilityAgent(eligibility_criteria)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Checklist Generation Endpoint
@app.post("/checklist")
def checklist_gen():
    try:
        result = checklistAgent()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Contract Risk Analysis Endpoint
@app.post("/risks")
def risk_analysis():
    try:
        result = risk_analysis_agent()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to ConsultAdd's RFP Analysis API!"}
