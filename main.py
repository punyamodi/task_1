from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_agent  # Importing from agent.py

app = FastAPI()

# Define request model
class QueryRequest(BaseModel):
    query: str
    human_input: str = ""

@app.post("/process-query")
async def process_query(request: QueryRequest):
    try:
        final_state = run_agent(request.query, request.human_input)
        return {"response": final_state.get("final_response", "N/A")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def home():
    return {"message": "Agent API is running"}
