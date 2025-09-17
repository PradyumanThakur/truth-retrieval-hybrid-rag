from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import main
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change "*" to specific domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    claim: str
    top_n: int = 10

@app.post("/factcheck")
async def factcheck(request: ClaimRequest):
    result = main(request.claim, top_n=request.top_n)
    return result
