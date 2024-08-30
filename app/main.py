import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .llm_utils import query_llm
from .pinecone_utils import generate_embedding, query_pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Determine the origin allowed based on environment
environment = os.getenv("ENVIRONMENT", "local")
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 300
    temperature: float = 0.5
    top_p: float = 0.95

@app.post("/query/")
async def handle_query(request: QueryRequest):
    query_embedding = generate_embedding(request.query)
    
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Error generating query embedding.")
    
    pinecone_response = query_pinecone(query_embedding)
    
    if not pinecone_response:
        raise HTTPException(status_code=404, detail="No relevant data found in Pinecone.")

    llm_response = query_llm(request.query, pinecone_response, max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p)
    
    return {"response": llm_response}
