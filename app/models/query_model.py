from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 300
    temperature: float = 0.5
    top_p: float = 0.95
