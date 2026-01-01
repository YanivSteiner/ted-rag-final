from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_lib import TedRagSystem
import uvicorn

# Initialize the application
app = FastAPI(title="TED RAG Assistant")

# System Configuration (can be adjusted here)
CHUNK_SIZE = 1000
OVERLAP_RATIO = 0.2
TOP_K = 5

# Initialize the RAG system once when the server starts
# Note: limit=20 loads only the first 20 talks to save time and budget on the first run.
# To process the full dataset, set limit=None (Be aware of API costs!).
rag_system = TedRagSystem("ted_talks_en.csv", chunk_size=CHUNK_SIZE, overlap_ratio=OVERLAP_RATIO, top_k=TOP_K)
#rag_system.load_and_process_data(limit=None) 

# Request model for the query (according to assignment requirements)
class QueryRequest(BaseModel):
    question: str

@app.get("/api/stats")
def get_stats():
    """Returns the configuration of the RAG system."""
    return {
        "chunk_size": rag_system.chunk_size,
        "overlap_ratio": rag_system.overlap_ratio,
        "top_k": rag_system.top_k
    }

@app.post("/api/prompt")
def query_rag(request: QueryRequest):
    """Processes a query using RAG and returns the answer."""
    try:
        result = rag_system.answer_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server on localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)