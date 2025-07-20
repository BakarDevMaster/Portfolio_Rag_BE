from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Portfolio AI Query Backend")

# Configure Groq client
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Pinecone client and index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-rag-index"
index = pc.Index(index_name)
print(f"Index {index_name} loaded with stats: {index.describe_index_stats()}")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pydantic model for query input
class QueryRequest(BaseModel):
    question: str

# Query endpoint
@app.post("/query")
async def query_agent(request: QueryRequest):
    try:
        user_input = request.question
        if not user_input:
            raise HTTPException(status_code=400, detail="Please provide a question")

        query_embedding = embedder.encode([user_input]).tolist()[0]
        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        retrieved_docs = [match['metadata']['text'] for match in result['matches']]
        context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant info found."

        messages = [
            {"role": "system", "content": f"Answer based on this context: {context}"},
            {"role": "user", "content": user_input}
        ]

        response = await client.chat.completions.create(
            model="llama3-8b-8192",  # Verify with Groq docs; adjust if needed
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return {"response": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}