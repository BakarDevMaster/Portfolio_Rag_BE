from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import redis.asyncio as redis
import json

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

# Upstash Redis configuration
REDIS_URL = "redis://default:AX7DAAIjcDFiMDBmNzA4NTE2MGQ0MDMzYTg1OThmYjRmNTU5MjAzOXAxMA@popular-dane-32451.upstash.io:6379"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Helper functions for conversation history
async def get_conversation_history(session_id):
    data = await redis_client.get(session_id)
    if data:
        return json.loads(data)
    return []

async def save_conversation_history(session_id, history):
    await redis_client.set(session_id, json.dumps(history), ex=86400)  # 1 day expiry

# Pydantic model for query input
class QueryRequest(BaseModel):
    question: str
    session_id: str  # New field for session tracking

# Query endpoint
@app.post("/query")
async def query_agent(request: QueryRequest):
    try:
        user_input = request.question
        session_id = request.session_id
        if not user_input or not session_id:
            raise HTTPException(status_code=400, detail="Please provide a question and session_id")

        # Retrieve conversation history
        history = await get_conversation_history(session_id)

        # Embed and retrieve context as before
        query_embedding = embedder.encode([user_input]).tolist()[0]
        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        retrieved_docs = [match['metadata']['text'] for match in result['matches']]
        context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant info found."

        # Build messages for LLM
        messages = history + [
            {"role": "system", "content": f"Answer based on this context: {context}"},
            {"role": "user", "content": user_input}
        ]

        response = await client.chat.completions.create(
            model="llama3-8b-8192",  # Verify with Groq docs; adjust if needed
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        answer = response.choices[0].message.content

        # Update conversation history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        await save_conversation_history(session_id, history)

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}