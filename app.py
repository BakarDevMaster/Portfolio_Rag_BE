from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import json
from upstash_redis.asyncio import Redis

load_dotenv()

app = FastAPI(title="Portfolio AI Query Backend")

# Configure Groq client
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Pinecone client and index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-rag-index"  # or your actual index name
index = pc.Index(index_name)
# print(f"Index {index_name} loaded with stats: {index.describe_index_stats()}")

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Upstash Redis configuration (HTTP API, async)
redis = Redis(
    url="https://popular-dane-32451.upstash.io",
    token="AX7DAAIjcDFiMDBmNzA4NTE2MGQ0MDMzYTg1OThmYjRmNTU5MjAzOXAxMA"
)

# Helper functions for conversation history
async def get_conversation_history(session_id):
    data = await redis.get(session_id)
    if data:
        return json.loads(data)
    return []

async def save_conversation_history(session_id, history):
    await redis.set(session_id, json.dumps(history))

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
            print("Missing question or session_id")
            raise HTTPException(status_code=400, detail="Please provide a question and session_id")

        # Retrieve conversation history
        try:
            print("Retrieving conversation history from Redis...")
            history = await get_conversation_history(session_id)
            print("History:", history)
        except Exception as e:
            print("Redis error:", e)
            raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")

        # Embed and retrieve context as before
        try:
            print("Embedding user input...")
            query_embedding = embedder.encode([user_input]).tolist()[0]
            print("Querying Pinecone...")
            result = index.query(vector=query_embedding, top_k=10, include_metadata=True)  # Increased to 10 for better coverage
            print("Querying Pinecone with:", user_input)
            
            # Filter matches by similarity threshold
            similarity_threshold = 0.25  # Adjust based on your needs
            relevant_matches = [match for match in result['matches'] if match['score'] >= similarity_threshold]
            
            print(f"Found {len(result['matches'])} total matches, {len(relevant_matches)} above threshold {similarity_threshold}")
            
            if relevant_matches:
                retrieved_docs = [match['metadata']['text'] for match in relevant_matches]
                
                # Format context more clearly with scores
                context = "Here is the relevant information about me (from my portfolio):\n\n"
                for i, (doc, match) in enumerate(zip(retrieved_docs, relevant_matches), 1):
                    score = match['score']
                    context += f"[Relevance: {score:.2f}] {doc}\n\n"
            else:
                context = "No highly relevant information found in my portfolio for this specific question."
                retrieved_docs = []
            
            print("Retrieved context:", context)
        except Exception as e:
            print("Pinecone error:", e)
            raise HTTPException(status_code=500, detail=f"Pinecone error: {str(e)}")

        # Build messages for LLM
        system_prompt = f"""You are Muhammad Abubakar speaking directly to the person. Always respond in first person as if you are me talking to them.
        Use a friendly, professional tone and speak directly to the person. Never say phrases like 'based on the portfolio' or 'the developer's'.
        Instead, use 'I', 'my', 'me' etc. For example, if asked about skills, say 'I am skilled in...' or 'My expertise includes...'
        
        IMPORTANT: Use the context provided below to answer questions about me. If the context doesn't contain enough information to fully answer the question, acknowledge what you know from the context and politely mention that you might need more specific information.
        
        Context about me:
        {context}
        
        If no relevant context is provided, politely explain that you don't have specific information about that topic in your current knowledge base and suggest they ask about my technical skills, projects, or experience."""
        
        messages = history + [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # Call LLM
        try:
            print("Calling Groq/OpenAI LLM...")
            response = await client.chat.completions.create(
                model="llama3-8b-8192",  # Verify with Groq docs; adjust if needed
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            print("LLM response:", answer)
        except Exception as e:
            print("Groq/OpenAI error:", e)
            raise HTTPException(status_code=500, detail=f"Groq/OpenAI error: {str(e)}")

        # Update conversation history
        try:
            print("Saving conversation history to Redis...")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
            await save_conversation_history(session_id, history)
            print("History saved.")
        except Exception as e:
            print("Redis save error:", e)
            raise HTTPException(status_code=500, detail=f"Redis save error: {str(e)}")

        return {"response": answer}

    except Exception as e:
        print("General error:", e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}