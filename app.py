from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import spacy
from pydantic import BaseModel
import json
from upstash_redis.asyncio import Redis
import asyncio
import re

load_dotenv()

app = FastAPI(title="Portfolio AI Query Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure Groq client
client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Pinecone client and index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-rag-index-384"  # or your actual index name
index = pc.Index(index_name)
# print(f"Index {index_name} loaded with stats: {index.describe_index_stats()}")

# Load lightweight spaCy model for embeddings (300-d)
# Ensure the model is downloaded:  python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')
TARGET_EMBED_DIM = 384  # Pinecone index dimension

# Portfolio-related keywords and topics for relevance detection
PORTFOLIO_KEYWORDS = [
    'muhammad', 'abubakar', 'portfolio', 'experience', 'skills', 'projects', 'education',
    'work', 'job', 'career', 'developer', 'programming', 'coding', 'software', 'technology',
    'frontend', 'backend', 'fullstack', 'web', 'mobile', 'app', 'application', 'development',
    'javascript', 'python', 'react', 'node', 'database', 'api', 'framework', 'library',
    'github', 'linkedin', 'contact', 'email', 'phone', 'resume', 'cv', 'qualification',
    'certificate', 'degree', 'university', 'college', 'internship', 'freelance', 'client',
    'achievement', 'award', 'recognition', 'expertise', 'proficient', 'familiar', 'knowledge'
]

# Non-portfolio topics that should be declined
NON_PORTFOLIO_TOPICS = [
    'weather', 'news', 'current events', 'politics', 'sports', 'entertainment', 'movies',
    'music', 'food', 'recipes', 'health', 'medical', 'travel', 'geography', 'history',
    'science', 'math', 'physics', 'chemistry', 'biology', 'astronomy', 'philosophy',
    'religion', 'culture', 'language', 'translation', 'joke', 'story', 'game', 'puzzle'
]

# Upstash Redis configuration (HTTP API, async)
redis = Redis(
    url='https://popular-dane-32451.upstash.io',
    token='AX7DAAIjcDFiMDBmNzA4NTE2MGQ0MDMzYTg1OThmYjRmNTU5MjAzOXAxMA'
)

# Helper functions for conversation history
async def get_conversation_history(session_id):
    data = await redis.get(session_id)
    if data:
        return json.loads(data)
    return []

async def save_conversation_history(session_id, history):
    await redis.set(session_id, json.dumps(history))

# Function to clean markdown formatting from response
def clean_markdown_formatting(text: str) -> str:
    """
    Remove markdown formatting from text to ensure plain text output.
    """
    # Remove markdown bullet points
    text = re.sub(r'^\s*[*-]\s+', '', text, flags=re.MULTILINE)
    # Remove markdown numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove markdown code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    return text

# Streaming generator function
async def stream_response(text: str):
    """
    Stream text character by character with a small delay.
    """
    for char in text:
        yield f"data: {json.dumps({'char': char})}\n\n"
        await asyncio.sleep(0.05)  # 50ms delay between characters
    yield f"data: {json.dumps({'done': True})}\n\n"

# Function to check if a question is portfolio-related
import difflib


def _fuzzy_in(text: str, terms: list[str], cutoff: float = 0.85) -> bool:
    """Return True if any term appears in text with fuzzy ratio >= cutoff"""
    words = text.split()
    for term in terms:
        # fast exact check first
        if term in text:
            return True
        # fuzzy check each word in question against the term
        for w in words:
            if difflib.SequenceMatcher(None, w, term).ratio() >= cutoff:
                return True
    return False


def is_portfolio_related(question: str) -> bool:
    """Determine if a question is related to Muhammad Abubakar's portfolio with typo tolerance."""
    question_lower = question.lower()

    # Decline if it matches any non-portfolio topic (exact or fuzzy)
    if _fuzzy_in(question_lower, NON_PORTFOLIO_TOPICS, cutoff=0.9):
        return False

    # Accept if it matches portfolio keywords (exact or fuzzy)
    if _fuzzy_in(question_lower, PORTFOLIO_KEYWORDS, cutoff=0.8):
        return True

    # Fallback to pattern heuristics
    portfolio_patterns = [
        'tell me about', 'what is your', 'what are your', 'how do you',
        'can you', 'do you have', 'what technologies', 'what languages',
        'your background', 'your experience', 'about you', 'who are you'
    ]
    
    for pattern in portfolio_patterns:
        if pattern in question_lower:
            return True
    
    # If no clear indicators, default to False (decline)
    return False

# Pydantic model for query input
class QueryRequest(BaseModel):
    question: str
    session_id: str  # New field for session tracking
    stream: bool = True  # New field to enable/disable streaming

# Query endpoint
@app.post("/query")
async def query_agent(request: QueryRequest):
    try:
        user_input = request.question
        session_id = request.session_id
        if not user_input or not session_id:
            print("Missing question or session_id")
            raise HTTPException(status_code=400, detail="Please provide a question and session_id")
        
        # Check if the question is portfolio-related
        if not is_portfolio_related(user_input):
            print(f"Non-portfolio question detected: {user_input}")
            decline_message = (
                "I'm sorry, but I'm specifically designed to answer questions about Muhammad Abubakar's "
                "portfolio, professional experience, skills, and projects. I can help you learn about my "
                "technical expertise, work experience, education, or any specific projects I've worked on. "
                "Please feel free to ask me anything related to my professional background!"
            )
            
            # Still save to conversation history for context
            try:
                history = await get_conversation_history(session_id)
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": decline_message})
                await save_conversation_history(session_id, history)
            except Exception as e:
                print(f"Error saving decline message to history: {e}")
            
            return {"response": decline_message}

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
            vec = nlp(user_input).vector
            # Pad to 384-d (append zeros)
            if vec.shape[0] < TARGET_EMBED_DIM:
                padding = [0.0] * (TARGET_EMBED_DIM - vec.shape[0])
                vec = vec.tolist() + padding
            else:
                vec = vec[:TARGET_EMBED_DIM].tolist()
            query_embedding = vec
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
        
        CRITICAL RESTRICTION: You must ONLY answer questions related to my portfolio, professional experience, skills, projects, education, and career. 
        Do NOT provide information about general topics, current events, weather, entertainment, or any non-portfolio related subjects.
        
        IMPORTANT FORMATTING: Respond in plain text format. Do NOT use markdown formatting, bullet points (*), numbered lists, or any special characters for formatting. Write in natural, conversational sentences and paragraphs. Use simple line breaks for separation if needed.
        
        IMPORTANT: Use ONLY the context provided below to answer questions about me. If the context doesn't contain enough information to fully answer the question, acknowledge what you know from the context and politely mention that you might need more specific information about my professional background.
        
        Context about me:
        {context}
        
        If no relevant context is provided, politely explain that you don't have specific information about that topic in your current knowledge base and suggest they ask about my technical skills, projects, work experience, or education."""
        
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

        # Clean markdown formatting from the response
        cleaned_answer = clean_markdown_formatting(answer)
        print("Cleaned response:", cleaned_answer)

        # Update conversation history
        try:
            print("Saving conversation history to Redis...")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": cleaned_answer})
            await save_conversation_history(session_id, history)
            print("History saved.")
        except Exception as e:
            print("Redis save error:", e)
            raise HTTPException(status_code=500, detail=f"Redis save error: {str(e)}")

        # Return streaming or regular response based on request
        if request.stream:
            return StreamingResponse(
                stream_response(cleaned_answer),
                media_type="text/event-stream" # Correct media type for SSE
            )
        else:
            return {"response": cleaned_answer}

    except Exception as e:
        print("General error:", e)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}