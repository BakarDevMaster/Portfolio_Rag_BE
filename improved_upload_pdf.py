from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import spacy
import os
from dotenv import load_dotenv
import re

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-rag-index-384"
index = pc.Index(index_name)
nlp = spacy.load('en_core_web_md')
TARGET_EMBED_DIM = 384  # must match Pinecone index dimension

def extract_text_from_pdf(pdf_path):
    """Extract and clean text from PDF"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single newline
    text = text.strip()
    
    return text

def improved_split_text(text, chunk_size=300, chunk_overlap=50):
    """Improved text splitting with better chunk boundaries"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        keep_separator=True
    )
    chunks = text_splitter.split_text(text)
    
    # Clean and filter chunks
    cleaned_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 20:  # Only keep chunks with meaningful content
            cleaned_chunks.append(chunk)
    
    return cleaned_chunks

def clear_existing_vectors():
    """Clear existing vectors from the index"""
    try:
        # Get all vector IDs (this is a simplified approach)
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"Clearing {stats.total_vector_count} existing vectors...")
            # Delete all vectors (you might need to implement this differently based on your setup)
            index.delete(delete_all=True)
            print("Existing vectors cleared.")
        else:
            print("No existing vectors to clear.")
    except Exception as e:
        print(f"Error clearing vectors: {e}")

def upload_improved_pdf():
    """Upload PDF with improved processing"""
    print("=== IMPROVED PDF UPLOAD ===")
    
    # Clear existing vectors first
    clear_existing_vectors()
    
    # Extract and process text
    pdf_path = "myowndatafortrainingchatbot.pdf"
    print(f"Extracting text from: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    
    # Split into improved chunks
    print("Splitting text into chunks...")
    chunks = improved_split_text(text, chunk_size=300, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks")
    
    # Show sample chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk[:100]}...")
    
    # Generate embeddings using spaCy and pad to 384-dim
    print("\nGenerating embeddings...")
    embeddings = []
    for chunk in chunks:
        vec = nlp(chunk).vector
        if vec.shape[0] < TARGET_EMBED_DIM:
            vec = vec.tolist() + [0.0] * (TARGET_EMBED_DIM - vec.shape[0])
        else:
            vec = vec[:TARGET_EMBED_DIM].tolist()
        embeddings.append(vec)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create vectors with better metadata
    vectors = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        vectors.append((
            f"chunk_{i}",  # Better ID format
            embedding,
            {
                "text": chunk,
                "chunk_id": i,
                "source": "portfolio_pdf",
                "length": len(chunk)
            }
        ))
    
    # Upload to Pinecone
    print("Uploading to Pinecone...")
    try:
        index.upsert(vectors=vectors)
        print(f"[SUCCESS] Successfully uploaded {len(chunks)} chunks to Pinecone!")
        
        # Verify upload
        stats = index.describe_index_stats()
        print(f"Index now contains {stats.total_vector_count} vectors")
        
    except Exception as e:
        print(f"[ERROR] Error uploading to Pinecone: {e}")

def test_retrieval():
    """Test the improved retrieval"""
    print("\n=== TESTING IMPROVED RETRIEVAL ===")
    
    test_queries = [
        "What are your programming skills?",
        "Tell me about your education",
        "What projects have you worked on?",
        "How can I contact you?"
    ]
    
    for query in test_queries:
        print(f"\nTesting: '{query}'")
        try:
            vec = nlp(query).vector
            if vec.shape[0] < TARGET_EMBED_DIM:
                vec = vec.tolist() + [0.0]*(TARGET_EMBED_DIM - vec.shape[0])
            else:
                vec = vec[:TARGET_EMBED_DIM].tolist()
            query_embedding = vec
            result = index.query(
                vector=query_embedding, 
                top_k=5, 
                include_metadata=True
            )
            
            print(f"Found {len(result.matches)} matches:")
            for i, match in enumerate(result.matches[:3]):
                score = match.score
                text = match.metadata['text'][:100]
                print(f"  {i+1}. Score: {score:.3f} - {text}...")
                
        except Exception as e:
            print(f"Error testing query: {e}")

if __name__ == "__main__":
    upload_improved_pdf()
    test_retrieval()
