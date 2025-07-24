import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import json

load_dotenv()

# Initialize components
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf-rag-index"
index = pc.Index(index_name)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def diagnose_pinecone_retrieval():
    """Diagnose issues with Pinecone retrieval"""
    print("=== PINECONE RETRIEVAL DIAGNOSIS ===")
    
    # 1. Check index stats
    print("\n1. INDEX STATISTICS:")
    try:
        stats = index.describe_index_stats()
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Index dimension: {stats.get('dimension', 'Unknown')}")
        print(f"   Index fullness: {stats.get('index_fullness', 'Unknown')}")
        
        if stats.get('total_vector_count', 0) == 0:
            print("   [ERROR] NO VECTORS FOUND! Your PDF data hasn't been uploaded.")
            return False
        else:
            print(f"   [OK] Found {stats.get('total_vector_count')} vectors")
    except Exception as e:
        print(f"   [ERROR] Error accessing index: {e}")
        return False
    
    # 2. Test embedding generation
    print("\n2. EMBEDDING GENERATION TEST:")
    test_query = "What are your programming skills?"
    try:
        query_embedding = embedder.encode([test_query]).tolist()[0]
        print(f"   [OK] Generated embedding for: '{test_query}'")
        print(f"   Embedding dimension: {len(query_embedding)}")
    except Exception as e:
        print(f"   [ERROR] Error generating embedding: {e}")
        return False
    
    # 3. Test Pinecone query
    print("\n3. PINECONE QUERY TEST:")
    try:
        result = index.query(
            vector=query_embedding, 
            top_k=5, 
            include_metadata=True,
            include_values=False
        )
        
        print(f"   Query returned {len(result.get('matches', []))} matches")
        
        if not result.get('matches'):
            print("   [ERROR] NO MATCHES FOUND!")
            print("   This could mean:")
            print("     - Your PDF wasn't uploaded properly")
            print("     - Embedding model mismatch")
            print("     - Query vector dimension mismatch")
            return False
        
        print("   [OK] Found matches:")
        for i, match in enumerate(result['matches'][:3]):
            score = match.get('score', 0)
            text = match.get('metadata', {}).get('text', 'No text')[:100]
            print(f"     Match {i+1}: Score={score:.3f}, Text='{text}...'")
            
    except Exception as e:
        print(f"   [ERROR] Error querying Pinecone: {e}")
        return False
    
    # 4. Test different query types
    print("\n4. TESTING DIFFERENT QUERY TYPES:")
    test_queries = [
        "programming skills",
        "education background", 
        "work experience",
        "projects",
        "contact information"
    ]
    
    for query in test_queries:
        try:
            query_emb = embedder.encode([query]).tolist()[0]
            result = index.query(vector=query_emb, top_k=3, include_metadata=True)
            matches = len(result.get('matches', []))
            best_score = result['matches'][0]['score'] if result.get('matches') else 0
            print(f"   '{query}': {matches} matches, best score: {best_score:.3f}")
        except Exception as e:
            print(f"   '{query}': Error - {e}")
    
    return True

def check_pdf_upload():
    """Check if PDF was uploaded correctly"""
    print("\n=== PDF UPLOAD CHECK ===")
    
    pdf_path = "myowndatafortrainingchatbot.pdf"
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}")
        print("Make sure the PDF file is in the project root directory")
        return False
    
    print(f"[OK] PDF file found: {pdf_path}")
    
    # Check file size
    file_size = os.path.getsize(pdf_path)
    print(f"File size: {file_size} bytes ({file_size/1024:.1f} KB)")
    
    if file_size < 1000:  # Less than 1KB is suspicious
        print("[WARNING] File seems very small - might be empty or corrupted")
    
    return True

def suggest_fixes():
    """Suggest fixes for common issues"""
    print("\n=== SUGGESTED FIXES ===")
    
    print("1. RE-UPLOAD YOUR PDF:")
    print("   Run: uv run python upload_pdf.py")
    
    print("\n2. VERIFY ENVIRONMENT VARIABLES:")
    print("   Check your .env file has:")
    print("   - PINECONE_API_KEY=your_key_here")
    print("   - GROQ_API_KEY=your_key_here")
    
    print("\n3. IMPROVE RETRIEVAL QUALITY:")
    print("   - Increase top_k from 5 to 10")
    print("   - Lower similarity threshold")
    print("   - Use better chunking strategy")
    
    print("\n4. CHECK EMBEDDING MODEL CONSISTENCY:")
    print("   - Both upload_pdf.py and app.py use 'all-MiniLM-L6-v2'")
    print("   - Make sure they're exactly the same")

def run_comprehensive_diagnosis():
    """Run all diagnostic tests"""
    print("COMPREHENSIVE PINECONE RETRIEVAL DIAGNOSIS")
    print("=" * 50)
    
    # Check PDF file
    pdf_ok = check_pdf_upload()
    
    # Check Pinecone retrieval
    retrieval_ok = diagnose_pinecone_retrieval()
    
    # Suggest fixes if issues found
    if not pdf_ok or not retrieval_ok:
        suggest_fixes()
    else:
        print("\n[SUCCESS] ALL CHECKS PASSED!")
        print("Your Pinecone retrieval should be working correctly.")

if __name__ == "__main__":
    run_comprehensive_diagnosis()
