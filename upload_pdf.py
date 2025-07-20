from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("pdf-rag-index")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Upload to Pinecone
pdf_path = "myowndatafortrainingchatbot.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = split_text(text)
embeddings = embedder.encode(chunks).tolist()
vectors = [(f"id{i}", embedding, {"text": chunk}) for i, (embedding, chunk) in enumerate(zip(embeddings, chunks))]
index.upsert(vectors=vectors)
print(f"Uploaded {len(chunks)} chunks to Pinecone.")