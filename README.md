RAG Portfolio Assistant

An AI-powered Retrieval-Augmented Generation (RAG) application trained on personal portfolio data. The system allows intelligent querying of your portfolio, resumes, and project documents with contextual awareness. It integrates Pinecone for vector search, Redis for memory management, and is deployed with FastAPI (managed by uv package manager).

🚀 Features

Contextual Question Answering:

Query personal portfolio/resume data.

AI provides answers grounded in your own documents.

Vector Search with Pinecone:

Efficient semantic search over PDF-embedded data.

Fast and scalable retrieval pipeline.

Conversation Memory with Redis:

Keeps track of past queries for contextual dialogue.

Supports multi-turn interactions without losing context.

FastAPI Backend:

RESTful API endpoints for querying and management.

Deployed using uv package manager for modern, lightweight execution.
