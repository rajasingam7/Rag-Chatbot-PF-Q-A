run the requirement.txt =====> pip install -r requirements.txt
Add your groq api key
🔹 Project Title

RAG-Based PDF Question Answering Chatbot using Groq & LangChain

🔹 What this project does

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about PDF documents and receive accurate, context-aware answers. Instead of relying only on an LLM’s memory, it retrieves relevant content from uploaded PDFs and uses that context to generate responses.

🔹 Why this project is needed

Large Language Models can hallucinate when answering questions. This project solves that by:

grounding answers in actual document content

improving accuracy and trustworthiness

enabling enterprise use cases like research analysis, compliance review, and knowledge assistants

🔹 High-level architecture

PDF Ingestion

PDFs are loaded from a local folder (research_papers)

Text is extracted using a PDF loader

Text Chunking

Documents are split into smaller overlapping chunks

This improves retrieval accuracy and embedding quality

Vector Embedding

Each text chunk is converted into a vector using HuggingFace embeddings

Embeddings represent semantic meaning

Vector Storage (FAISS)

Vectors are stored in a FAISS index for fast similarity search

User Query Flow

User enters a question in the Streamlit UI

Relevant document chunks are retrieved using vector similarity

Answer Generation (Groq LLM)

Retrieved context is passed to a Groq-hosted Llama 3.1 model

The model generates an answer strictly based on the provided context

Result Display

Final answer is shown

Retrieved document chunks are displayed for transparency

🔹 Key technologies used

Streamlit – Interactive web UI

LangChain (Classic API) – RAG orchestration

Groq LLM – Fast inference using Llama 3.1

HuggingFace Embeddings – Text vectorization

FAISS – High-performance vector similarity search

Python – Core implementation

🔹 Key features

Context-aware document Q&A

Fast response time using Groq inference

Prevents hallucinations by restricting answers to document context

Scalable vector search using FAISS

Simple and clean Streamlit interface

🔹 Real-world use cases

Research paper analysis

Internal knowledge base chatbot

Compliance and audit document review

Legal and policy document Q&A

Educational content exploration

🔹 What makes this project strong

Uses RAG, not plain LLM prompting

Handles large documents efficiently

Clean separation of ingestion, retrieval, and generation

Easily extendable to chat history, OCR, or cloud deployment

🔹 Future enhancements (optional)

Multi-PDF upload via UI

Chat history support

OCR for scanned PDFs

Source citation with page numbers

Cloud deployment (AWS / GCP / Azure)
