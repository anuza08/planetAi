# 🌌 Q&A Planet


https://github.com/user-attachments/assets/0d39e5e6-9658-4cdd-9782-8d08c6fdaaff

<img src="https://github.com/user-attachments/assets/b276b74b-c6be-49e2-9f41-39b655f615fa" alt="Screenshot 2024-10-27 212211" width="500"/>
<img src="https://github.com/user-attachments/assets/949a21cf-6e9e-48d0-bc7c-8873e0e70987" alt="Screenshot 2024-10-27 212211" width="500"/>
<img src="https://github.com/user-attachments/assets/34c9f975-7401-409e-9479-eb04caf239c1" alt="Screenshot 2024-10-27 212211" width="500"/>
<img src="https://github.com/user-attachments/assets/3cae8018-02c1-4700-bd3b-acf4918a922a" alt="Screenshot 2024-10-27 212211" width="500"/>

## Added loader for enhancing user experience
<img src="https://github.com/user-attachments/assets/a0a24373-7505-444a-b3a7-492dad361b08" alt="Screenshot 2024-10-27 212211" width="500"/>


**Q&A Planet** is a AI platform where you can upload any document and ask question related to it

In today's digital world, we often need to extract information from large documents quickly and accurately. Manually reading through lengthy PDFs to find answers to specific questions is time-consuming and inefficient. This system solves that problem by allowing users to:

Upload PDF documents

Ask natural language questions about the content

Get accurate answers extracted by an AI model

<h2>Solution Overview</h2>
This application uses the FLAN-T5 large language model to provide question-answering capabilities for uploaded PDF documents. The system:

Processes uploaded PDFs to extract text

Creates searchable vector embeddings of the content

Uses semantic search to find relevant passages

Generates accurate answers to user questions

**Key Features**
PDF Upload: Users can upload any PDF document

Document Management: System stores and organizes uploaded documents

Question Answering: Users can ask natural language questions about document content

Conversation History: All questions and answers are saved for future reference

Fast Semantic Search: Uses FAISS for efficient vector similarity search

**Technology Stack**
<h2>Backend</h2>
FastAPI: Python web framework for building the API

PyMuPDF (fitz): PDF text extraction library

SQLAlchemy: ORM for database operations

FAISS: Vector similarity search library from Facebook AI

<h2>AI/ML Components</h2>
FLAN-T5-large: Google's instruction-tuned language model for question answering

HuggingFace Transformers: For model loading and inference

Sentence Transformers: For creating document embeddings (all-MiniLM-L6-v2 model)

<h2>Infrastructure</h2>
SQLite: Lightweight database for storing documents and QA pairs

PyTorch: Deep learning framework for model inference

CUDA: GPU acceleration (if available)
