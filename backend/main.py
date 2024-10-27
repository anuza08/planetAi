import os  # Import os module
import logging  
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF for PDF handling
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging (write logs to a file and stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler()  # Also print logs to stdout
    ]
)

logger = logging.getLogger(__name__)  # Initialize the logger for this module

# Initialize FastAPI app
app = FastAPI()

# CORS settings
origins = [
    "http://localhost:5175",
    "http://localhost:3000",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary storage for PDFs and metadata
documents = {}
last_document_id = None  # Track the last document ID

class QuestionRequest(BaseModel):
    document_id: int = None  # Optional, uses the last uploaded document if not provided
    question: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global last_document_id  # Access global state
    pdf_text = ""

    try:
        pdf_content = await file.read()  # Read uploaded PDF content
        logger.info(f"Uploaded file size: {len(pdf_content)} bytes")

        # Extract text using PyMuPDF
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()

        logger.info(f"Extracted text length: {len(pdf_text)} characters")

    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to read the PDF document.")

    if pdf_text:
        doc_id = len(documents) + 1
        documents[doc_id] = pdf_text
        last_document_id = doc_id
        logger.info(f"Document uploaded with ID: {doc_id}")
        return {"document_id": doc_id}
    else:
        logger.warning("No text extracted from PDF.")
        raise HTTPException(status_code=400, detail="No text found in the PDF document.")

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    global last_document_id

    # Use the last uploaded document if document_id is not provided
    doc_id = request.document_id or last_document_id
    logger.debug(f"Received request with document_id: {doc_id} and question: {request.question}")

    # Check if the document exists
    if not doc_id or doc_id not in documents:
        available_ids = list(documents.keys())
        logger.error(f"Document with ID {doc_id} not found. Available IDs: {available_ids}")
        raise HTTPException(status_code=404, detail=f"Document not found. Available document IDs: {available_ids}")

    pdf_text = documents[doc_id]

    try:
        # Split the text into chunks
        splitter = RecursiveCharacterTextSplitter()
        text_chunks = splitter.split_text(pdf_text)

        if not text_chunks:
            logger.error("No chunks created from PDF text.")
            raise HTTPException(status_code=500, detail="No text chunks created from the document.")

        # Process each chunk with the question-answering model
        answers = []
        for chunk in text_chunks:
            result = qa_pipeline(question=request.question, context=chunk)
            answers.append(result["answer"])

        # Combine answers or choose the best one
        final_answer = " ".join(answers)  # Simplified combining, or you could use more logic to select the best answer

        return {"answer": final_answer}

    except Exception as e:
        logger.exception("Error during question processing")
        raise HTTPException(status_code=500, detail="An error occurred while processing the question.")