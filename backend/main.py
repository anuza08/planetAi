# import os  # Import os module
import logging  
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
# import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  
# from langchain.prompts import PromptTemplate
# from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), 
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger(__name__) 


app = FastAPI()

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


documents = {}
last_document_id = None  

class QuestionRequest(BaseModel):
    document_id: int = None  
    question: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global last_document_id  
    pdf_text = ""

    try:
        pdf_content = await file.read()  
        logger.info(f"Uploaded file size: {len(pdf_content)} bytes")

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

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

rerank_model = SentenceTransformer('all-MiniLM-L6-v2')  

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    global last_document_id

   
    doc_id = request.document_id or last_document_id
    logger.debug(f"Received request with document_id: {doc_id} and question: {request.question}")

   
    if not doc_id or doc_id not in documents:
        available_ids = list(documents.keys())
        logger.error(f"Document with ID {doc_id} not found. Available IDs: {available_ids}")
        raise HTTPException(status_code=404, detail=f"Document not found. Available document IDs: {available_ids}")

    pdf_text = documents[doc_id]

    try:
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)  # Adjust chunk size if necessary
        text_chunks = splitter.split_text(pdf_text)

        if not text_chunks:
            logger.error("No chunks created from PDF text.")
            raise HTTPException(status_code=500, detail="No text chunks created from the document.")

       
        candidate_answers = []
        for chunk in text_chunks:
            result = qa_pipeline(question=request.question, context=chunk)
            candidate_answers.append((result["answer"], result["score"], chunk))

      
        best_answer = max(candidate_answers, key=lambda x: x[1])

       
        query_embedding = rerank_model.encode(request.question, convert_to_tensor=True)
        answer_embeddings = [rerank_model.encode(answer[0], convert_to_tensor=True) for answer in candidate_answers]
        rerank_scores = [util.pytorch_cos_sim(query_embedding, ans_emb).item() for ans_emb in answer_embeddings]

       
        best_rerank_answer = candidate_answers[rerank_scores.index(max(rerank_scores))][0]

       
        final_answer = best_rerank_answer if rerank_scores else best_answer[0]

        return {"answer": final_answer}

    except Exception as e:
        logger.exception("Error during question processing")
        raise HTTPException(status_code=500, detail="An error occurred while processing the question.")