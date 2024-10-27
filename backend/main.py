from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import logging
from sqlalchemy.orm import Session
from database import SessionLocal, engine, get_db
from models import Document, QuestionAnswer

# Create the database tables
Document.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), 
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger(__name__) 

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
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

# In-memory storage for documents
documents = {}
last_document_id = None  

class QuestionRequest(BaseModel):
    document_id: int = None  
    question: str

class QuestionsAndAnswersResponse(BaseModel):
    question: str
    answer: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), title: str = None, db: Session = Depends(get_db)):
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
        # Save to database
        document = Document(title=title if title else "Untitled Document", text=pdf_text)
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Use the ID from the database instead of a simple counter
        last_document_id = document.id
        documents[last_document_id] = pdf_text
        logger.info(f"Document uploaded with ID: {last_document_id}")
        return {"document_id": last_document_id}
    else:
        logger.warning("No text extracted from PDF.")
        raise HTTPException(status_code=400, detail="No text found in the PDF document.")

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
rerank_model = SentenceTransformer('all-MiniLM-L6-v2')  

@app.post("/ask_question")
async def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    global last_document_id

    doc_id = request.document_id or last_document_id
    logger.debug(f"Received request with document_id: {doc_id} and question: {request.question}")

    if not doc_id or doc_id not in documents:
        available_ids = list(documents.keys())
        logger.error(f"Document with ID {doc_id} not found. Available IDs: {available_ids}")
        raise HTTPException(status_code=404, detail=f"Document not found. Available document IDs: {available_ids}")

    pdf_text = documents[doc_id]

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
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

        # Save the question and answer to the database
        qa_entry = QuestionAnswer(document_id=doc_id, question=request.question, answer=final_answer)
        db.add(qa_entry)
        db.commit()
        db.refresh(qa_entry)

        return {"answer": final_answer}

    except Exception as e:
        logger.exception("Error during question processing")
        raise HTTPException(status_code=500, detail="An error occurred while processing the question.")

@app.get("/document/{document_id}/questions", response_model=list[QuestionsAndAnswersResponse])
def get_questions_answers(document_id: int, db: Session = Depends(get_db)):
    questions_answers = db.query(QuestionAnswer).filter(QuestionAnswer.document_id == document_id).all()

    if not questions_answers:
        raise HTTPException(status_code=404, detail="No questions found for this document.")

    return [{"question": qa.question, "answer": qa.answer} for qa in questions_answers]

@app.get("/documents")
async def get_all_documents(db: Session = Depends(get_db)):
    try:
        # Query all documents from the database
        documents = db.query(Document).all()

        # Prepare a response with documents and their question-answers
        response = []
        for doc in documents:
            qa_list = [
                {"question": qa.question, "answer": qa.answer}
                for qa in doc.question_answers
            ]
            response.append({
                "document_id": doc.id,
                "title": doc.title,
                "text": doc.text,  # Optionally include the text
                "questions_answers": qa_list,
            })

        return response
    except Exception as e:
        logger.exception("Error fetching documents")
        raise HTTPException(status_code=500, detail="An error occurred while fetching documents.")
