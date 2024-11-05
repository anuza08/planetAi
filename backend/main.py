from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import logging
from sqlalchemy.orm import Session
from database import engine, get_db
from models import Document, QuestionAnswer


Document.metadata.create_all(bind=engine)


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

class QuestionsAndAnswersResponse(BaseModel):
    question: str
    answer: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), title: str = None, db: Session = Depends(get_db)):
    global last_document_id  
    pdf_text = ""
    print(file)
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
      
        document = Document(title=title if title else "Untitled Document", text=pdf_text)
        db.add(document)
        db.commit()
        db.refresh(document)
       
        last_document_id = document.id
        documents[last_document_id] = pdf_text
        logger.info(f"Document uploaded with ID: {last_document_id}")
        return {"document_id": last_document_id}
    else:
        logger.warning("No text extracted from PDF.")
        raise HTTPException(status_code=400, detail="No text found in the PDF document.")

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")
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
        # Split the text into chunks as needed
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        text_chunks = splitter.split_text(pdf_text)

        if not text_chunks:
            logger.error("No chunks created from PDF text.")
            raise HTTPException(status_code=500, detail="No text chunks created from the document.")

        # Use the larger model to get a detailed answer
        answer_texts = []
        for chunk in text_chunks:
            result = qa_pipeline(f"Question: {request.question}\nContext: {chunk}")
            answer_texts.append(result[0]["generated_text"])  # Store generated answers from each chunk

        # Combine results and potentially rerank or summarize if needed
        final_answer = " ".join(answer_texts)  # Optionally, further process to select the best responses

        # Save the QA pair in the database
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
        documents = db.query(Document).all()

        response = []
        for doc in documents:
            qa_list = [
                {"question": qa.question, "answer": qa.answer}
                for qa in doc.question_answers
            ]
            response.append({
                "document_id": doc.id,
                "title": doc.title,
                "text": doc.text,  
                "questions_answers": qa_list,
            })

        return response
    except Exception as e:
        logger.exception("Error fetching documents")
        raise HTTPException(status_code=500, detail="An error occurred while fetching documents.")
