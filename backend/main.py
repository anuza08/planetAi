from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz 
from langchain_community.llms import OpenAI  
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

origins = [
   "http://localhost:5175",
    "http://localhost:3000",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary storage for PDFs and metadata

documents = {}
class QuestionRequest(BaseModel):
    document_id: int
    question: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the PDF and extract text
    pdf_text = ""
    with fitz.open(stream=await file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()
    
    doc_id = len(documents) + 1
    documents[doc_id] = pdf_text
    return {"document_id": doc_id}

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    print(f"Received document_id: {request.document_id}, question: {request.question}")  # Debugging line
    pdf_text = documents.get(request.document_id, "")
    
    if not pdf_text:
        return {"error": "Document not found."}

    try:
        # Initialize the text splitter and LLM
        splitter = RecursiveCharacterTextSplitter()  # Initialize the text splitter
        text_chunks = splitter.split_text(pdf_text)  # Split the PDF text into manageable chunks
        
        # Initialize the LLM with necessary parameters
        llm = OpenAI()  # Make sure to set your API key or any required parameters
        
        # Create the LLMChain
        chain = LLMChain(llm=llm, prompt="Ask your question.")

        # Construct the input for the chain
        inputs = {
            "question": request.question,
            "text_chunks": text_chunks
        }

        # Generate the answer
        answer = await chain.apredict(inputs)
        
        return {"answer": answer}
    except Exception as e:
        print(f"Error: {e}")  # Log the error
        return {"error": "An error occurred."}
