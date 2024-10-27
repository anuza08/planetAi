from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz
from langchain_openai import OpenAI
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
last_document_id = None  # Global variable to store the last document ID

class QuestionRequest(BaseModel):
    document_id: int
    question: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global last_document_id  # Use the global variable
    pdf_text = ""
    
    try:
        # Read and open the PDF file
        pdf_content = await file.read()  # Read the uploaded file
        print(f"Uploaded file size: {len(pdf_content)} bytes")  # Log the file size

        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()
                
        print(f"Extracted text length: {len(pdf_text)} characters")  # Log the length of extracted text

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return {"error": "Failed to read the PDF document."}

    if pdf_text:
        doc_id = len(documents) + 1
        documents[doc_id] = pdf_text
        last_document_id = doc_id  # Update the global document ID
        print(f"Document successfully uploaded with ID: {doc_id}")
        print(f"Documents dictionary now contains: {documents}")  # Confirm successful storage
        return {"document_id": doc_id}
    else:
        print("No text extracted from PDF.")
        return {"error": "No text found in PDF document."}


@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    global last_document_id  # Access the global variable
    doc_id = request.document_id or last_document_id  # Use provided ID or fallback to last ID
    pdf_text = documents.get(doc_id)
    
    if pdf_text is None:
        print(f"Document with ID {doc_id} not found in the documents dictionary.")
        print(f"Current documents dictionary: {documents}")  # Log current state for debugging
        return {"error": "Document not found."}

    try:
        # Initialize the text splitter and LLM
        splitter = RecursiveCharacterTextSplitter()  # Initialize the text splitter
        text_chunks = splitter.split_text(pdf_text)  # Split the PDF text into manageable chunks

        # Initialize the LLM with necessary parameters
        llm = OpenAI(openai_api_key="your_openai_api_key")  # Ensure API key is set

        # Create the prompt template and LLMChain
        prompt_template = {"text": "Answer the following question based on provided context."}
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Construct inputs for the chain
        inputs = {
            "question": request.question,
            "text_chunks": text_chunks
        }

        # Generate the answer
        answer = await chain.apredict(inputs)
        
        return {"answer": answer}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": "An error occurred."}
