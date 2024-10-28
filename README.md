# 🌌 Q&A Planet
<img src="[https://github.com/user-attachments/assets/949a21cf-6e9e-48d0-bc7c-8873e0e70987](https://github.com/user-attachments/assets/b276b74b-c6be-49e2-9f41-39b655f615fa)" alt="Screenshot 2024-10-27 212211" width="500"/>
<img src="https://github.com/user-attachments/assets/949a21cf-6e9e-48d0-bc7c-8873e0e70987" alt="Screenshot 2024-10-27 212211" width="500"/>
<img src="https://github.com/user-attachments/assets/34c9f975-7401-409e-9479-eb04caf239c1" alt="Screenshot 2024-10-27 212211" width="500"/>
<img src="https://github.com/user-attachments/assets/3cae8018-02c1-4700-bd3b-acf4918a922a" alt="Screenshot 2024-10-27 212211" width="500"/>

**Q&A Planet** is a AI platform where you can upload any document and ask question related to it

## 🛠 Technologies used 

### Frontend Framework
- **ReactJs**: For building the user interface.
- **VueJs**: For additional components and interactivity.

### Backend Framework
- **FastAPI**: For building the API efficiently.
- **Python**: The programming language used for the backend.

### Data Validation and Parsing
- **Pydantic**: For validating request and response data.

### CORS Handling
- **CORS Middleware from FastAPI**: To enable cross-origin resource sharing.

### File Handling
- **PyMuPDF**: For processing PDF files.

### Text Processing
- **LangChain**: For text splitting and handling large documents.
- **Hugging Face Transformers**: For natural language processing tasks.
- **Sentence Transformers**: For embedding sentences and retrieving contextual information.

### Database Management
- **SQLAlchemy ORM**: For database interactions and object-relational mapping.

## 📥 Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.
Copy the proejct url

<h4>Frontend</h4>

```bash
cd ./frontend
```
```bash
npm install
```
and the frontend server will start

<h4>Backend Server</h4>

```bash
cd ./backend
```

To install all the dependencies
```bash
pip install fastapi uvicorn sqlalchemy pymupdf langchain transformers sentence-transformers fastapi[all]
```
To run the backend server 
```bash
uvicorn main:app --reload
```

the server will run at port 
```bash
http://127.0.0.1:8000/docs#/
```

## 📡 API Documentation
## BASE URL 
http://localhost:8000

<h3> 🚀 Endpoints</h3>

### 1. Upload PDF Document

- **Endpoint:** `/upload_pdf`
- **Method:** `POST`
- **Description:** Upload a PDF file to the server for processing.

#### Request

- **Headers:**
  - `Content-Type`: `multipart/form-data`

- **Body:**
  - `file`: The PDF file to be uploaded (required).

#### Response

- **Status Code:** `200 OK`
- **Response Body:**
  ```json
  {
    "document_id": "string" // The ID of the uploaded document.
  }


### 2. Ask a Question

- **Endpoint:** `/ask_question`
- **Method:** `POST`
- **Description:** Submit a question related to the uploaded document and receive an answer.

#### Request

- **Headers:**
  - `Content-Type`: `application/json`

- **Body:**
  - `document_id`: The ID of the document to ask questions about.
  - `question`: The question to be asked.
 
  #### Response
- **Status Code:** `200 OK`
- **Response Body:**
  ```json
  {
    "answer": "string" // The answer to the asked question.
  }

### 3. Get Document Questions

- **Endpoint:** `/document/{document_id}/questions`
- **Method:** `GET`
- **Description:** Retrieve a list of questions and answers associated with a specific document.

#### Request

- **Path Parameters:**
  - `document_id`: The ID of the document for which questions are retrieved (required).

  #### Response
- **Status Code:** `200 OK`
- **Response Body:**
  ```json
  [
    {
      "question": "string", // The question asked.
       "answer": "string"     // The answer provided.
     }
  ] 
