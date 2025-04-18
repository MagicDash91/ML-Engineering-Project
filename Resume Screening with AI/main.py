from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, logging
from typing import List
from fpdf import FPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import markdown  # Import markdown module

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
gemini_model = "gemini-2.0-flash"  # or your preferred Gemini model

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resume Matcher</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="bg-light">
        <div class="container py-5">
            <h2 class="mb-4">Resume Matcher</h2>
            <form action="/analyze/" enctype="multipart/form-data" method="post">
                <div class="mb-3">
                    <label class="form-label">Job Description</label>
                    <textarea class="form-control" name="prompt" rows="6" required></textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label">Upload Resumes (PDF/DOCX)</label>
                    <input class="form-control" type="file" name="files" multiple required>
                </div>
                
                <button type="submit" class="btn btn-primary">Analyze</button>
            </form>
            
            <!-- Placeholder for results -->
            <div id="results"></div>
        </div>
    </body>
    </html>
    """

@app.post("/analyze/")
async def analyze_resumes(
    files: List[UploadFile] = File(...),
    prompt: str = Form(...),
):
    task_id = str(uuid.uuid4())
    logging.info(f"Starting resume analysis task {task_id}")

    os.makedirs("static", exist_ok=True)
    file_paths = []
    for file in files:
        file_path = os.path.join("static", f"{task_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_paths.append(file_path)

    # === Gemini Setup ===
    api_key = "**********************************"
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=api_key)

    # Load documents
    docs = []
    for file_path in file_paths:
        loader = UnstructuredFileLoader(file_path)
        docs.extend(loader.load())

    # Prompt template
    prompt = PromptTemplate.from_template(f"""
        You are an expert at analyzing and interpreting resumes in context of a job description.
        
        Job Description:
        {prompt}

        Resume Content:
        {{text}}

        Instructions:
        - Analyze how well the resume matches the job description.
        - Identify relevant skills, experiences, and education.
        - Provide a score from 1 to 10 based on the match quality.
        - Include a short explanation for the score.

        Provide the final score and explanation clearly.
    """)

    results = []
    for doc in docs:
        llm_chain = LLMChain(llm=llm, prompt=prompt.partial(job=prompt))
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        output = stuff_chain.invoke([doc])
        results.append(output["output_text"])

    # Clean up uploaded files
    for path in file_paths:
        os.remove(path)

    # Convert results to HTML using markdown
    html_results = "<h3 class='mb-3'>Candidate Rankings</h3><ul class='list-group'>"
    for idx, res in enumerate(results, start=1):
        markdown_content = res  # Assuming results are in Markdown format
        html_content = markdown.markdown(markdown_content)  # Convert to HTML
        html_results += f"<li class='list-group-item'><b>Candidate {idx}</b><br>{html_content}</li>"
    html_results += "</ul><br><a href='/' class='btn btn-secondary'>Back</a>"

    # Return form with results below it
    return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Resume Matcher Results</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        </head>
        <body class="bg-light">
            <div class="container py-5">
                <h2 class="mb-4">Resume Matcher</h2>
                <form action="/analyze/" enctype="multipart/form-data" method="post">
                    <div class="mb-3">
                        <label class="form-label">Job Description</label>
                        <textarea class="form-control" name="prompt" rows="6" required>{prompt}</textarea>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Upload Resumes (PDF/DOCX)</label>
                        <input class="form-control" type="file" name="files" multiple required>
                    </div>
                    <button type="submit" class="btn btn-primary">Analyze</button>
                </form>

                <!-- Results placed below the form -->
                <div class="mt-5">{html_results}</div>
            </div>
        </body>
        </html>
    """)

if __name__ == "__main__":
    import uvicorn
    import asyncio

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
        timeout_keep_alive=600,
        log_level="info",
        access_log=True,
    )
