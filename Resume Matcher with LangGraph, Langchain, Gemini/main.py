import os
import uuid
import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import google.generativeai as genai
import langgraph as lg  # Correctly import LangGraph
import markdown  # Import markdown module
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
gemini_model = "gemini-2.0-flash"  # Or your preferred Gemini model

# Define a function for document loading
from pathlib import Path

def load_documents(file_paths):
    docs = []
    
    # Ensure file_paths is a list and iterate over each file path
    if isinstance(file_paths, list):
        for file_path in file_paths:
            # Ensure file_path is a string and not a HumanMessage
            if isinstance(file_path, str) and Path(file_path).exists():
                try:
                    loader = UnstructuredFileLoader(file_path)
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
            else:
                print(f"Invalid file path detected: {file_path}")
    else:
        print(f"Expected a list of file paths, but got {type(file_paths)}")

    return docs


# Define a function for document analysis
def analyze_documents_func(docs, prompt):
    api_key = "AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8"
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=api_key)

    prompt_template = PromptTemplate.from_template(f"""
        You are an expert at analyzing and interpreting resumes in the context of a job description.
        
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
        llm_chain = LLMChain(llm=llm, prompt=prompt_template.partial(job=prompt))
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        output = stuff_chain.invoke([doc])
        results.append(output["output_text"])
    return results

# LangGraph State Definition
class State(TypedDict):
    messages: list  # These are file paths or docs, not chat messages
    prompt: str

# Initialize LangGraph
graph_builder = StateGraph(State)

# Define task functions for LangGraph nodes
def task_load_documents(state: State):
    file_paths = state["messages"]
    docs = load_documents(file_paths)
    return {"messages": docs, "prompt": state["prompt"]}


def task_analyze_documents(state: State):
    docs = state["messages"]
    prompt = state["prompt"]
    results = analyze_documents_func(docs, prompt)
    return {"messages": results, "prompt": prompt}



def task_combine_results(state: State):
    results = state["messages"]
    html_results = "<h3 class='mb-3'>Candidate Rankings</h3><ul class='list-group'>"
    for idx, res in enumerate(results, start=1):
        html_content = markdown.markdown(res)
        html_results += f"<li class='list-group-item'><b>Candidate {idx}</b><br>{html_content}</li>"
    html_results += "</ul><br><a href='/' class='btn btn-secondary'>Back</a>"
    return {"messages": html_results, "prompt": state["prompt"]}

# Add nodes to the graph
graph_builder.add_node("load_documents", task_load_documents)
graph_builder.add_node("analyze_documents", task_analyze_documents)
graph_builder.add_node("combine_results", task_combine_results)

# Define the edges for the graph
graph_builder.add_edge(START, "load_documents")
graph_builder.add_edge("load_documents", "analyze_documents")
graph_builder.add_edge("analyze_documents", "combine_results")
graph_builder.add_edge("combine_results", END)

# Compile the graph
graph = graph_builder.compile()

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
        </div>
    </body>
    </html>
    """

@app.post("/analyze/")
async def analyze_resumes_endpoint(
    files: list[UploadFile] = File(...),
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

    # Create a state with the file paths and prompt
    state = {"messages": file_paths, "prompt": prompt}
    
    # Execute the LangGraph
    try:
        result_state = graph.invoke(state)
    except Exception as e:
        logging.exception("Error during LangGraph execution")
        return HTMLResponse(content=f"<h2>Error during resume analysis:</h2><pre>{str(e)}</pre>")


    # Extract and render the results
    html_results = result_state["messages"]

    # Clean up uploaded files
    for path in file_paths:
        os.remove(path)

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
                <div class="mt-5">{html_results}</div>
            </div>
        </body>
        </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000, timeout_keep_alive=600, log_level="info", access_log=True)
