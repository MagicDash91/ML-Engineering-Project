from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub

from datetime import date
from contextlib import redirect_stdout
import io
import matplotlib.pyplot as plt
import os
import base64
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_html_form(response_output="", console_output="", img_data="", selected_db=""):
    return f"""
    <html>
    <head>
        <title>Chat With Your Database</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #f7f9fc;
                color: #333;
                max-width: 900px;
                margin: auto;
                padding: 2rem;
                position: relative;
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
            }}
            .section {{
                background-color: #fff;
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            }}
            .form-group {{
                margin-bottom: 1.2rem;
            }}
            label {{
                display: block;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }}
            input, select, textarea {{
                width: 100%;
                padding: 0.6rem;
                border: 1px solid #ccc;
                border-radius: 8px;
            }}
            button {{
                background-color: white;
                color: #333;
                border: 1px solid #ccc;
                padding: 0.8rem 1.5rem;
                border-radius: 12px;
                cursor: pointer;
                font-size: 1rem;
                margin: 0.3rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                transition: all 0.3s ease;
            }}
            button:hover {{
                background-color: #f0f0f0;
                transform: translateY(-1px);
            }}
            .output-box {{
                background-color: #eef;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;
                white-space: pre-wrap;
                font-family: monospace;
            }}
            .db-modal {{
                position: fixed;
                top: 10%;
                left: 50%;
                transform: translate(-50%, 0);
                background: #fff;
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
                z-index: 1000;
                display: none;
                width: 70%;
                max-width: 700px;
            }}
            .db-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1.2rem;
                margin-top: 1rem;
            }}
            .db-option {{
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 1rem;
                background-color: #f9f9f9;
                border-radius: 12px;
                cursor: pointer;
                transition: background-color 0.3s ease;
                border: 1px solid #ddd;
            }}
            .db-option:hover {{
                background-color: #e6f2ff;
            }}
            .db-button-icon {{
                width: 40px;
                height: 40px;
                margin-bottom: 0.5rem;
            }}
            .overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.5);
                display: none;
                z-index: 999;
            }}
        </style>
    </head>
    <body>
        <h1>🧠 Chat With Your Database</h1>

        <form action="/ask" method="post" class="section">
            <div class="form-group">
                <label>Select Your Database</label>
                <button type="button" onclick="openDBModal()">Choose Database</button>
                <input type="hidden" id="db_type" name="db_type" value="{selected_db}">
            </div>

            <div class="form-group">
                <label>Host:</label>
                <input type="text" name="host" value="34.16.80.156" required>
            </div>
            <div class="form-group">
                <label>Port:</label>
                <input type="number" name="port" value="5432" required>
            </div>
            <div class="form-group">
                <label>Database Name:</label>
                <input type="text" name="dbname" value="Sales" required>
            </div>
            <div class="form-group">
                <label>User:</label>
                <input type="text" name="user" value="postgres" required>
            </div>
            <div class="form-group">
                <label>Password:</label>
                <input type="password" name="password" value="villaastern2" required>
            </div>
            <div class="form-group">
                <label>Your Question:</label>
                <textarea name="user_question" rows="5" required>What are the total sales by product?</textarea>
            </div>
            <div style="text-align:center; margin-top:1rem;">
                <button type="submit">Submit Query</button>
            </div>
        </form>

        {"<div class='section'><h2>📜 Agent Response</h2><div class='output-box'>" + response_output + "</div></div>" if response_output else ""}
        {"<div class='section'><h2>💻 Console Output</h2><div class='output-box'>" + console_output + "</div></div>" if console_output else ""}
        {f'<div class="section"><h2>📊 Chart</h2><img src="data:image/png;base64,{img_data}" style="max-width:100%; border-radius: 12px;"></div>' if img_data else ""}

        <div class="overlay" id="overlay" onclick="closeDBModal()"></div>
        <div class="db-modal" id="dbModal">
            <h2>Select a Database</h2>
            <div class="db-grid">
                <div class="db-option" onclick="setDBType('postgresql')">
                    <img src="/static/postgre.png" class="db-button-icon" alt="PostgreSQL"><span>PostgreSQL</span>
                </div>
                <div class="db-option" onclick="setDBType('mysql')">
                    <img src="/static/mysql.avif" class="db-button-icon" alt="MySQL"><span>MySQL</span>
                </div>
                <div class="db-option" onclick="setDBType('snowflake')">
                    <img src="/static/snowflake.png" class="db-button-icon" alt="Snowflake"><span>Snowflake</span>
                </div>
                <div class="db-option" onclick="setDBType('bigquery')">
                    <img src="/static/bigquery.png" class="db-button-icon" alt="BigQuery"><span>BigQuery</span>
                </div>
                <div class="db-option" onclick="setDBType('oracle')">
                    <img src="/static/oracle.png" class="db-button-icon" alt="Oracle"><span>Oracle</span>
                </div>
                <div class="db-option" onclick="setDBType('sqlserver')">
                    <img src="/static/sqlserver.png" class="db-button-icon" alt="SQL Server"><span>SQL Server</span>
                </div>
                <div class="db-option" onclick="setDBType('mongodb')">
                    <img src="/static/mongodb.svg" class="db-button-icon" alt="MongoDB"><span>MongoDB</span>
                </div>
                <div class="db-option" onclick="setDBType('duckdb')">
                    <img src="/static/duckdb.png" class="db-button-icon" alt="DuckDB"><span>DuckDB</span>
                </div>
            </div>
        </div>

        <script>
            function openDBModal() {{
                document.getElementById("overlay").style.display = "block";
                document.getElementById("dbModal").style.display = "block";
            }}
            function closeDBModal() {{
                document.getElementById("overlay").style.display = "none";
                document.getElementById("dbModal").style.display = "none";
            }}
            function setDBType(type) {{
                document.getElementById('db_type').value = type;
                closeDBModal();
                alert('Selected DB: ' + type);
            }}
        </script>
    </body>
    </html>
    """




@app.get("/", response_class=HTMLResponse)
async def read_form():
    return HTMLResponse(content=get_html_form())

@app.post("/ask", response_class=HTMLResponse)
async def ask_db(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    dbname: str = Form(...),
    user: str = Form(...),
    password: str = Form(...),
    user_question: str = Form(...)
):
    try:
        if db_type == "postgresql":
            uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        elif db_type == "mysql":
            uri = f"mysql://{user}:{password}@{host}:{port}/{dbname}"
        elif db_type == "snowflake":
            uri = f"snowflake://{user}:{password}@{host}/{dbname}"
        else:
            return HTMLResponse(content="<h3>Unsupported DB type</h3>", status_code=400)

        db = SQLDatabase.from_uri(uri)

        tools = [
            Tool(name="Query DB", func=QuerySQLDataBaseTool(db=db).run, description="Query the DB"),
            Tool(name="Database Schema Info", func=InfoSQLDatabaseTool(db=db).run, description="Get schema info"),
            Tool(name="Python REPL", func=PythonREPLTool().run, description="Run Python code")
        ]

        instructions = f"""
        You are a database expert. Today is {date.today()}.
        You ONLY answer questions by writing and executing SQL queries on the provided database.
        The database name is {dbname}.
        Always use the available tools: first get table names and schema if needed, then query.
        Avoid guessing.
        If unsure, use the schema info tool.
        """

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key="AIzaSyAMAYxkjP49QZRCg21zImWWAu7c3YHJ0a8"
        )

        prompt = hub.pull("langchain-ai/react-agent-template").partial(instructions=instructions)
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = agent_executor.invoke({
                "input": user_question,
                "chat_history": []
            })

        response_output = result["output"]
        console_output = buffer.getvalue()

        fig = plt.gcf()
        img_data = ""
        if fig.get_axes():
            plt.savefig("temp_plot.png")
            with open("temp_plot.png", "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            os.remove("temp_plot.png")
            plt.clf()

        return HTMLResponse(content=get_html_form(response_output, console_output, img_data, selected_db=db_type))

    except Exception as e:
        return HTMLResponse(content=f"<h3>Error:</h3><pre>{str(e)}</pre>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=8000, reload=True)
