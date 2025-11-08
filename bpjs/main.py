"""
BPJS JKN Simple Conversational Assistant
Based on working Transjakarta template - NO OCR, just conversation
"""

import os
import uuid
import json
import sqlite3
from datetime import datetime
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracing
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# FastAPI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================================================================
# STATE
# ============================================================================

class ConversationState(TypedDict):
    messages: List[dict]
    session_id: str
    intent: Optional[str]
    conversation_step: str

# ============================================================================
# NODES
# ============================================================================

def greet_user(state: ConversationState) -> ConversationState:
    """Welcome message"""
    greeting = "👋 Halo! Selamat datang di Asisten JKN BPJS Kesehatan.\n\nAda yang bisa saya bantu?"

    state["messages"].append({
        "role": "assistant",
        "content": greeting,
        "quick_replies": [
            {"text": "📝 Daftar JKN", "action": "registration", "type": "button"},
            {"text": "💰 Pengajuan Klaim", "action": "claim", "type": "button"},
            {"text": "📢 Pengaduan", "action": "complaint", "type": "button"},
            {"text": "❓ Tanya JKN", "action": "ask", "type": "button"}
        ]
    })
    state["conversation_step"] = "awaiting_query"
    return state

def handle_general_query(state: ConversationState) -> ConversationState:
    """Handle any user query"""
    user_messages = [m for m in state["messages"] if m["role"] == "user"]
    if not user_messages:
        state["conversation_step"] = "awaiting_query"
        return state

    last_message = user_messages[-1]["content"]

    # Use LLM to generate response
    prompt = f"""You are a helpful BPJS JKN assistant. Answer in Indonesian.

User question: {last_message}

Provide a helpful, concise answer about JKN BPJS Kesehatan."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
    except:
        content = "Maaf, saya tidak bisa memahami pertanyaan Anda. Bisa dijelaskan lebih detail?"

    state["messages"].append({
        "role": "assistant",
        "content": content,
        "quick_replies": [
            {"text": "📝 Daftar JKN", "action": "registration", "type": "button"},
            {"text": "💰 Klaim", "action": "claim", "type": "button"},
            {"text": "📢 Pengaduan", "action": "complaint", "type": "button"}
        ]
    })
    state["conversation_step"] = "awaiting_query"
    return state

# ============================================================================
# GRAPH
# ============================================================================

def route_entry(state: ConversationState) -> str:
    step = state.get("conversation_step", "init")
    if step == "init":
        return "greet"
    else:
        return "handle_general_query"

workflow = StateGraph(ConversationState)
workflow.add_node("greet", greet_user)
workflow.add_node("handle_general_query", handle_general_query)
workflow.set_conditional_entry_point(route_entry)
workflow.add_edge("greet", END)
workflow.add_edge("handle_general_query", END)

memory = MemorySaver()
conversational_agent = workflow.compile(checkpointer=memory)

print("✅ LangGraph agent ready")

# ============================================================================
# FASTAPI
# ============================================================================

app = FastAPI(title="BPJS JKN Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

class MessageRequest(BaseModel):
    message: Optional[str] = ""
    session_id: Optional[str] = None
    action: Optional[str] = None
    value: Optional[str] = None

@app.post("/api/chat")
async def chat(request: MessageRequest):
    """Handle chat"""
    try:
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in sessions:
            sessions[session_id] = {
                "messages": [],
                "session_id": session_id,
                "intent": None,
                "conversation_step": "init"
            }

        state = sessions[session_id]

        # Handle input
        if request.action:
            if request.action in ["registration", "claim", "complaint", "ask"]:
                state["messages"].append({"role": "user", "content": request.value or request.action})
                state["conversation_step"] = "awaiting_query"
        else:
            if request.message:
                state["messages"].append({"role": "user", "content": request.message})
                state["conversation_step"] = "awaiting_query"

        # Run agent
        config = {"configurable": {"thread_id": session_id}}
        result = conversational_agent.invoke(state, config)
        sessions[session_id] = result

        # Get response
        assistant_messages = [m for m in result["messages"] if m["role"] == "assistant"]
        last_msg = assistant_messages[-1] if assistant_messages else {"role": "assistant", "content": "Halo!"}

        return {
            "session_id": session_id,
            "response": last_msg
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "session_id": session_id if 'session_id' in locals() else str(uuid.uuid4()),
            "response": {
                "role": "assistant",
                "content": f"Maaf, terjadi kesalahan: {str(e)}"
            }
        }

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return HTML_TEMPLATE

# ============================================================================
# HTML - BASED ON WORKING TRANSJAKARTA TEMPLATE
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BPJS JKN AI Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #00a651;
            --secondary-color: #005a2c;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #00a651 0%, #005a2c 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
            height: 90vh;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 20px;
            text-align: center;
        }

        .chat-header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: bold;
        }

        .chat-header p {
            margin: 5px 0 0 0;
            opacity: 0.9;
            font-size: 14px;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.assistant {
            justify-content: flex-start;
        }

        .message-bubble {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
        }

        .user .message-bubble {
            background: var(--secondary-color);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .assistant .message-bubble {
            background: white;
            color: #212529;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .quick-replies {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .quick-reply-btn {
            background: white;
            border: 2px solid var(--primary-color);
            color: var(--primary-color);
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }

        .quick-reply-btn:hover {
            background: var(--primary-color);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 166, 81, 0.3);
        }

        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            border-color: var(--primary-color);
        }

        .send-btn {
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .send-btn:hover {
            background: var(--secondary-color);
            transform: scale(1.1);
        }

        .typing-indicator {
            display: none;
            padding: 10px;
            background: white;
            border-radius: 18px;
            width: fit-content;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🏥 BPJS JKN AI Assistant</h1>
            <p>Asisten Virtual JKN BPJS Kesehatan</p>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="typing-indicator" id="typingIndicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>

        <div class="chat-input-container">
            <div class="input-group">
                <input
                    type="text"
                    class="chat-input"
                    id="messageInput"
                    placeholder="Ketik pesan Anda..."
                    onkeypress="handleKeyPress(event)"
                >
                <button class="send-btn" onclick="sendMessage()">
                    ➤
                </button>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let sessionId = null;

        // Initialize chat
        window.onload = async function() {
            await sendMessage('', 'init');
        };

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        async function sendMessage(message = null, action = null, value = null) {
            const input = document.getElementById('messageInput');
            const messageText = message || input.value.trim();

            if (!messageText && !action) return;

            // Add user message to UI (if not init)
            if (messageText && action !== 'init') {
                addUserMessage(messageText);
                input.value = '';
            }

            // Show typing
            showTypingIndicator();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: messageText || '',
                        session_id: sessionId,
                        action: action,
                        value: value
                    })
                });

                const data = await response.json();
                sessionId = data.session_id;

                hideTypingIndicator();
                addAssistantMessage(data.response);

            } catch (error) {
                console.error('Error:', error);
                hideTypingIndicator();
                addAssistantMessage({
                    content: 'Maaf, terjadi kesalahan. Silakan coba lagi.',
                    role: 'assistant'
                });
            }
        }

        function addUserMessage(text) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user';
            messageDiv.innerHTML = `
                <div class="message-bubble">${escapeHtml(text)}</div>
            `;
            messagesDiv.appendChild(messageDiv);
            scrollToBottom();
        }

        function addAssistantMessage(message) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';

            let content = `<div class="message-bubble">${formatMessage(message.content)}`;

            // Add quick replies
            if (message.quick_replies) {
                content += createQuickRepliesHTML(message.quick_replies);
            }

            content += '</div>';
            messageDiv.innerHTML = content;
            messagesDiv.appendChild(messageDiv);
            scrollToBottom();
        }

        function createQuickRepliesHTML(replies) {
            let html = '<div class="quick-replies">';
            replies.forEach(reply => {
                html += `<button class="quick-reply-btn" onclick="handleQuickReply('${reply.action}', '${escapeHtml(reply.value || reply.text)}')">${reply.text}</button>`;
            });
            html += '</div>';
            return html;
        }

        async function handleQuickReply(action, value) {
            await sendMessage(value, action, value);
        }

        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'block';
            scrollToBottom();
        }

        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }

        function scrollToBottom() {
            const messagesDiv = document.getElementById('chatMessages');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function formatMessage(text) {
            // Convert markdown bold
            text = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            // Convert newlines
            text = text.replace(/\\n/g, '<br>');
            return text;
        }

        function escapeHtml(text) {
            const map = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            };
            return String(text).replace(/[&<>"']/g, m => map[m]);
        }
    </script>
</body>
</html>
"""

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 BPJS JKN AI Assistant (Simple Version)")
    print("=" * 60)
    print("✅ LangGraph agent ready")
    print("=" * 60)
    print("📍 Open http://localhost:8000")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
