"""
main.py – Entry point for the AI Financial Risk & Compliance Monitor.

Usage:
    python main.py
    # Then open http://localhost:8003
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8003,
        reload=False,
        log_level="info",
    )
