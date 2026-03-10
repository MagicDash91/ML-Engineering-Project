"""
main.py – Entry point for the combined Bank Analytics + Digital Marketing system.

Usage:
    python main.py
    # Then open http://localhost:8002
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
        log_level="info",
    )
