"""
Banking Analytics — Entry Point
================================
Launches the FastAPI web server.

Usage:
    python main.py              → http://localhost:8000
    python main.py --port 9000  → http://localhost:9000
    python main.py --reload     → hot-reload (dev mode)
"""

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Banking Analytics Multi-Agent System")
    parser.add_argument("--host",   default="127.0.0.1", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload for development")
    args = parser.parse_args()

    print(f"\n  Banking Analytics AI Team")
    print(f"  ► http://localhost:{args.port}\n")

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
