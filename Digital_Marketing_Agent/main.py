"""
main.py – CLI entry point for Digital Marketing Agent.
Usage:
    python main.py                # http://localhost:8001
    python main.py --port 9001    # custom port
    python main.py --reload       # hot-reload dev mode
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Digital Marketing Multi-Agent System")
    parser.add_argument("--host",   default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",   default=8001, type=int, help="Bind port (default: 8001)")
    parser.add_argument("--reload", action="store_true",  help="Enable hot-reload (dev mode)")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   Digital Marketing Multi-Agent System       ║")
    print("  ║   Powered by Gemini 2.5 Flash + Veo 3        ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"  ► http://localhost:{args.port}")
    print()

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
