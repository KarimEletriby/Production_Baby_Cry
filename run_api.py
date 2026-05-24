"""
Run the FastAPI server.

Usage (development, with auto-reload):
    python run_api.py

Usage (production):
    set ENV=production && python run_api.py
"""
import os
import uvicorn


if __name__ == "__main__":
    env = os.getenv("ENV", "development")
    is_dev = env == "development"

    print(f"🚀 Starting server in {env.upper()} mode")
    print(f"   Auto-reload: {'enabled' if is_dev else 'disabled'}")
    print(f"   URL: http://127.0.0.1:8000")
    print(f"   Docs: http://127.0.0.1:8000/docs")
    print(f"   Health: http://127.0.0.1:8000/health\n")

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=is_dev,
        log_level="info",
        access_log=False,  # we have structured logging instead
    )