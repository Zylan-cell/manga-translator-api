import argparse
import uvicorn
import os

def main():
    parser = argparse.ArgumentParser(description="Manga Processing API")
    parser.add_argument("--host", default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", 8000)))
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (dev)")

    args = parser.parse_args()

    print("-" * 30)
    print(f"API Server starting on http://{args.host}:{args.port}")
    print("-" * 30)

    uvicorn.run(
        "app.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()