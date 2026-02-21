import uvicorn


def main() -> None:
    """Start the FastAPI server."""
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()

