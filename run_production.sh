#!/bin/bash
# Production script to run the MCP platform with uvicorn configuration

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Setting up with uv..."
    uv venv
    uv sync
fi

# Activate virtual environment
source .venv/bin/activate

# Run with production uvicorn settings that override the application's internal uvicorn config
# Note: The application internally configures uvicorn through the WebSocket server,
# but this provides an alternative way to run with explicit uvicorn CLI options

echo "Starting MCP Platform with production uvicorn configuration..."
echo "- Backlog: 2048 (socket queue for accepting connections)"
echo "- Workers: 1 (required for WebSocket state management)"
echo "- Max connections handled by application-level limits"

# Option 1: Use the application's built-in server (recommended)
echo "Using application's built-in WebSocket server with configured limits..."
uv run mcp-platform

# Option 2: Alternative - Direct uvicorn with module (uncomment to use)
# echo "Using direct uvicorn with CLI options..."
# uv run uvicorn src.main:app \
#     --host localhost \
#     --port 8000 \
#     --backlog 2048 \
#     --workers 1 \
#     --no-access-log \
#     --timeout-keep-alive 5 \
#     --limit-max-requests 1000 \
#     --loop uvloop \
#     --http h11
