#!/usr/bin/env python3

import asyncio
import json
import websockets

async def test_websocket():
    uri = "ws://localhost:8000/ws/chat"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            
            # Send a test message
            message = {
                "action": "chat",
                "request_id": "test-123",
                "payload": {
                    "text": "Hello, can you help me?",
                    "streaming": True,
                    "metadata": {}
                }
            }
            
            print(f"Sending message: {json.dumps(message, indent=2)}")
            await websocket.send(json.dumps(message))
            
            # Wait for responses
            response_count = 0
            async for response in websocket:
                response_count += 1
                print(f"Response {response_count}: {response}")
                
                # Parse the response
                try:
                    data = json.loads(response)
                    if data.get("status") == "complete":
                        print("Received completion signal")
                        break
                    elif data.get("status") == "error":
                        print(f"Received error: {data.get('chunk', {}).get('error')}")
                        break
                except json.JSONDecodeError:
                    print(f"Invalid JSON response: {response}")
                
                # Safety timeout
                if response_count > 20:
                    print("Too many responses, breaking...")
                    break
                    
    except Exception as e:
        print(f"WebSocket connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
