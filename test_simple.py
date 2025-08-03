#!/usr/bin/env python3

import asyncio
import json
import websockets
import sys

async def test_simple():
    uri = "ws://localhost:8000/ws/chat"
    
    try:
        print("Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")
            
            # Send a test message
            message = {
                "action": "chat",
                "request_id": "test-123",
                "payload": {
                    "text": "Hello, are you working?",
                    "streaming": True,
                    "metadata": {}
                }
            }
            
            print("Sending test message...")
            await websocket.send(json.dumps(message))
            print("✓ Message sent!")
            
            # Wait for first response with timeout
            print("Waiting for response...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"✓ Got response: {response}")
                
                # Try to get one more response
                try:
                    response2 = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"✓ Got second response: {response2}")
                except asyncio.TimeoutError:
                    print("⚠ No second response within timeout")
                    
            except asyncio.TimeoutError:
                print("✗ No response received within 10 seconds")
                return False
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing WebSocket connection...")
    result = asyncio.run(test_simple())
    sys.exit(0 if result else 1)
