#!/usr/bin/env python3
"""
Configuration Demo Script

This script demonstrates how the exposed configuration defaults work.
It shows what values are currently set and how changing them affects the system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import Configuration
from src.history.conversation_utils import estimate_response_tokens


def main():
    """Demonstrate configuration capabilities."""
    print("🔧 MCP Backend Configuration Demo")
    print("=" * 50)
    
    # Load configuration
    config = Configuration()
    
    # Show context management configuration
    print("\n📋 Context Management Configuration:")
    context_config = config.get_context_config()
    print(f"  • Maximum context window: {context_config['max_tokens']} tokens")
    print(f"  • Reserved tokens for response: {context_config['reserve_tokens']} tokens")
    print(f"  • Short conversation limit: {context_config['conversation_limits']['short']} tokens")
    print(f"  • Medium conversation limit: {context_config['conversation_limits']['medium']} tokens")
    print(f"  • Long conversation limit: {context_config['conversation_limits']['long']} tokens")
    print(f"  • Preserve recent messages: {context_config['preserve_recent']} messages")
    
    # Show response token estimation
    print("\n🎯 Response Token Estimation:")
    response_config = context_config['response_tokens']
    print(f"  • Short response: {response_config['short']} tokens")
    print(f"  • Medium response: {response_config['medium']} tokens") 
    print(f"  • Long response: {response_config['long']} tokens")
    print(f"  • Maximum response: {response_config['max']} tokens")
    
    # Show streaming configuration
    print("\n🌊 Streaming Configuration:")
    streaming_config = config.get_streaming_config()
    print(f"  • Streaming enabled: {streaming_config.get('enabled', 'Not set')}")
    
    # Show tool notifications configuration
    print("\n🔧 Tool Notifications Configuration:")
    tool_config = config.get_tool_notifications_config()
    print(f"  • Notifications enabled: {tool_config.get('enabled', 'Not set')}")
    print(f"  • Show arguments: {tool_config.get('show_args', 'Not set')}")
    print(f"  • Icon: {tool_config.get('icon', 'Not set')}")
    print(f"  • Format: {tool_config.get('format', 'Not set')}")
    
    # Show logging configuration  
    print("\n📝 Chat Service Logging Configuration:")
    chat_config = config.get_chat_service_config()
    logging_config = chat_config.get('logging', {})
    print(f"  • System prompt logging: {logging_config.get('system_prompt', 'Not set')}")
    print(f"  • LLM replies logging: {logging_config.get('llm_replies', 'Not set')}")
    print(f"  • Tool execution logging: {logging_config.get('tool_execution', 'Not set')}")
    print(f"  • Tool results logging: {logging_config.get('tool_results', 'Not set')}")
    
    # Show max tool hops
    print("\n🔄 Tool Execution Configuration:")
    max_hops = config.get_max_tool_hops()
    print(f"  • Maximum tool hops: {max_hops}")
    
    # Show MCP connection configuration
    print("\n🔌 MCP Connection Configuration:")
    mcp_config = config.get_mcp_connection_config()
    print(f"  • Max reconnect attempts: {mcp_config['max_reconnect_attempts']}")
    print(f"  • Initial reconnect delay: {mcp_config['initial_reconnect_delay']}s")
    print(f"  • Max reconnect delay: {mcp_config['max_reconnect_delay']}s")
    print(f"  • Connection timeout: {mcp_config['connection_timeout']}s")
    print(f"  • Ping timeout: {mcp_config['ping_timeout']}s")
    
    # Demonstrate response estimation with different conversation lengths
    print("\n🧮 Response Estimation Demo:")
    test_conversations = [
        [{"role": "user", "content": "Hi"}],  # Short
        [{"role": "user", "content": "Tell me about " + "AI " * 50}],  # Medium  
        [{"role": "user", "content": "Explain " + "complex topics " * 100}],  # Long
    ]
    
    for i, conv in enumerate(test_conversations):
        estimated = estimate_response_tokens(
            conv,
            context_config['conversation_limits'],
            context_config['response_tokens']
        )
        conv_type = ["short", "medium", "long"][i]
        print(f"  • {conv_type.capitalize()} conversation: {estimated} tokens estimated")
    
    print("\n✅ All configurations loaded successfully!")
    print("\nTo customize these values, edit src/config.yaml and restart the server.")


if __name__ == "__main__":
    main()
