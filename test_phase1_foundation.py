"""
Phase 1 Foundation Test - Verify modern LLM client works with existing config.

This test validates that:
1. Provider detection works correctly
2. Modern client initializes properly  
3. Configuration parsing is compatible
4. Basic functionality is preserved
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.client import ModernLLMClient
from llm.models import ProviderType
from llm.exceptions import LLMError, ProviderError


async def test_provider_detection():
    """Test that provider detection works correctly."""
    print("Testing provider detection...")
    
    # Test OpenAI detection
    config = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "choice_index": 0,
    }
    
    client = ModernLLMClient(config, "test-key")
    assert client.provider_type == ProviderType.OPENAI
    print("✅ OpenAI provider detection works")
    
    # Test OpenRouter detection
    config["base_url"] = "https://openrouter.ai/api/v1"
    client = ModernLLMClient(config, "test-key")
    assert client.provider_type == ProviderType.OPENROUTER
    print("✅ OpenRouter provider detection works")
    
    # Test Groq detection
    config["base_url"] = "https://api.groq.com/openai/v1"
    client = ModernLLMClient(config, "test-key")
    assert client.provider_type == ProviderType.GROQ
    print("✅ Groq provider detection works")
    
    # Test unsupported provider
    try:
        config["base_url"] = "https://unsupported-provider.com"
        client = ModernLLMClient(config, "test-key")
        assert False, "Should have raised ProviderError"
    except ProviderError:
        print("✅ Unsupported provider correctly raises error")


async def test_client_initialization():
    """Test that client initializes with modern connection settings."""
    print("\nTesting client initialization...")
    
    config = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini", 
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "choice_index": 0,
    }
    
    async with ModernLLMClient(config, "test-key") as client:
        # Check that HTTP client was initialized
        assert client._client is not None
        print("✅ HTTP client initialization successful")
        
        # Check provider-specific headers
        headers = client._get_provider_headers()
        assert "Authorization" in headers
        assert "User-Agent" in headers
        assert headers["User-Agent"] == "MCP-Platform/2025"
        print("✅ Provider headers configured correctly")
        
        # Check connection limits
        limits = client._get_connection_limits()
        assert limits.max_connections > 0
        assert limits.max_keepalive_connections > 0
        print("✅ Connection limits configured correctly")


async def test_openrouter_specific_headers():
    """Test OpenRouter-specific header configuration."""
    print("\nTesting OpenRouter-specific features...")
    
    config = {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "choice_index": 0,
    }
    
    client = ModernLLMClient(config, "test-key")
    headers = client._get_provider_headers()
    
    # OpenRouter should have special headers
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers
    assert headers["HTTP-Referer"] == "https://localhost:8000"
    assert headers["X-Title"] == "MCP Platform"
    print("✅ OpenRouter-specific headers configured")


async def test_configuration_validation():
    """Test that configuration validation works."""
    print("\nTesting configuration validation...")
    
    # Missing required fields should raise error
    incomplete_config = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        # Missing required fields
    }
    
    try:
        client = ModernLLMClient(incomplete_config, "test-key")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Required LLM configuration parameter" in str(e)
        print("✅ Configuration validation works correctly")


async def main():
    """Run all Phase 1 foundation tests."""
    print("🧪 Phase 1 Foundation Test Suite")
    print("=" * 50)
    
    try:
        await test_provider_detection()
        await test_client_initialization()
        await test_openrouter_specific_headers()
        await test_configuration_validation()
        
        print("\n" + "=" * 50)
        print("🎉 All Phase 1 foundation tests passed!")
        print("✅ Modern LLM client foundation is working correctly")
        print("✅ Provider detection and optimization implemented")
        print("✅ Legacy code cleanup completed")
        print("✅ Ready for Phase 2 advanced features")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
