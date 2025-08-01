"""Configuration management for the MCP client."""

import json
import os
from typing import Any

import yaml
from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration from YAML and environment variables."""
        self.load_env()  # Load .env for API keys
        self._config = self._load_yaml_config()  # Load YAML config

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    def _load_yaml_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path) as file:
            return yaml.safe_load(file)

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path) as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the API key for the active LLM provider.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        # Get active provider
        llm_config = self._config.get("llm", {})
        active_provider = llm_config.get("active", "openai")

        # Map provider names to environment variable names
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }

        env_key = provider_key_map.get(active_provider)
        if not env_key:
            raise ValueError(
                f"Unknown provider '{active_provider}' - no API key mapping found"
            )

        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(
                f"API key '{env_key}' not found in environment variables "
                f"for provider '{active_provider}'"
            )

        return api_key

    def get_config_dict(self) -> dict[str, Any]:
        """Get the full configuration dictionary for websocket server.

        Returns:
            The complete configuration dictionary.
        """
        return self._config

    def get_llm_config(self) -> dict[str, Any]:
        """Get active LLM provider configuration from YAML.

        Returns:
            Active LLM provider configuration dictionary.
        """
        llm_config = self._config.get("llm", {})
        active_provider = llm_config.get("active", "openai")
        providers = llm_config.get("providers", {})

        if active_provider not in providers:
            raise ValueError(
                f"Active provider '{active_provider}' not found in providers config"
            )

        return providers[active_provider]

    def get_websocket_config(self) -> dict[str, Any]:
        """Get WebSocket configuration from YAML.

        Returns:
            WebSocket configuration dictionary.
        """
        return self._config.get("chat", {}).get("websocket", {})

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration from YAML.

        Returns:
            Logging configuration dictionary.
        """
        return self._config.get("logging", {})

    def get_chat_service_config(self) -> dict[str, Any]:
        """Get chat service configuration from YAML.

        Returns:
            Chat service configuration dictionary.
        """
        return self._config.get("chat", {}).get("service", {})

    def get_max_tool_hops(self) -> int:
        """Get the maximum number of tool hops allowed.

        Returns:
            Maximum number of tool hops (default: 8).
        """
        service_config = self.get_chat_service_config()
        max_hops = service_config.get("max_tool_hops", 8)

        # Validate that it's a positive integer
        if not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError("max_tool_hops must be a positive integer")

        return max_hops

    def get_mcp_connection_config(self) -> dict[str, Any]:
        """Get MCP connection configuration from YAML.

        Returns:
            MCP connection configuration dictionary with validated defaults.
        """
        mcp_config = self._config.get("mcp", {})
        connection_config = mcp_config.get("connection", {})

        # Get values with defaults
        max_attempts = connection_config.get("max_reconnect_attempts", 5)
        initial_delay = connection_config.get("initial_reconnect_delay", 1.0)
        max_delay = connection_config.get("max_reconnect_delay", 30.0)
        connection_timeout = connection_config.get("connection_timeout", 30.0)
        ping_timeout = connection_config.get("ping_timeout", 10.0)

        # Validate configuration values
        if max_attempts < 1:
            raise ValueError("max_reconnect_attempts must be at least 1")
        if initial_delay <= 0:
            raise ValueError("initial_reconnect_delay must be positive")
        if max_delay < initial_delay:
            raise ValueError("max_reconnect_delay must be >= initial_reconnect_delay")
        if connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if ping_timeout <= 0:
            raise ValueError("ping_timeout must be positive")

        return {
            "max_reconnect_attempts": max_attempts,
            "initial_reconnect_delay": initial_delay,
            "max_reconnect_delay": max_delay,
            "connection_timeout": connection_timeout,
            "ping_timeout": ping_timeout,
        }

    def get_context_config(self) -> dict[str, Any]:
        """Get context management configuration from YAML.

        Returns:
            Context configuration dictionary with validated defaults.
        """
        service_config = self.get_chat_service_config()
        context_config = service_config.get("context", {})

        # Get values with defaults
        max_tokens = context_config.get("max_tokens", 4000)
        reserve_tokens = context_config.get("reserve_tokens", 500)

        # Conversation limits
        limits = context_config.get("conversation_limits", {})
        short_limit = limits.get("short", 100)
        medium_limit = limits.get("medium", 500)
        long_limit = limits.get("long", 1500)

        # Response token estimates
        response_tokens = context_config.get("response_tokens", {})
        short_response = response_tokens.get("short", 150)
        medium_response = response_tokens.get("medium", 300)
        long_response = response_tokens.get("long", 500)
        max_response = response_tokens.get("max", 800)

        # Optimization settings
        preserve_recent = context_config.get("preserve_recent", 5)

        # Validate configuration values
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        if not (short_limit < medium_limit < long_limit):
            raise ValueError("conversation_limits must be in ascending order")
        if not (short_response <= medium_response <= long_response <= max_response):
            raise ValueError("response_tokens must be in non-decreasing order")
        if preserve_recent < 0:
            raise ValueError("preserve_recent must be non-negative")

        return {
            "max_tokens": max_tokens,
            "reserve_tokens": reserve_tokens,
            "conversation_limits": {
                "short": short_limit,
                "medium": medium_limit,
                "long": long_limit,
            },
            "response_tokens": {
                "short": short_response,
                "medium": medium_response,
                "long": long_response,
                "max": max_response,
            },
            "preserve_recent": preserve_recent,
        }

    def get_streaming_config(self) -> dict[str, Any]:
        """Get streaming configuration from YAML.

        Returns:
            Streaming configuration dictionary.
        """
        service_config = self.get_chat_service_config()
        return service_config.get("streaming", {})

    def get_tool_notifications_config(self) -> dict[str, Any]:
        """Get tool notification configuration from YAML.

        Returns:
            Tool notification configuration dictionary.
        """
        service_config = self.get_chat_service_config()
        return service_config.get("tool_notifications", {})
