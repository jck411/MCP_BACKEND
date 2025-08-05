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
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError(
                    f"Config file must be YAML dict, got {type(config)}"
                )
            return config

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

    def get_websocket_concurrency_config(self) -> dict[str, Any]:
        """Get WebSocket concurrency configuration from YAML.

        Returns:
            WebSocket concurrency configuration dictionary with validated values.

        Raises:
            ValueError: If required concurrency parameters are missing or invalid.
        """
        websocket_config = self.get_websocket_config()
        concurrency_config = websocket_config.get("concurrency", {})

        # Required configuration keys
        required_keys = ["max_connections", "connection_queue_size", "uvicorn"]
        for key in required_keys:
            if key not in concurrency_config:
                raise ValueError(
                    f"websocket.concurrency.{key} must be explicitly configured "
                    "in config.yaml under chat.websocket.concurrency"
                )

        # Validate uvicorn sub-configuration
        uvicorn_config = concurrency_config["uvicorn"]
        uvicorn_required = [
            "backlog", "workers", "access_log", "keepalive_timeout",
            "max_keepalive_requests", "h11_max_incomplete_event_size",
            "h11_max_request_line_size", "h11_max_header_size"
        ]
        for key in uvicorn_required:
            if key not in uvicorn_config:
                raise ValueError(
                    f"websocket.concurrency.uvicorn.{key} must be explicitly "
                    "configured in config.yaml"
                )

        # Validate configuration values
        max_connections = concurrency_config["max_connections"]
        queue_size = concurrency_config["connection_queue_size"]
        backlog = uvicorn_config["backlog"]

        if max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        if queue_size < max_connections:
            raise ValueError("connection_queue_size must be >= max_connections")
        if backlog < queue_size:
            raise ValueError("uvicorn.backlog must be >= connection_queue_size")

        return concurrency_config

    def get_http_client_config(self) -> dict[str, Any]:
        """Get HTTP client configuration for the active LLM provider.

        Returns:
            HTTP client configuration dictionary with validated values.

        Raises:
            ValueError: If required HTTP client parameters are missing or invalid.
        """
        llm_config = self.get_llm_config()
        http_config = llm_config.get("http_client", {})

        # Required configuration keys
        required_keys = [
            "max_connections", "max_keepalive", "keepalive_expiry",
            "connect_timeout", "read_timeout", "write_timeout", "pool_timeout",
            "requests_per_minute", "concurrent_requests", "max_retries",
            "backoff_factor"
        ]

        for key in required_keys:
            if key not in http_config:
                raise ValueError(
                    f"http_client.{key} must be explicitly configured "
                    f"for provider '{self._config['llm']['active']}' in config.yaml"
                )

        # Validate configuration values
        max_conn = http_config["max_connections"]
        max_keepalive = http_config["max_keepalive"]
        concurrent_reqs = http_config["concurrent_requests"]

        if max_conn < 1:
            raise ValueError("http_client.max_connections must be at least 1")
        if max_keepalive > max_conn:
            raise ValueError("http_client.max_keepalive must be <= max_connections")
        if concurrent_reqs > max_conn:
            raise ValueError(
                "http_client.concurrent_requests must be <= max_connections"
            )

        return http_config

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
            Maximum number of tool hops.

        Raises:
            ValueError: If max_tool_hops is not configured or invalid.
        """
        service_config = self.get_chat_service_config()

        if "max_tool_hops" not in service_config:
            raise ValueError(
                "max_tool_hops must be explicitly configured in config.yaml "
                "under chat.service"
            )

        max_hops = service_config["max_tool_hops"]

        # Validate that it's a positive integer
        if not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError("max_tool_hops must be a positive integer")

        return max_hops

    def get_mcp_connection_config(self) -> dict[str, Any]:
        """Get MCP connection configuration from YAML.

        Returns:
            MCP connection configuration dictionary with validated values.

        Raises:
            ValueError: If required connection parameters are missing or invalid.
        """
        mcp_config = self._config.get("mcp", {})
        connection_config = mcp_config.get("connection", {})

        # Required configuration keys
        required_keys = [
            "max_reconnect_attempts",
            "initial_reconnect_delay",
            "max_reconnect_delay",
            "connection_timeout",
            "ping_timeout"
        ]

        # Check all required keys are present
        for key in required_keys:
            if key not in connection_config:
                raise ValueError(
                    f"{key} must be explicitly configured in config.yaml "
                    f"under mcp.connection"
                )

        # Get values without defaults
        max_attempts = connection_config["max_reconnect_attempts"]
        initial_delay = connection_config["initial_reconnect_delay"]
        max_delay = connection_config["max_reconnect_delay"]
        connection_timeout = connection_config["connection_timeout"]
        ping_timeout = connection_config["ping_timeout"]

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
            Context configuration dictionary with validated values.

        Raises:
            ValueError: If required context parameters are missing or invalid.
        """
        service_config = self.get_chat_service_config()
        context_config = service_config.get("context", {})

        # Validate all required configuration sections exist
        self._validate_context_config_structure(context_config)

        # Extract and validate values
        return self._build_context_config_dict(context_config)

    def _validate_context_config_structure(
        self, context_config: dict[str, Any]
    ) -> None:
        """Validate that all required context configuration sections exist."""
        # Check required top-level keys
        required_keys = ["max_tokens", "reserve_tokens", "preserve_recent"]
        for key in required_keys:
            if key not in context_config:
                raise ValueError(
                    f"{key} must be explicitly configured in config.yaml "
                    f"under chat.service.context"
                )

        # Check required nested sections
        nested_sections = ["conversation_limits", "response_tokens"]
        for section in nested_sections:
            if section not in context_config:
                raise ValueError(
                    f"{section} must be explicitly configured in config.yaml "
                    "under chat.service.context"
                )

        # Validate conversation_limits sub-keys
        limits = context_config["conversation_limits"]
        limits_required = ["short", "medium", "long"]
        for key in limits_required:
            if key not in limits:
                raise ValueError(
                    f"conversation_limits.{key} must be explicitly configured "
                    "in config.yaml"
                )

        # Validate response_tokens sub-keys
        response_tokens = context_config["response_tokens"]
        response_required = ["short", "medium", "long", "max"]
        for key in response_required:
            if key not in response_tokens:
                raise ValueError(
                    f"response_tokens.{key} must be explicitly configured "
                    "in config.yaml"
                )

    def _build_context_config_dict(
        self, context_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Build and validate the context configuration dictionary."""
        # Extract values
        max_tokens = context_config["max_tokens"]
        reserve_tokens = context_config["reserve_tokens"]
        preserve_recent = context_config["preserve_recent"]

        limits = context_config["conversation_limits"]
        response_tokens = context_config["response_tokens"]

        # Validate ranges and relationships
        self._validate_context_config_values(
            max_tokens, reserve_tokens, preserve_recent, limits, response_tokens
        )

        return {
            "max_tokens": max_tokens,
            "reserve_tokens": reserve_tokens,
            "conversation_limits": {
                "short": limits["short"],
                "medium": limits["medium"],
                "long": limits["long"],
            },
            "response_tokens": {
                "short": response_tokens["short"],
                "medium": response_tokens["medium"],
                "long": response_tokens["long"],
                "max": response_tokens["max"],
            },
            "preserve_recent": preserve_recent,
        }

    def _validate_context_config_values(
        self,
        max_tokens: int,
        reserve_tokens: int,
        preserve_recent: int,
        limits: dict[str, Any],
        response_tokens: dict[str, Any],
    ) -> None:
        """Validate context configuration values are within acceptable ranges."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        if preserve_recent < 0:
            raise ValueError("preserve_recent must be non-negative")

        # Validate conversation limits are in ascending order
        if not (limits["short"] < limits["medium"] < limits["long"]):
            raise ValueError("conversation_limits must be in ascending order")

        # Validate response tokens are in non-decreasing order
        tokens = [
            response_tokens["short"],
            response_tokens["medium"],
            response_tokens["long"],
            response_tokens["max"]
        ]
        if not all(tokens[i] <= tokens[i + 1] for i in range(len(tokens) - 1)):
            raise ValueError("response_tokens must be in non-decreasing order")

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

    def get_streaming_backoff_config(self) -> dict[str, Any]:
        """Get streaming backoff configuration from YAML.

        Returns:
            Streaming backoff configuration dictionary.

        Raises:
            ValueError: If required streaming backoff parameters are missing.
        """
        service_config = self.get_chat_service_config()
        streaming_config = service_config.get("streaming", {})
        backoff_config = streaming_config.get("backoff", {})

        # Required configuration keys
        required_keys = ["max_attempts", "initial_delay", "flush_every_n_deltas"]
        for key in required_keys:
            if key not in backoff_config:
                raise ValueError(
                    f"streaming.backoff.{key} must be explicitly configured "
                    "in config.yaml under chat.service.streaming.backoff"
                )

        return backoff_config

    def get_server_execution_config(self) -> dict[str, Any]:
        """Get server execution configuration from YAML.

        Returns:
            Server execution configuration dictionary.

        Raises:
            ValueError: If required server execution parameters are missing.
        """
        mcp_config = self._config.get("mcp", {})
        server_config = mcp_config.get("server_execution", {})

        # Required configuration keys
        required_keys = ["default_args", "default_env", "require_enabled_flag"]
        for key in required_keys:
            if key not in server_config:
                raise ValueError(
                    f"mcp.server_execution.{key} must be explicitly configured "
                    "in config.yaml"
                )

        return server_config

    def get_chat_store_config(self) -> dict[str, Any]:
        """Get chat store configuration from YAML.

        Returns:
            Chat store configuration dictionary.

        Raises:
            ValueError: If required chat store parameters are missing.
        """
        chat_store_config = self._config.get("chat_store", {})
        visibility_config = chat_store_config.get("visibility", {})

        # Required configuration keys
        required_keys = ["system_updates_visible_to_llm", "default_visible_to_llm"]
        for key in required_keys:
            if key not in visibility_config:
                raise ValueError(
                    f"chat_store.visibility.{key} must be explicitly configured "
                    "in config.yaml"
                )

        return chat_store_config

    def get_repository_config(self) -> dict[str, Any]:
        """Get repository configuration from YAML.

        Returns:
            Repository configuration dictionary with persistence and pool settings.

        Raises:
            ValueError: If required repository parameters are missing.
        """
        chat_config = self._config.get("chat", {})
        service_config = chat_config.get("service", {})
        repo_config = service_config.get("repository", {})

        # Create new dictionary without mutating the original
        result_config = {**repo_config}

        # Require explicit persistence configuration
        if "persistence" not in result_config:
            raise ValueError(
                "repository.persistence must be explicitly configured in config.yaml "
                "under chat.service.repository.persistence"
            )

        # Validate required persistence keys
        persistence_config = result_config["persistence"]
        required_keys = [
            "enabled",
            "retention_policy",
            "max_tokens_per_conversation",
            "max_messages_per_conversation",
            "retention_days",
            "clear_on_startup"
        ]

        for key in required_keys:
            if key not in persistence_config:
                raise ValueError(
                    f"repository.persistence.{key} must be explicitly configured "
                    "in config.yaml"
                )

        # Validate persistence configuration values
        valid_policies = ["token_limit", "message_count", "time_based", "unlimited"]
        if persistence_config["retention_policy"] not in valid_policies:
            raise ValueError(
                "repository.persistence.retention_policy must be one of: "
                f"{valid_policies}"
            )

        # Validate connection pool configuration if present
        if "connection_pool" in result_config:
            pool_config = result_config["connection_pool"]

            # Validate pool configuration
            if pool_config.get("enable_pooling", True):
                required_pool_keys = [
                    "max_connections",
                    "max_readers",
                    "max_writers",
                    "connection_timeout",
                    "health_check_interval"
                ]

                for key in required_pool_keys:
                    if key not in pool_config:
                        raise ValueError(
                            f"repository.connection_pool.{key} must be explicitly "
                            "configured when enable_pooling is true"
                        )

                # Validate pool sizing constraints
                max_readers = pool_config["max_readers"]
                max_writers = pool_config["max_writers"]
                max_connections = pool_config["max_connections"]

                if max_readers + max_writers > max_connections:
                    raise ValueError(
                        f"connection_pool: max_readers ({max_readers}) + "
                        f"max_writers ({max_writers}) cannot exceed "
                        f"max_connections ({max_connections})"
                    )

                if max_readers < 1:
                    raise ValueError(
                        "connection_pool.max_readers must be at least 1"
                    )

                if max_writers < 1:
                    raise ValueError(
                        "connection_pool.max_writers must be at least 1"
                    )

        return result_config
