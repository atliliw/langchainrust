"""Thin Python client for the langchainrust OpenAI-compatible server."""

from .client import ChatResult, Client, LangChainRustError, ToolCall

__all__ = ["ChatResult", "Client", "LangChainRustError", "ToolCall"]
__version__ = "0.31.0"
